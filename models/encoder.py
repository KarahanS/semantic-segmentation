from typing import Optional
import timm
import torch
import torch.nn as nn
from timm.layers import (
    resample_patch_embed,
    resample_abs_pos_embed,
    resample_abs_pos_embed_nhwc,
)
from timm.models._manipulate import checkpoint_seq
from torch.nn.functional import interpolate


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_name,
        img_size: tuple[int, int],
        ckpt_path,
        sub_norm,
        patch_size,
        pretrained,
    ):
        super().__init__()

        model_kwargs = {
            "model_name": encoder_name,
            "pretrained": pretrained,
            "num_classes": 0,
        }
        self.encoder = timm.create_model(**model_kwargs)
        
        
        """ timm forward_features
        def forward_features(self, x):
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                # dynamically resize abs pos embedding if needed
                x = x + resample_abs_pos_embed_nhwc(self.pos_embed, x.shape[1:3])
            x = self.pos_drop(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                x = self.blocks(x)
            x = self.neck(x.permute(0, 3, 1, 2))
            return x
        """ 
        
        ckpt_state = torch.load(ckpt_path, map_location="cpu") if ckpt_path else {}
        has_obj_in_ckpt = "obj_token" in ckpt_state
        
        if not hasattr(self.encoder, "obj_token") and has_obj_in_ckpt:             
            # (a) parameter
            self.encoder.obj_token = nn.Parameter(
                torch.zeros_like(self.encoder.cls_token)
            )
            # tell the model it now has two prefix tokens
            self.encoder.num_prefix_tokens = 2

            # make sure pos_embed has 2 prefix rows
            if self.encoder.pos_embed.shape[1] == 197:          # 1 + 196
                cls_row = self.encoder.pos_embed[:, :1]
                patch_rows = self.encoder.pos_embed[:, 1:]
                self.encoder.pos_embed = nn.Parameter(
                    torch.cat([cls_row, cls_row, patch_rows], dim=1)
                )                                               # 2 + 196 = 198

            # ---------------------------------------------------------------------
            # monkey‑patch forward_features only once
            if not hasattr(self.encoder, "_with_obj"):
                def _ff_obj(this, x, attn_mask=None):
                    B = x.shape[0]
                    x = this.patch_embed(x)                     # (B, N, D)

                    # prepend CLS & OBJ exactly once
                    cls_ = this.cls_token.expand(B, -1, -1)
                    obj = this.obj_token.expand(B, -1, -1)
                    x = torch.cat((cls_, obj, x), dim=1)         # (B, 2+N, D)

                    # add positional embeddings & dropout
                    x = x + this.pos_embed
                    x = this.pos_drop(x)
                    x = this.patch_drop(x)
                    x = this.norm_pre(x)

                    if attn_mask is not None:
                        for blk in this.blocks:
                            x = blk(x, attn_mask=attn_mask)
                    elif this.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint_seq(this.blocks, x)
                    else:
                        x = this.blocks(x)

                    return this.norm(x)

                self.encoder.forward_features = _ff_obj.__get__(self.encoder, self.encoder.__class__)
                self.encoder._with_obj = True
            # ---------------------------------------------------------------------
            
        pixel_mean = torch.tensor(self.encoder.default_cfg["mean"]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(self.encoder.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self.grid_size = tuple(round(size / patch_size) for size in img_size)

        self.embed_dim = (
            self.encoder.embed_dim
            if hasattr(self.encoder, "embed_dim")
            else self.encoder.num_features
        )
        

        if sub_norm:
            for block in self.encoder.blocks:
                new_mlp = type(block.mlp)(
                    in_features=block.mlp.fc1.in_features,
                    hidden_features=block.mlp.fc1.out_features,
                    act_layer=type(block.mlp.act),
                    drop=block.mlp.drop1.p,
                    norm_layer=nn.LayerNorm,
                )
                new_mlp.load_state_dict(block.mlp.state_dict(), strict=False)
                block.mlp = new_mlp
                block.attn.proj = nn.Sequential(
                    nn.LayerNorm(block.attn.proj.in_features), block.attn.proj
                )

        if hasattr(self.encoder, "neck"):
            self.encoder.neck = nn.Identity()

        if ckpt_path:
            self.encoder.load_state_dict(torch.load(ckpt_path))

        if hasattr(self.encoder, "rope"):
            self.encoder.rope = timm.create_model(
                img_size=img_size, patch_size=patch_size, **model_kwargs
            ).rope

        if hasattr(self.encoder, "blocks"):
            for block in self.encoder.blocks:
                old_window_size = None
                if hasattr(block, "window_size"):
                    old_window_size = block.window_size
                    window_ratio = (
                        old_window_size / self.encoder.patch_embed.grid_size[0]
                    )
                    new_window_size = window_ratio * (img_size[0] / patch_size)

                    if new_window_size != round(new_window_size):
                        raise ValueError("invalid window size")

                    block.window_size = int(new_window_size)

                if hasattr(block.attn, "rel_pos_h"):
                    block.attn.rel_pos_h = self.interpolate_rel_pos(
                        block.attn.rel_pos_h,
                        img_size[0] / patch_size,
                        self.encoder.patch_embed.grid_size[0],
                        block.window_size,
                        old_window_size,
                    )

                if hasattr(block.attn, "rel_pos_w"):
                    block.attn.rel_pos_w = self.interpolate_rel_pos(
                        block.attn.rel_pos_w,
                        img_size[1] / patch_size,
                        self.encoder.patch_embed.grid_size[1],
                        block.window_size,
                        old_window_size,
                    )

        if hasattr(self.encoder, "patch_embed"):
            if (
                self.encoder.patch_embed.grid_size[0]
                != self.encoder.patch_embed.grid_size[1]
                or self.encoder.patch_embed.patch_size[0]
                != self.encoder.patch_embed.patch_size[1]
            ):
                raise ValueError("pretrained grid and patch size must be square")

            self.encoder.patch_embed.patch_size = (patch_size, patch_size)
            self.encoder.patch_embed.proj.kernel_size = (patch_size, patch_size)
            self.encoder.patch_embed.proj.stride = (patch_size, patch_size)
            self.encoder.patch_embed.proj.weight = nn.Parameter(
                resample_patch_embed(
                    self.encoder.patch_embed.proj.weight,
                    [patch_size, patch_size],
                )
            )

            self.encoder.patch_embed.grid_size = self.grid_size
            self.encoder.patch_embed.num_patches = self.grid_size[0] * self.grid_size[1]
            self.encoder.patch_embed.img_size = img_size

        if hasattr(self.encoder, "pos_embed"):
            if self.encoder.pos_embed.dim() == 4:
                pos_embed = resample_abs_pos_embed_nhwc(
                    self.encoder.pos_embed, [max(self.grid_size), max(self.grid_size)]
                )[:, : self.grid_size[0], : self.grid_size[1], :]
            else:
                num_prefix_tokens = (
                    0
                    if getattr(self.encoder, "no_embed_class", False)
                    else self.encoder.num_prefix_tokens
                )
                pos_embed = resample_abs_pos_embed(
                    self.encoder.pos_embed,
                    [
                        max(self.grid_size),
                        max(self.grid_size),
                    ],
                    num_prefix_tokens=num_prefix_tokens,
                )
                prefix_pos_embed = pos_embed[:, :num_prefix_tokens, :]
                pos_embed = pos_embed[:, num_prefix_tokens:, :]
                pos_embed = pos_embed.reshape(
                    1, max(self.grid_size), max(self.grid_size), -1
                )[:, : self.grid_size[0], : self.grid_size[1], :]
                pos_embed = torch.cat(
                    [prefix_pos_embed, pos_embed.flatten(1, 2)], dim=1
                )

            self.encoder.pos_embed = nn.Parameter(pos_embed)

    @staticmethod
    def interpolate_rel_pos(
        rel_pos, grid_size, old_grid_size, window_size=None, old_window_size=None
    ):
        block_size = (rel_pos.shape[0] + 1) / 2

        if block_size == old_grid_size:
            max_rel_dist = grid_size * 2 + 1
        elif block_size == old_window_size:
            if window_size is None:
                raise ValueError("window_size must be specified for non-global blocks")

            max_rel_dist = window_size * 2 + 1
        else:
            raise ValueError("invalid block size")

        max_rel_dist = int(max_rel_dist)

        rel_pos = rel_pos.reshape(1, rel_pos.shape[0], -1)
        rel_pos = rel_pos.permute(0, 2, 1)
        rel_pos = interpolate(rel_pos, size=max_rel_dist, mode="linear")
        rel_pos = rel_pos.reshape(-1, max_rel_dist).permute(1, 0)

        return nn.Parameter(rel_pos)

    def forward(self, x: torch.Tensor):
        x = (x - self.pixel_mean) / self.pixel_std

        x = self.encoder.forward_features(x)

        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x[:, self.encoder.num_prefix_tokens :]

        return x
