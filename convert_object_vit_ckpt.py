import torch, sys, re
from pathlib import Path

ckpt_in  = Path("/scratch/work/saritak1/checkpoints/coco_cribo_v137_Lnn_Lo_Lp/checkpoint.pth")               # original .pth
ckpt_out = ckpt_in.with_suffix(".timm.pth")

raw = torch.load(ckpt_in, map_location="cpu")

# 1) pick teacher weights if they exist
for key in ("teacher", "teacher_state_dict", "teacher_backbone", "model_teacher"):
    if key in raw:
        raw = raw[key]
        break

# 2) strip wrapper prefixes only
clean = {}
for k, v in raw.items():
    for p in ("teacher.", "module.", "backbone.", "model."):
        if k.startswith(p):
            k = k[len(p):]
    clean[k] = v

# 3) keep ViT tensors (INC obj_token, NOT dist_token)
vit_keys = ("patch_embed", "blocks", "cls_token", "obj_token", "pos_embed", "norm")
clean = {k: v for k, v in clean.items() if k.startswith(vit_keys)}

torch.save(clean, ckpt_out)
print("✓ wrote", ckpt_out)
