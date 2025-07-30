import torch
from pathlib import Path

# path to the original DINO checkpoint (teacher + student + optimizer, etc.)
ckpt_path = Path("/scratch/work/saritak1/checkpoints/dino_Li/checkpoint.pth")

# ---- 1. load raw checkpoint -------------------------------------------------
raw = torch.load(ckpt_path, map_location="cpu")

# ---- 2. pick the teacher parameters -----------------------------------------
# DINO checkpoints usually store separate dicts for student and teacher.
# Try several common field names; fall back to the whole dict if nothing matches.
for key in ["teacher", "teacher_state_dict", "teacher_backbone", "model_teacher"]:
    if key in raw:
        state_dict = raw[key]
        break
else:  # no explicit teacher key found
    state_dict = raw

# ---- 3. flatten prefixes like 'module.' and 'backbone.' ---------------------
clean_sd = {}
for k, v in state_dict.items():
    # remove up to three common wrapper prefixes
    for prefix in ("teacher.", "module.", "backbone.", "model."):
        if k.startswith(prefix):
            k = k[len(prefix):]
    clean_sd[k] = v

# ---- 4. drop non‑ViT keys ---------------------------------------------------
vit_keys = ("patch_embed", "blocks", "cls_token", "dist_token", "pos_embed", "norm")
clean_sd = {k: v for k, v in clean_sd.items() if k.startswith(vit_keys)}

# ---- 5. remove the distillation token if your model does not use it ---------
clean_sd.pop("dist_token", None)  # harmless if the key is absent

# ---- 6. save in timm‑compatible format --------------------------------------
torch.save(clean_sd, ckpt_path.with_suffix(".pth.timm"))

print(f"✅ Saved timm‑style weights to {ckpt_path.with_suffix('.pth.timm')}")
