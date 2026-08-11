import glob

import torch

for model_type in ["diffusion", "regressor"]:
    print(f"--- {model_type} ---")
    files = sorted(
        glob.glob(f"downloads/model/foretrack4d/dexycb_stage1/{model_type}/epoch*.pt"),
        key=lambda f: int(f.split("epoch")[-1].split(".")[0]),
    )
    for f in files:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
        epoch = ckpt["epoch"] + 1
        val_loss = ckpt["val_loss"]
        print(f"  epoch {epoch}: val_loss={val_loss:.4f}")
