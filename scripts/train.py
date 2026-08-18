import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from foretrack.data.mixed import build_dataset
from foretrack.models.conditioning import ImageEncoder, QueryTokenizer
from foretrack.models.denoiser import TrackDenoiser
from foretrack.models.diffusion import GaussianDiffusion, depth_scaled_l2, visibility_bce
from foretrack.models.regressor import TrackRegressor


def worker_init_fn(worker_id: int):
    seed = torch.utils.data.get_worker_info().seed % (2**32)
    np.random.seed(seed)


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        base_cfg = load_config(str(path.parent / cfg.pop("defaults")))
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def build_model(cfg: dict, device: str, model_type: str):
    m = cfg["model"]
    image_encoder = ImageEncoder(vit_init=m["vit_init"]).to(device)
    query_tokenizer = QueryTokenizer(
        latent_dim=m["latent_dim"], dropout_prob=m["query_dropout_prob"], vit_feat_dim=1280,
        disable_query_cond=m.get("disable_query_cond", False),
    ).to(device)
    common_args = dict(
        n=m["num_query_points"], t=m["num_timesteps"], latent_dim=m["latent_dim"],
        num_layers=m["num_layers"], num_heads=m["num_heads"],
    )
    if model_type == "diffusion":
        net = TrackDenoiser(**common_args).to(device)
        diffusion = GaussianDiffusion(num_steps=cfg["diffusion"]["num_steps"])
    else:
        net = TrackRegressor(**common_args).to(device)
        diffusion = None
    return image_encoder, query_tokenizer, net, diffusion


def forward_cond(image_encoder, query_tokenizer, batch, device):
    images = batch["image"].to(device)
    query_xyz_t0 = batch["query_xyz_t0"].to(device)
    query_uv = batch["query_uv"].to(device)
    orig_h = batch["orig_image_size"][0][0].item()
    orig_w = batch["orig_image_size"][1][0].item()

    patch_feats = image_encoder(images)
    cond = query_tokenizer(query_xyz_t0, patch_feats, query_uv, orig_image_size=(orig_h, orig_w))
    return cond.transpose(0, 1), query_xyz_t0  # (num_cond, B, D), transformer's seqlen-first convention


def make_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return lr_lambda


def compute_loss(model_type, net, diffusion, cond, query_xyz_t0, tracks, depth, mask, offset_reg_weight=0.0, depth_scaled=True, visibility_gt=None, alpha_vis=3.0):
    if model_type == "diffusion":

        def model_fn(x_noisy, timestep, cond, query_xyz_t0):
            return net(x_noisy, timestep, cond, query_xyz_t0)

        return diffusion.training_losses(
            model_fn, tracks, cond, query_xyz_t0, depth, mask=mask, offset_reg_weight=offset_reg_weight,
            depth_scaled=depth_scaled, visibility_gt=visibility_gt, alpha_vis=alpha_vis,
        )
    else:
        pred, vis_logits = net(cond, query_xyz_t0)
        loss = depth_scaled_l2(pred, tracks, depth, mask=mask, depth_scaled=depth_scaled)
        if offset_reg_weight > 0:
            offset = pred - query_xyz_t0.unsqueeze(1)
            loss = loss + offset_reg_weight * offset.pow(2).mean()
        if visibility_gt is not None:
            loss = loss + alpha_vis * visibility_bce(vis_logits, visibility_gt, mask)
        return loss


@torch.no_grad()
def validate(image_encoder, query_tokenizer, net, diffusion, model_type, val_loader, device, autocast_dtype, max_batches=10, depth_scaled=True, alpha_vis=3.0):
    image_encoder.eval()
    query_tokenizer.eval()
    net.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=autocast_dtype, enabled=device == "cuda"):
            cond, query_xyz_t0 = forward_cond(image_encoder, query_tokenizer, batch, device)
            tracks = batch["tracks"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["frame_mask"].to(device)
            visibility_gt = batch["visibility"].to(device)
            loss = compute_loss(model_type, net, diffusion, cond, query_xyz_t0, tracks, depth, mask, depth_scaled=depth_scaled, visibility_gt=visibility_gt, alpha_vis=alpha_vis)
        losses.append(loss.item())
    image_encoder.train()
    query_tokenizer.train()
    net.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_type", choices=["diffusion", "regressor"], default="diffusion")
    parser.add_argument("--debug", action="store_true", help="use train_debug overrides (small batch, 1 epoch)")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)  # otherwise piped/redirected output doesn't

    cfg = load_config(args.config)
    if args.debug and "train_debug" in cfg:
        cfg["train"] = deep_merge(cfg["train"], cfg["train_debug"])

    os.environ.setdefault("DOWNLOADS_DIR", str(Path(__file__).resolve().parent.parent / "downloads"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if cfg["train"].get("precision") == "bf16" else torch.float32
    print(f"device: {device}, autocast dtype: {autocast_dtype}, model_type: {args.model_type}")

    n, t_len = cfg["model"]["num_query_points"], cfg["model"]["num_timesteps"]
    noise_factor = float(cfg["train"].get("noise_factor", 0.0))
    scale_factor = float(cfg["train"].get("scale_factor", 0.0))
    offset_reg_weight = float(cfg["train"].get("offset_reg_weight", 0.0))
    depth_scaled = bool(cfg["diffusion"].get("depth_scaled_loss", True))
    use_visibility_loss = bool(cfg["train"].get("use_visibility_loss", True))
    alpha_vis = float(cfg["train"].get(f"alpha_vis_{args.model_type}", cfg["train"].get("alpha_vis", 3.0)))
    print(f"depth_scaled_loss: {depth_scaled}, use_visibility_loss: {use_visibility_loss}, alpha_vis: {alpha_vis}")
    train_ds = build_dataset(cfg, "train", n, t_len, augment=True, noise_factor=noise_factor, scale_factor=scale_factor)
    val_ds = build_dataset(cfg, "val", n, t_len, augment=False)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}, augment: noise={noise_factor}, scale={scale_factor}")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=4, drop_last=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=2)

    image_encoder, query_tokenizer, net, diffusion = build_model(cfg, device, args.model_type)

    init_ckpt = cfg["train"].get(f"init_ckpt_{args.model_type}", cfg["train"].get("init_ckpt"))
    if init_ckpt:
        state = torch.load(init_ckpt, map_location=device)
        image_encoder.load_state_dict(state["image_encoder"])
        query_tokenizer.load_state_dict(state["query_tokenizer"])
        net.load_state_dict(state["net"])
        print(f"initialized from checkpoint: {init_ckpt} (epoch {state['epoch']}, val_loss {state['val_loss']:.4f})")

    params = list(image_encoder.parameters()) + list(query_tokenizer.parameters()) + list(net.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(cfg["train"]["lr"]))

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = int(cfg["train"].get("warmup_frac", 0.05) * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(warmup_steps, total_steps))
    print(f"lr schedule: base_lr={cfg['train']['lr']}, warmup_steps={warmup_steps}, total_steps={total_steps}")
    print(f"offset_reg_weight: {offset_reg_weight}")

    ckpt_dir = Path(cfg["train"]["ckpt_dir"]) / args.model_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    t0 = time.time()
    for epoch in range(cfg["train"]["epochs"]):
        for batch in train_loader:
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=autocast_dtype, enabled=device == "cuda"):
                cond, query_xyz_t0 = forward_cond(image_encoder, query_tokenizer, batch, device)
                tracks = batch["tracks"].to(device)
                depth = batch["depth"].to(device)
                mask = batch["frame_mask"].to(device)
                visibility_gt = batch["visibility"].to(device) if use_visibility_loss else None
                loss = compute_loss(args.model_type, net, diffusion, cond, query_xyz_t0, tracks, depth, mask, offset_reg_weight=offset_reg_weight, depth_scaled=depth_scaled, visibility_gt=visibility_gt, alpha_vis=alpha_vis)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
            optimizer.step()
            scheduler.step()

            if step % cfg["train"]["log_every"] == 0:
                elapsed = time.time() - t0
                cur_lr = scheduler.get_last_lr()[0]
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} lr {cur_lr:.2e} elapsed {elapsed:.0f}s")
            step += 1

        is_last = epoch == cfg["train"]["epochs"] - 1
        if (epoch + 1) % cfg["train"]["val_every_epoch"] == 0 or is_last:
            val_loss = validate(image_encoder, query_tokenizer, net, diffusion, args.model_type, val_loader, device, autocast_dtype, depth_scaled=depth_scaled, alpha_vis=alpha_vis if use_visibility_loss else 0.0)
            print(f"epoch {epoch} val_loss {val_loss:.4f}")
            torch.save(
                {
                    "image_encoder": image_encoder.state_dict(),
                    "query_tokenizer": query_tokenizer.state_dict(),
                    "net": net.state_dict(),
                    "model_type": args.model_type,
                    "epoch": epoch,
                    "step": step,
                    "val_loss": val_loss,
                },
                ckpt_dir / f"epoch{epoch + 1}.pt",
            )
            print(f"saved checkpoint: {ckpt_dir / f'epoch{epoch + 1}.pt'}")


if __name__ == "__main__":
    main()
