# Standalone forecaster inference for the demo: runs under venv_glacier specifically
# (venv_labeling carries a different torch version),
# invoked as a subprocess from src/foretrack/demo_pipeline.py exactly like labeling/run_tapip3d.py
# already does for TAPIP3D's separate env. Communicates via npz on disk, no GT/baselines needed
# (unlike scripts/eval.py, which this borrows its model-loading code from).

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from foretrack.data.dexycb import load_transl_stats
from foretrack.data.transforms import denormalize_translation, normalize_translation
from foretrack.eval.ood_gate import DISAGREEMENT_THRESHOLD_CM, conditioned_disagreement
from foretrack.models.conditioning import ImageEncoder, QueryTokenizer
from foretrack.models.denoiser import TrackDenoiser
from foretrack.models.diffusion import GaussianDiffusion


def load_config(path: str) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        base_cfg = load_config(str(path.parent / cfg.pop("defaults")))
        merged = dict(base_cfg)
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        cfg = merged
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--input_npz", required=True, help="image (H,W,3 uint8), query_uv (N,2), query_xyz_t0 (N,3) raw metric, orig_h, orig_w")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.setdefault("DOWNLOADS_DIR", str(Path(__file__).resolve().parent.parent / "downloads"))
    cfg = load_config(args.config)
    m = cfg["model"]
    n, t_len = m["num_query_points"], m["num_timesteps"]

    data = np.load(args.input_npz)
    image = data["image"]  # (H, W, 3) uint8, RGB, ALREADY cropped to CROP_TARGET_SIZE by the caller
    query_uv = data["query_uv"].astype(np.float32)  # (N, 2) pixel coords in the cropped frame
    query_xyz_t0_raw = data["query_xyz_t0"].astype(np.float32)  # (N, 3) raw metric, camera(0) frame
    orig_h, orig_w = int(data["orig_h"]), int(data["orig_w"])

    ds_cfg = cfg["data"]["datasets"][cfg["dataset"].split("+")[0]]
    transl_mean, transl_std = load_transl_stats(ds_cfg["transl_stats"])

    from foretrack.data.dexycb import IMAGENET_MEAN, IMAGENET_STD
    image_t = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    image_t = torch.from_numpy(image_t.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    query_uv_t = torch.from_numpy(query_uv).float().unsqueeze(0).to(device)
    query_xyz_t0_norm = normalize_translation(query_xyz_t0_raw, transl_mean, transl_std)
    query_xyz_t0_t = torch.from_numpy(query_xyz_t0_norm).float().unsqueeze(0).to(device)

    ckpt = torch.load(args.diffusion_ckpt, map_location=device, weights_only=True)
    image_encoder = ImageEncoder(vit_init=m["vit_init"]).to(device).eval()
    image_encoder.load_state_dict(ckpt["image_encoder"])
    query_tokenizer = QueryTokenizer(latent_dim=m["latent_dim"], dropout_prob=0.0, vit_feat_dim=1280).to(device).eval()
    query_tokenizer.load_state_dict(ckpt["query_tokenizer"])
    query_tokenizer_uncond = QueryTokenizer(latent_dim=m["latent_dim"], dropout_prob=0.0, vit_feat_dim=1280, disable_query_cond=True).to(device).eval()
    net = TrackDenoiser(n=n, t=t_len, latent_dim=m["latent_dim"], num_layers=m["num_layers"], num_heads=m["num_heads"]).to(device)
    net.load_state_dict(ckpt["net"])
    net.eval()
    diffusion = GaussianDiffusion(num_steps=cfg["diffusion"]["num_steps"])

    with torch.no_grad():
        patch_feats = image_encoder(image_t)
        cond = query_tokenizer(query_xyz_t0_t, patch_feats, query_uv_t, orig_image_size=(orig_h, orig_w)).transpose(0, 1)
        cond_uncond = query_tokenizer_uncond(query_xyz_t0_t, patch_feats, query_uv_t, orig_image_size=(orig_h, orig_w)).transpose(0, 1)

    shape = (1, t_len, n, 3)
    schedule = list(range(diffusion.num_steps - 1, -1, -diffusion.num_steps // args.num_inference_steps))
    if schedule[-1] != 0:
        schedule.append(0)

    def sample(c):
        x_t = torch.randn(shape, device=device)
        for i, t_val in enumerate(schedule):
            t = torch.full((1,), t_val, device=device, dtype=torch.long)
            t_prev_val = schedule[i + 1] if i + 1 < len(schedule) else -1
            t_prev = torch.full((1,), t_prev_val, device=device, dtype=torch.long)
            x_t = diffusion.p_sample(net, x_t, t, c, query_xyz_t0_t, t_prev=t_prev, eta=args.eta)
        return x_t

    samples_norm = []
    with torch.no_grad():
        for _ in range(args.samples):
            samples_norm.append(sample(cond)[0].cpu().numpy())
        uncond_sample_norm = sample(cond_uncond)[0].cpu().numpy()

    samples = np.stack([denormalize_translation(s, transl_mean, transl_std) for s in samples_norm])  # (S, T, N, 3)
    uncond_sample = denormalize_translation(uncond_sample_norm, transl_mean, transl_std)

    # OOD gate signal: how much conditioning changes the prediction, relative to what
    # in-domain conditioning produces -- reported so the frontend can flag low confidence.
    disagreement_cm = conditioned_disagreement(samples[0], uncond_sample)
    low_confidence = bool(disagreement_cm < DISAGREEMENT_THRESHOLD_CM)

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        samples=samples,  # (S, T, N, 3) metric
        disagreement_cm=disagreement_cm,
        low_confidence=low_confidence,
    )
    print(f"wrote {args.samples} forecast samples to {args.output_npz} (disagreement={disagreement_cm:.2f}cm, low_confidence={low_confidence})")


if __name__ == "__main__":
    main()
