import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# some cluster nodes fail cuDNN handle creation while plain CUDA kernels work; the only conv
# here is the ViT patch embed, so the fallback costs nothing measurable
if os.environ.get("FORETRACK_DISABLE_CUDNN"):
    torch.backends.cudnn.enabled = False

from foretrack.data.dexycb import load_transl_stats
from foretrack.data.mixed import DATASET_CLASSES
from foretrack.data.transforms import denormalize_translation
from foretrack.eval.baselines import constant_velocity, static
from foretrack.eval.metrics import (
    ade,
    ade_first_frame_aligned,
    ade_global_aligned,
    ade_per_timestep,
    apd3d,
    dataset_diversity,
    diversity,
    fde,
)
from foretrack.eval.ood_gate import gated_prediction
from foretrack.models.conditioning import ImageEncoder, QueryTokenizer
from foretrack.models.denoiser import TrackDenoiser
from foretrack.models.diffusion import GaussianDiffusion
from foretrack.models.regressor import TrackRegressor


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


def forward_cond(image_encoder, query_tokenizer, batch, device):
    images = batch["image"].to(device)
    query_xyz_t0 = batch["query_xyz_t0"].to(device)
    query_uv = batch["query_uv"].to(device)
    orig_h = batch["orig_image_size"][0][0].item()
    orig_w = batch["orig_image_size"][1][0].item()
    patch_feats = image_encoder(images)
    cond = query_tokenizer(query_xyz_t0, patch_feats, query_uv, orig_image_size=(orig_h, orig_w))
    return cond.transpose(0, 1), query_xyz_t0


@torch.no_grad()
def sample_diffusion(net, diffusion, cond, query_xyz_t0, shape, device, num_inference_steps=20, eta=1.0):
    """strided ancestral sampling: only num_inference_steps of the 1000 trained steps are
    actually run, jumping between them (valid for an x0-prediction model, see p_sample's
    docstring) -- full 1000-step sampling is far too slow to run at eval scale. eta=1.0 is the
    standard DDPM ancestral sampler (fully stochastic); eta=0.0 is DDIM's deterministic
    special case -- see p_sample's docstring."""
    schedule = list(range(diffusion.num_steps - 1, -1, -diffusion.num_steps // num_inference_steps))
    if schedule[-1] != 0:
        schedule.append(0)
    x_t = torch.randn(shape, device=device)
    bz = shape[0]
    for i, t_val in enumerate(schedule):
        t = torch.full((bz,), t_val, device=device, dtype=torch.long)
        t_prev_val = schedule[i + 1] if i + 1 < len(schedule) else -1
        t_prev = torch.full((bz,), t_prev_val, device=device, dtype=torch.long)
        x_t = diffusion.p_sample(net, x_t, t, cond, query_xyz_t0, t_prev=t_prev, eta=eta)
    return x_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--diffusion_ckpt", default=None)
    parser.add_argument("--regressor_ckpt", default=None)
    parser.add_argument(
        "--num_batches", type=int, default=None,
        help="subset of val batches, for a quick smoke check. Full val set by default -- a "
        "small fixed subset (formerly a default of 10, ~10% of val) gave numbers that varied by "
        "60%+ depending on which batches happened to be included (val_loader isn't shuffled, so "
        "this stayed silently fixed across many training attempts and produced misleading "
        "beats-baseline/doesn't-beat-baseline verdicts).",
    )
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--samples_per_input", type=int, default=5)
    parser.add_argument(
        "--eta", type=float, default=0.0,
        help="DDIM interpolation for p_sample: 0.0 = fully deterministic sampling (default --"
        "removes per-step injected noise while keeping the initial x_T draw random, which alone "
        "was enough for healthy cross-sample diversity; closed the gap to static/regressor "
        "decisively and consistently across seeds), 1.0 = standard "
        "stochastic DDPM ancestral sampling. Training is unaffected either way -- p_sample is "
        "eval-only, training always regresses directly against true x0.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="regressor/static/const_vel are fully deterministic given a fixed seed + "
        "cudnn.benchmark=False (no RNG use at all) -- varying this only matters for the "
        "diffusion model's actual sampling noise, useful for checking how much a single-seed "
        "diffusion readout can be trusted.",
    )
    parser.add_argument(
        "--split", default="val",
        help="which split to evaluate on. 'val' is what every architecture/hyperparameter "
        "decision in this project was made against (5 training attempts, repeatedly) -- it's no "
        "longer a clean generalization estimate on its own. 'test' is genuinely held out and was "
        "never looked at while iterating; use it for the real, final read before trusting a "
        "result.",
    )
    parser.add_argument(
        "--eval_dataset", default=None,
        help="which single dataset (by name, e.g. 'dexycb') to evaluate on -- eval is always "
        "per-dataset, never a concatenated mix. Defaults to the first name in "
        "cfg['dataset'] (a single-dataset config's only choice).",
    )
    parser.add_argument(
        "--ood_gate", action="store_true",
        help="reports an additional 'diffusion_gated' row: falls back to the static baseline "
        "whenever the diffusion model's conditioned and unconditional (disable_query_cond=True) "
        "predictions nearly agree, the signal that conditioning isn't contributing meaningfully "
        "for this input (see foretrack.eval.ood_gate). Roughly doubles diffusion sampling cost "
        "(one extra unconditional pass per input). Off by default -- this changes what the "
        "system's prediction IS, not just how it's measured, so it's opt-in and reported as its "
        "own row rather than silently replacing 'diffusion'.",
    )
    parser.add_argument(
        "--horizon_plot", default=None,
        help="path to save an error-vs-horizon plot (ForeHand4D Fig. 8 analog) -- per-timestep "
        "ADE for every reported model, nanmean-aggregated across the eval "
        "set (sequences shorter than the horizon contribute NaN past their real length rather "
        "than pulling the average down with padded-frame zeros). Skipped if not given.",
    )
    args = parser.parse_args()

    # found real run-to-run nondeterminism while validating a p_sample fix: two back-to-back
    # eval runs of the SAME regressor checkpoint (a plain forward pass, no RNG use at all) gave
    # ADE 24.54 vs 22.89 -- cudnn's default algorithm-selection heuristic (`benchmark=True`) can
    # pick different, numerically-non-identical kernels between runs. Fixing this so eval numbers
    # can actually be compared/trusted across code changes, not just within a single run.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    cfg = load_config(args.config)
    os.environ.setdefault("DOWNLOADS_DIR", str(Path(__file__).resolve().parent.parent / "downloads"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n, t_len = cfg["model"]["num_query_points"], cfg["model"]["num_timesteps"]
    eval_dataset = args.eval_dataset or cfg["dataset"].split("+")[0]
    ds_cfg = cfg["data"]["datasets"][eval_dataset]
    val_ds = DATASET_CLASSES[eval_dataset](ds_cfg["gt_root"], args.split, n=n, t=t_len, transl_stats_path=ds_cfg["transl_stats"])
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    transl_mean, transl_std = load_transl_stats(ds_cfg["transl_stats"])
    num_batches = args.num_batches if args.num_batches is not None else len(val_loader)
    print(f"val set: {len(val_ds)}, evaluating {num_batches}/{len(val_loader)} batches")

    m = cfg["model"]

    def load_cond(ckpt, disable_query_cond=m.get("disable_query_cond", False)):
        # image_encoder and query_tokenizer are trained jointly with the denoiser/regressor
        # (see train.py's `params = list(image_encoder.parameters()) + ...` and its checkpoint
        # dict's "image_encoder"/"query_tokenizer" keys) -- each checkpoint has its own learned
        # conditioning pathway, not a shared frozen one, so this must be loaded per-checkpoint,
        # not constructed once and reused across diffusion/regressor.
        ie = ImageEncoder(vit_init=m["vit_init"]).to(device).eval()
        ie.load_state_dict(ckpt["image_encoder"])
        qt = QueryTokenizer(
            latent_dim=m["latent_dim"], dropout_prob=0.0, vit_feat_dim=1280,
            disable_query_cond=disable_query_cond,
        ).to(device).eval()
        qt.load_state_dict(ckpt["query_tokenizer"])
        return ie, qt

    diffusion_net, regressor_net, diffusion = None, None, None
    image_encoder_diff, query_tokenizer_diff = None, None
    image_encoder_reg, query_tokenizer_reg = None, None
    if args.diffusion_ckpt:
        ckpt = torch.load(args.diffusion_ckpt, map_location=device, weights_only=True)
        diffusion_net = TrackDenoiser(n=n, t=t_len, latent_dim=m["latent_dim"], num_layers=m["num_layers"], num_heads=m["num_heads"]).to(device)
        diffusion_net.load_state_dict(ckpt["net"])
        diffusion_net.eval()
        image_encoder_diff, query_tokenizer_diff = load_cond(ckpt)
        diffusion = GaussianDiffusion(num_steps=cfg["diffusion"]["num_steps"])
        print(f"loaded diffusion ckpt: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")
    if args.regressor_ckpt:
        ckpt = torch.load(args.regressor_ckpt, map_location=device, weights_only=True)
        regressor_net = TrackRegressor(n=n, t=t_len, latent_dim=m["latent_dim"], num_layers=m["num_layers"], num_heads=m["num_heads"]).to(device)
        regressor_net.load_state_dict(ckpt["net"])
        regressor_net.eval()
        image_encoder_reg, query_tokenizer_reg = load_cond(ckpt)
        print(f"loaded regressor ckpt: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")

    query_tokenizer_uncond = None
    if args.ood_gate and diffusion_net is not None:
        # disable_query_cond always returns zeros regardless of loaded weights (see
        # QueryTokenizer.forward), so no checkpoint state to load here.
        query_tokenizer_uncond = QueryTokenizer(latent_dim=m["latent_dim"], dropout_prob=0.0, vit_feat_dim=1280, disable_query_cond=True).to(device).eval()

    # "diffusion" selects the best of S samples using the ground truth (minADE_S, the standard
    # trajectory-forecasting metric); "diffusion_1" averages the metric over samples instead,
    # which is what a single draw actually delivers with no oracle to pick with. The baselines
    # are deterministic, so only the latter compares like with like.
    model_names = ["static", "const_vel", "regressor", "diffusion", "diffusion_1"]
    if args.ood_gate and diffusion_net is not None:
        model_names.append("diffusion_gated")
    results = {k: [] for k in model_names}
    fde_results = {k: [] for k in model_names}
    ff_results = {k: [] for k in model_names}  # M-F analog: first-frame aligned
    global_results = {k: [] for k in model_names}  # M-G analog: globally aligned
    apd3d_results = {k: [] for k in model_names}
    horizon_curves = {k: [] for k in model_names}  # list of (T,) per-sample arrays, nanmean'd at the end
    diversity_results = []
    pred_motions = []  # displacement from t=0, for dataset-level diversity
    gt_motions = []

    def record_mean_over_samples(name, preds, gt_b, mask_b, fx, fy, visibility_b):
        """expected metric of one random sample, not the metric of an averaged trajectory --
        averaging multimodal trajectories would produce a path the model never predicts."""
        results[name].append(float(np.mean([ade(p, gt_b, mask_b) for p in preds])))
        fde_results[name].append(float(np.mean([fde(p, gt_b, mask_b) for p in preds])))
        ff_results[name].append(float(np.mean([ade_first_frame_aligned(p, gt_b, mask_b) for p in preds])))
        global_results[name].append(float(np.mean([ade_global_aligned(p, gt_b, mask_b) for p in preds])))
        apd3d_results[name].append(float(np.mean([apd3d(p, gt_b, fx, fy, visibility_b, mask_b) for p in preds])))
        if args.horizon_plot:
            horizon_curves[name].append(np.nanmean(np.stack([ade_per_timestep(p, gt_b, mask_b) for p in preds]), axis=0))

    def record(name, pred, gt_b, mask_b, fx, fy, visibility_b):
        results[name].append(ade(pred, gt_b, mask_b))
        fde_results[name].append(fde(pred, gt_b, mask_b))
        ff_results[name].append(ade_first_frame_aligned(pred, gt_b, mask_b))
        global_results[name].append(ade_global_aligned(pred, gt_b, mask_b))
        apd3d_results[name].append(apd3d(pred, gt_b, fx, fy, visibility_b, mask_b))
        if args.horizon_plot:
            horizon_curves[name].append(ade_per_timestep(pred, gt_b, mask_b))

    for bi, batch in enumerate(val_loader):
        if bi >= num_batches:
            break

        gt_norm = batch["tracks"].numpy()
        mask = batch["frame_mask"].numpy()
        query_xyz_t0_norm = batch["query_xyz_t0"].numpy()
        visibility = batch["visibility"].numpy()
        intrinsics = batch["intrinsics"].numpy()  # (B, 3, 3), original (uncropped) camera calibration

        gt = denormalize_translation(gt_norm, transl_mean, transl_std)
        query_xyz_t0 = denormalize_translation(query_xyz_t0_norm, transl_mean, transl_std)

        for b in range(gt.shape[0]):
            m_b = mask[b]
            fx, fy = intrinsics[b, 0, 0], intrinsics[b, 1, 1]
            pred_static = static(query_xyz_t0[b], t_len)
            record("static", pred_static, gt[b], m_b, fx, fy, visibility[b])

            pred_cv = constant_velocity(gt[b, 0], gt[b, 1], t_len)
            record("const_vel", pred_cv, gt[b], m_b, fx, fy, visibility[b])

        if regressor_net is not None:
            with torch.no_grad():
                cond, query_xyz_t0_cond = forward_cond(image_encoder_reg, query_tokenizer_reg, batch, device)
                pred_norm, _ = regressor_net(cond, query_xyz_t0_cond)  # visibility logits unused in eval
                pred_norm = pred_norm.cpu().numpy()
            pred = denormalize_translation(pred_norm, transl_mean, transl_std)
            for b in range(gt.shape[0]):
                record("regressor", pred[b], gt[b], mask[b], intrinsics[b, 0, 0], intrinsics[b, 1, 1], visibility[b])

        if diffusion_net is not None:
            with torch.no_grad():
                cond, query_xyz_t0_cond = forward_cond(image_encoder_diff, query_tokenizer_diff, batch, device)
            shape = (gt.shape[0], t_len, n, 3)
            samples = []
            for _ in range(args.samples_per_input):
                s_norm = sample_diffusion(diffusion_net, diffusion, cond, query_xyz_t0_cond, shape, device, args.num_inference_steps, args.eta)
                samples.append(denormalize_translation(s_norm.cpu().numpy(), transl_mean, transl_std))
            samples = np.stack(samples)  # (S, B, T, N, 3)

            if query_tokenizer_uncond is not None:
                with torch.no_grad():
                    cond_uncond, _ = forward_cond(image_encoder_diff, query_tokenizer_uncond, batch, device)
                s_uncond_norm = sample_diffusion(diffusion_net, diffusion, cond_uncond, query_xyz_t0_cond, shape, device, args.num_inference_steps, args.eta)
                pred_uncond = denormalize_translation(s_uncond_norm.cpu().numpy(), transl_mean, transl_std)

            for b in range(gt.shape[0]):
                fx, fy = intrinsics[b, 0, 0], intrinsics[b, 1, 1]
                per_sample_ade = [ade(samples[s, b], gt[b], mask[b]) for s in range(args.samples_per_input)]
                best = int(np.argmin(per_sample_ade))
                record("diffusion", samples[best, b], gt[b], mask[b], fx, fy, visibility[b])
                record_mean_over_samples("diffusion_1", samples[:, b], gt[b], mask[b], fx, fy, visibility[b])
                diversity_results.append(diversity(samples[:, b]))
                valid = mask[b]
                pred_motions.append(samples[0, b][valid] - samples[0, b][0])
                gt_motions.append(gt[b][valid] - gt[b][0])
                if query_tokenizer_uncond is not None:
                    pred_static_b = static(query_xyz_t0[b], t_len)
                    pred_gated = gated_prediction(samples[best, b], pred_uncond[b], pred_static_b, mask[b])
                    record("diffusion_gated", pred_gated, gt[b], mask[b], fx, fy, visibility[b])

        print(f"batch {bi + 1}/{num_batches} done")

    print("\n=== Results (min-of-samples for diffusion; ADE/FDE/M-F/M-G in cm, APD3D in %) ===")
    for k in model_names:
        if len(results[k]) == 0:
            continue
        print(
            f"{k:12s}  ADE: {np.mean(results[k]):7.2f}   FDE: {np.mean(fde_results[k]):7.2f}   "
            f"M-F: {np.mean(ff_results[k]):7.2f}   M-G: {np.mean(global_results[k]):7.2f}   "
            f"APD3D: {np.mean(apd3d_results[k]):6.2f}   n={len(results[k])}"
        )
    if diversity_results:
        print(f"\nmultimodality (mean pairwise L2 between {args.samples_per_input} samples of the same input, cm): {np.mean(diversity_results):.2f}")
    if len(pred_motions) >= 2:
        n_t = min(m.shape[0] for m in pred_motions)
        pm = np.stack([m[:n_t] for m in pred_motions])
        gm = np.stack([m[:n_t] for m in gt_motions])
        print(f"dataset diversity, predicted: {dataset_diversity(pm):.2f}   ground truth: {dataset_diversity(gm):.2f}")
        print("(closer to the ground-truth value is better -- a much larger value means predictions")
        print(" are scattered rather than distributed like real motion)")

    print("\nnote: const_vel is oracle-ish (uses real GT frame 1) -- not a fair comparison to")
    print("the actual model, which only sees a single frame.")

    if args.horizon_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # the two baselines can track each other almost exactly, so vary linestyle and width:
        # with a single style the later curve hides the earlier one and the plot looks broken
        styles = {
            "static": dict(color="0.25", ls="--", lw=2.4),
            "const_vel": dict(color="0.55", ls=":", lw=2.0),
            "regressor": dict(color="tab:green", ls="-", lw=1.8),
            "diffusion": dict(color="tab:red", ls="-", lw=1.8),
            "diffusion_1": dict(color="tab:purple", ls="-.", lw=1.8),
            "diffusion_gated": dict(color="tab:brown", ls=(0, (5, 1)), lw=1.5),
        }
        labels = {"diffusion": "diffusion (minADE-5)", "diffusion_1": "diffusion (1 sample)"}
        plt.figure(figsize=(7, 5))
        for k in model_names:
            if len(horizon_curves[k]) == 0:
                continue
            curve = np.nanmean(np.stack(horizon_curves[k]), axis=0)  # (T,)
            plt.plot(curve, label=labels.get(k, k), **styles.get(k, {}))
        plt.xlabel("forecast horizon (timestep)")
        plt.ylabel("ADE (cm)")
        plt.title("error vs. horizon")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.horizon_plot, dpi=150)
        print(f"\nsaved error-vs-horizon plot: {args.horizon_plot}")


if __name__ == "__main__":
    main()
