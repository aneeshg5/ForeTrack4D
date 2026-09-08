import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from foretrack.data.mixed import DATASET_CLASSES
from foretrack.eval.baselines import static
from foretrack.eval.metrics import ade, fde
from foretrack.models.conditioning import ImageEncoder, QueryTokenizer, fourier_embed
from foretrack.models.denoiser import TrackDenoiser
from foretrack.models.diffusion import GaussianDiffusion


def _deep_merge(base, override):
    # must recurse: a config that overrides only datasets.<name>.gt_root would otherwise drop
    # every sibling key, transl_stats included
    merged = dict(base)
    for k, v in override.items():
        merged[k] = _deep_merge(merged[k], v) if isinstance(v, dict) and isinstance(merged.get(k), dict) else v
    return merged


def load_config(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        cfg = _deep_merge(load_config(str(path.parent / cfg.pop("defaults"))), cfg)
    return cfg


@torch.no_grad()
def sample(net, diffusion, cond, query_xyz_t0, shape, device, steps=20, norm_mean=None, norm_scale=None):
    schedule = list(range(diffusion.num_steps - 1, -1, -diffusion.num_steps // steps))
    if schedule[-1] != 0:
        schedule.append(0)
    x_t = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(0))
    bz = shape[0]
    for i, tv in enumerate(schedule):
        t = torch.full((bz,), tv, device=device, dtype=torch.long)
        tp = torch.full((bz,), schedule[i + 1] if i + 1 < len(schedule) else -1, device=device, dtype=torch.long)
        x_t = diffusion.p_sample(net, x_t, t, cond, query_xyz_t0, t_prev=tp, eta=0.0, norm_mean=norm_mean, norm_scale=norm_scale)
    return x_t


TRAIN_MEAN = np.array([0.316, -0.092, -0.077], dtype=np.float32)
TRAIN_STD = np.array([0.956, 1.042, 1.231], dtype=np.float32)


def spread(x):
    # mean pairwise L2 between rows, the same "how different are these from each other" measure
    # used for dataset diversity, so the numbers are read the same way
    if x.shape[0] < 2:
        return float("nan")
    d = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    iu = np.triu_indices(x.shape[0], k=1)
    return float(d[iu].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_dataset", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--gt_root", default=None, help="override the config gt_root, for the audited clean set")
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--tag", default="")
    p.add_argument("--renorm_mode", default="spread", choices=["spread", "depth"],
        help="'spread' divides by the per-sequence spread of the query points, which also rescales "
        "predicted motion by object extent. 'depth' divides by median query depth instead, keeping "
        "relative scale intact across scenes -- the principled form, and the one computable from "
        "input alone.")
    p.add_argument("--scale_const", default="1,1,1",
        help="global constant applied after depth division, calibrated once in-domain so the "
        "resulting spread matches the training moments.")
    p.add_argument(
        "--renorm", action="store_true",
        help="re-normalize each sequence using only its own t=0 query points, mapped onto the "
        "in-domain training moments, instead of the dataset translation stats. Diagnostic only: "
        "it answers whether restoring the input range restores conditioning use, and is not a "
        "deployable setting.",
    )
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    cfg = load_config(args.config)
    m = cfg["model"]
    n, t_len = m["num_query_points"], m["num_timesteps"]
    ds_cfg = cfg["data"]["datasets"][args.eval_dataset]
    root = args.gt_root or ds_cfg["gt_root"]
    ds = DATASET_CLASSES[args.eval_dataset](root, args.split, n=n, t=t_len, transl_stats_path=ds_cfg.get("transl_stats"), norm_mode=cfg["data"].get("norm_mode", "dataset"))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    net = TrackDenoiser(n=n, t=t_len, latent_dim=m["latent_dim"], num_layers=m["num_layers"], num_heads=m["num_heads"]).to(device)
    net.load_state_dict(ckpt["net"])
    net.eval()
    ie = ImageEncoder(vit_init=m["vit_init"]).to(device).eval()
    ie.load_state_dict(ckpt["image_encoder"])
    qt = QueryTokenizer(latent_dim=m["latent_dim"], dropout_prob=0.0, vit_feat_dim=1280).to(device).eval()
    qt.load_state_dict(ckpt["query_tokenizer"])
    diffusion = GaussianDiffusion(num_steps=cfg["diffusion"]["num_steps"])

    q_all, tok_all, vit_all, four_all = [], [], [], []
    swap_self, swap_cross = [], []
    ade_model, fde_model, ade_static = [], [], []
    raw_std = []

    for bi, batch in enumerate(loader):
        if bi >= args.num_batches:
            break
        images = batch["image"].to(device)
        q = batch["query_xyz_t0"].to(device)
        uv = batch["query_uv"].to(device)
        oh, ow = batch["orig_image_size"][0][0].item(), batch["orig_image_size"][1][0].item()

        nm = batch["norm_mean"].numpy()[:, None, None]
        ns = batch["norm_scale"].numpy()[:, None, None]
        gt = batch["tracks"].numpy() * ns + nm
        mask = batch["frame_mask"].numpy().astype(bool)
        q_metric = q.cpu().numpy() * ns[:, 0] + nm[:, 0]

        if args.renorm:
            # moments come from the t=0 query points alone, so this uses nothing a deployed
            # model wouldn't have at inference time
            mu = q_metric.mean(axis=1, keepdims=True)
            if args.renorm_mode == "depth":
                # one scalar per sequence, so relative geometry survives the transform
                k = np.array([float(v) for v in args.scale_const.split(",")], dtype=np.float32)
                sd = np.median(q_metric[..., 2], axis=1)[:, None, None] / k + 1e-6
            else:
                sd = q_metric.std(axis=1, keepdims=True) + 1e-6
            raw_std.append(((q_metric - mu) / sd).reshape(-1, 3))
            q = torch.from_numpy(((q_metric - mu) / sd * TRAIN_STD + TRAIN_MEAN).astype(np.float32)).to(device)

        def to_metric(pred_norm):
            if args.renorm:
                return (pred_norm - TRAIN_MEAN) / TRAIN_STD * sd[:, None] + mu[:, None]
            return pred_norm * ns + nm

        feats = ie(images)
        tokens = qt(q, feats, uv, orig_image_size=(oh, ow))  # (B, N, D)

        q_all.append(q.cpu().numpy())
        tok_all.append(tokens.mean(dim=1).cpu().numpy())          # per-input conditioning summary
        vit_all.append(feats.mean(dim=(2, 3)).cpu().numpy())      # per-input image summary
        four_all.append(fourier_embed(q, freq=qt.fourier_freq).cpu().numpy())

        # does the prediction actually depend on which input's conditioning it gets?
        cond = tokens.transpose(0, 1)
        shape = (q.shape[0], t_len, n, 3)
        nm_t = batch["norm_mean"].to(device)
        ns_t = batch["norm_scale"].to(device)
        pred = sample(net, diffusion, cond, q, shape, device, norm_mean=nm_t, norm_scale=ns_t)
        rolled = torch.roll(tokens, shifts=1, dims=0).transpose(0, 1)
        pred_swap = sample(net, diffusion, rolled, q, shape, device, norm_mean=nm_t, norm_scale=ns_t)

        pn = to_metric(pred.cpu().numpy())
        ps = to_metric(pred_swap.cpu().numpy())
        for b in range(pn.shape[0]):
            ade_model.append(ade(pn[b], gt[b], mask[b]))
            fde_model.append(fde(pn[b], gt[b], mask[b]))
            ade_static.append(ade(static(q_metric[b], t_len), gt[b], mask[b]))
        swap_self.append(np.linalg.norm(pn - pn[np.roll(np.arange(pn.shape[0]), 1)], axis=-1).mean() * 100)
        swap_cross.append(np.linalg.norm(pn - ps, axis=-1).mean() * 100)

    q_all = np.concatenate(q_all)            # (S, N, 3) normalized
    tok_all = np.concatenate(tok_all)        # (S, D)
    vit_all = np.concatenate(vit_all)        # (S, C)
    four_all = np.concatenate(four_all)      # (S, N, F)

    flat = q_all.reshape(-1, 3)
    tag = args.tag or f"{args.eval_dataset}/{args.split}"
    print(f"\n=== {tag}  (n_seq={q_all.shape[0]}) ===")
    print(f"normalized query xyz  mean {np.round(flat.mean(0), 3)}  std {np.round(flat.std(0), 3)}")
    print(f"  frac |coord| > 3     {float((np.abs(flat) > 3).mean()):.4f}")
    print(f"  frac |coord| > 5     {float((np.abs(flat) > 5).mean()):.4f}")
    print(f"  max |coord|          {float(np.abs(flat).max()):.2f}")

    # if the high Fourier bands alias, points that are close in space stop having similar
    # embeddings, and the position half of the conditioning turns into noise
    fn = four_all / (np.linalg.norm(four_all, axis=-1, keepdims=True) + 1e-8)
    sims = []
    for s in range(min(q_all.shape[0], 40)):
        d = np.linalg.norm(q_all[s][:, None] - q_all[s][None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = d.argmin(axis=1)
        sims.append((fn[s] * fn[s][nn]).sum(-1).mean())
    print(f"fourier cos-sim to nearest neighbour point   {np.mean(sims):.4f}")

    print(f"vit feature spread across inputs             {spread(vit_all):.4f}")
    print(f"cond token spread across inputs              {spread(tok_all):.4f}")
    print(f"prediction spread across inputs (cm)         {np.mean(swap_self):.3f}")
    print(f"prediction change when cond is swapped (cm)  {np.mean(swap_cross):.3f}")
    frac = np.mean(swap_cross) / max(np.mean(swap_self), 1e-6)
    print(f"  as a fraction of across-input spread         {frac:.3f}")
    print(f"ADE (cm)   model {np.mean(ade_model):.2f}   static {np.mean(ade_static):.2f}")
    print(f"FDE (cm)   model {np.mean(fde_model):.2f}")
    if raw_std:
        print(f"pre-rescale std of the renormalized coords    {np.round(np.concatenate(raw_std).std(0), 4)}")


if __name__ == "__main__":
    main()
