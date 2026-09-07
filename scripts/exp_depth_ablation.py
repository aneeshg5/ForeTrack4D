import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from foretrack.eval.metrics import (
    ade,
    ade_first_frame_aligned,
    ade_global_aligned,
    ade_per_timestep,
    apd3d,
    fde,
)
from foretrack.labeling.filter import accept_clip
from foretrack.labeling.run_tapip3d import estimate_depths, run_with_query_points


def load_video(image_paths):
    frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in image_paths]
    return np.stack(frames)


def load_sensor_depth(image_paths, scale):
    depths = []
    for p in image_paths:
        d = cv2.imread(str(Path(p).parent / f"aligned_depth_to_color_{Path(p).stem.split('_')[-1]}.png"), cv2.IMREAD_UNCHANGED)
        depths.append(d.astype(np.float32) / scale)
    return np.stack(depths)


def sample_depth(depth, uv):
    h, w = depth.shape
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    return depth[v, u]


def project(points, intrinsics):
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    z = np.maximum(points[..., 2], 1e-6)
    return np.stack([points[..., 0] / z * fx + cx, points[..., 1] / z * fy + cy], axis=-1)


def score(tracks, visibility, gt, intrinsics):
    return {
        "ade": ade(tracks, gt),
        "fde": fde(tracks, gt),
        "ade_first_aligned": ade_first_frame_aligned(tracks, gt),
        "ade_global_aligned": ade_global_aligned(tracks, gt),
        "apd3d": apd3d(tracks, gt, float(intrinsics[0, 0]), float(intrinsics[1, 1]), visibility),
        "curve": ade_per_timestep(tracks, gt).tolist(),
    }


def run_condition(name, video, intrinsics, query_point, depths, gt, visibility, work_dir, cfg):
    out_npz = Path(work_dir) / f"{name}.npz"
    run_with_query_points(
        video, intrinsics, query_point, str(out_npz),
        cfg["tapip3d_python"], cfg["tapip3d_repo"], cfg["tapip3d_checkpoint"],
        str(Path(work_dir) / name), depths=depths,
    )
    result = np.load(out_npz)
    tracks, vis = result["tracks"].astype(np.float32), result["visibility"]
    t = min(tracks.shape[0], gt.shape[0])
    out = score(tracks[:t], visibility[:t], gt[:t], intrinsics)
    out["accepted"] = bool(accept_clip(tracks[:t], vis[:t], cfg.get("filter", {})))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="How much does monocular depth cost our pseudo-labels? Runs TAPIP3D twice per "
                    "sequence, once on sensor depth and once on its own estimated depth, and scores "
                    "both against mesh-derived ground truth."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--gt_root", required=True, help="gt_tracks/<dataset>/<split>")
    parser.add_argument("--n_seq", type=int, default=20)
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="sensor depth png units per meter")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = sorted(Path(args.gt_root).glob("*/*/*.npz"))
    rng = np.random.default_rng(args.seed)
    seqs = [seqs[i] for i in rng.permutation(len(seqs))[: args.n_seq]]

    rows = []
    for i, seq in enumerate(seqs):
        rec_path = out_dir / f"{seq.parent.parent.name}_{seq.parent.name}_{seq.stem}.json"
        if rec_path.exists():
            rows.append(json.loads(rec_path.read_text()))
            continue

        z = np.load(seq, allow_pickle=True)
        gt, visibility, intrinsics = z["tracks"], z["visibility"], z["intrinsics"]
        image_paths = [str(p) for p in z["image_paths"]]
        video = load_video(image_paths)
        sensor = load_sensor_depth(image_paths, args.depth_scale)
        work = out_dir / "work" / rec_path.stem

        # identical query points in both conditions, so depth is the only variable
        query_point = np.concatenate([np.zeros((gt.shape[1], 1), np.float32), gt[0]], axis=1)

        rec = {"seq": str(seq), "object": str(z["object_name"]), "frames": int(gt.shape[0])}
        rec["sensor"] = run_condition("sensor", video, intrinsics, query_point, sensor, gt, visibility, work, cfg)

        est = estimate_depths(video, intrinsics, cfg["tapip3d_python"], cfg["tapip3d_repo"], cfg["tapip3d_checkpoint"], str(work / "depth_pass"))
        mono = est["depths"].astype(np.float32)
        rec["mono"] = run_condition("mono", video, intrinsics, query_point, mono, gt, visibility, work, cfg)

        # depth error at the tracked pixels themselves, which is what the lift actually reads
        uv = project(gt, intrinsics)
        mh, mw = mono.shape[1:3]
        scale = np.array([mw / video.shape[2], mh / video.shape[1]], np.float32)
        d_mono = np.stack([sample_depth(mono[t], uv[t] * scale) for t in range(gt.shape[0])])
        d_sensor = np.stack([sample_depth(sensor[t], uv[t]) for t in range(gt.shape[0])])
        valid = (d_sensor > 0.1) & (d_mono > 0.1)
        rec["depth_err_median_cm"] = float(np.median(np.abs(d_mono[valid] - d_sensor[valid])) * 100) if valid.any() else None
        rec["depth_err_median_frac"] = float(np.median(np.abs(d_mono[valid] - d_sensor[valid]) / d_sensor[valid])) if valid.any() else None

        rec_path.write_text(json.dumps(rec))
        rows.append(rec)
        print(f"[{i + 1}/{len(seqs)}] {rec['object']}  sensor ADE {rec['sensor']['ade']:.1f}cm  "
              f"mono ADE {rec['mono']['ade']:.1f}cm  depth err {rec['depth_err_median_cm']:.1f}cm", flush=True)

    summary = {"n": len(rows)}
    for cond in ("sensor", "mono"):
        for key in ("ade", "fde", "ade_first_aligned", "ade_global_aligned", "apd3d"):
            summary[f"{cond}_{key}"] = float(np.mean([r[cond][key] for r in rows]))
        summary[f"{cond}_accept_rate"] = float(np.mean([r[cond]["accepted"] for r in rows]) * 100)
    summary["depth_err_median_cm"] = float(np.median([r["depth_err_median_cm"] for r in rows if r["depth_err_median_cm"]]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "-" * 64)
    print(f"{'metric':<22}{'sensor depth':>14}{'mono depth':>14}{'delta':>12}")
    for key in ("ade", "fde", "ade_first_aligned", "ade_global_aligned", "apd3d", "accept_rate"):
        s, m = summary[f"sensor_{key}"], summary[f"mono_{key}"]
        print(f"{key:<22}{s:>14.2f}{m:>14.2f}{m - s:>+12.2f}")
    print(f"\nmedian depth error at tracked pixels: {summary['depth_err_median_cm']:.1f}cm")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
