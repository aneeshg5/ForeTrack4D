import argparse
import glob
import json
from pathlib import Path

import numpy as np

# Physical plausibility for a hand-held object. Reference tracks for zero-shot eval come from
# a monocular/Aria depth pipeline plus TAPIP3D, the same class of pipeline that
# scripts/exp_depth_ablation.py measured at ~10cm label error, so they need the same scrutiny
# we give pseudo-labels. TAPVid-MV (arXiv:2609.01899) visually verifies every sequence it
# ships; this is the automatic half of that.
MAX_SPEED_MPS = 3.0
DEPTH_RANGE_M = (0.15, 1.6)


def audit(path: str, fps: float, max_speed: float, depth_range) -> dict:
    tr = np.load(path, allow_pickle=True)["tracks"].astype(np.float64)
    step = np.linalg.norm(np.diff(tr, axis=0), axis=-1).mean(axis=1)
    speed = float(step.max() * fps)
    z = float(np.median(tr[0, :, 2]))
    reasons = []
    if not np.isfinite(tr).all():
        reasons.append("non-finite coordinates")
    if speed > max_speed:
        reasons.append(f"implausible speed {speed:.1f} m/s")
    if not (depth_range[0] <= z <= depth_range[1]):
        reasons.append(f"query depth {z:.2f}m outside manipulation range")
    return {
        "file": path, "frames": int(tr.shape[0]), "query_depth_m": round(z, 3),
        "peak_speed_mps": round(speed, 2),
        "moved_cm": round(float(np.linalg.norm(tr[-1] - tr[0], axis=-1).mean() * 100), 1),
        "reasons": reasons, "ok": not reasons,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Flag reference tracks that are physically implausible for a hand-held "
                    "object, so a zero-shot benchmark is not scored against tracker failures."
    )
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_speed", type=float, default=MAX_SPEED_MPS)
    parser.add_argument("--depth_range", type=float, nargs=2, default=DEPTH_RANGE_M)
    parser.add_argument("--link_clean_to", default=None, help="write a gated copy (symlinks) here")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.gt_root}/**/*.npz", recursive=True))
    rows = [audit(f, args.fps, args.max_speed, tuple(args.depth_range)) for f in files]
    bad = [r for r in rows if not r["ok"]]
    for r in bad:
        print(f"REJECT {Path(r['file']).name}: {'; '.join(r['reasons'])}")
    print(f"\n{len(rows) - len(bad)}/{len(rows)} sequences pass ({100 * (len(rows) - len(bad)) / max(len(rows), 1):.0f}%)")

    kept = [r for r in rows if r["ok"]]
    if kept:
        moved = np.array([r["moved_cm"] for r in kept])
        print(f"kept motion: median {np.median(moved):.1f}cm  p90 {np.percentile(moved, 90):.1f}cm  max {moved.max():.1f}cm")

    if args.link_clean_to:
        out = Path(args.link_clean_to)
        out.mkdir(parents=True, exist_ok=True)
        for r in kept:
            src = Path(r["file"]).resolve()
            dst = out / src.name
            if not dst.exists():
                dst.symlink_to(src)
        print(f"linked {len(kept)} clean sequences into {out}")
    if args.report:
        Path(args.report).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
