import argparse
import glob

import numpy as np
from tqdm import tqdm

from foretrack.labeling.filter import depth_variance_gate, jerk_gate


def _peak_jerk(tracks: np.ndarray) -> float:
    if tracks.shape[0] < 4:
        return 0.0
    velocity = np.diff(tracks, axis=0)
    accel = np.diff(velocity, axis=0)
    jerk = np.diff(accel, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=-1)
    return float(np.median(jerk_mag.max(axis=0)))


def _depth_var(tracks: np.ndarray) -> float:
    return float(np.median(tracks[..., 2].var(axis=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dirs", nargs="+", required=True, help="e.g. /data01/aneeshg5/gt_tracks/dexycb/train ...")
    parser.add_argument("--percentile", type=float, default=99.0, help="reject rate on clean lab GT at this percentile")
    args = parser.parse_args()

    files = []
    for gt_dir in args.gt_dirs:
        files.extend(sorted(glob.glob(f"{gt_dir}/**/*.npz", recursive=True)))
    print(f"found {len(files)} files across {len(args.gt_dirs)} dirs")

    jerks, depth_vars = [], []
    for f in tqdm(files):
        tracks = np.load(f)["tracks"].astype(np.float64)
        jerks.append(_peak_jerk(tracks))
        depth_vars.append(_depth_var(tracks))

    jerks, depth_vars = np.array(jerks), np.array(depth_vars)
    jerk_threshold = float(np.percentile(jerks, args.percentile))
    depth_var_threshold = float(np.percentile(depth_vars, args.percentile))

    print(f"jerk: median={np.median(jerks):.6f} p{args.percentile}={jerk_threshold:.6f} max={jerks.max():.6f}")
    print(f"depth_var: median={np.median(depth_vars):.6f} p{args.percentile}={depth_var_threshold:.6f} max={depth_vars.max():.6f}")

    passed_jerk = sum(jerk_gate(np.load(f)["tracks"].astype(np.float64), jerk_threshold) for f in files)
    passed_depth = sum(depth_variance_gate(np.load(f)["tracks"].astype(np.float64), depth_var_threshold) for f in files)
    print(f"sanity check at these thresholds: {passed_jerk}/{len(files)} pass jerk_gate, {passed_depth}/{len(files)} pass depth_variance_gate")

    print()
    print(f"DEFAULT_JERK_THRESHOLD = {jerk_threshold:.6f}")
    print(f"DEFAULT_DEPTH_VAR_THRESHOLD = {depth_var_threshold:.6f}")


if __name__ == "__main__":
    main()
