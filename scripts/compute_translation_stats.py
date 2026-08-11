import argparse
import glob
import json

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="e.g. /data01/aneeshg5/gt_tracks/dexycb/train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.gt_dir}/**/*.npz", recursive=True))
    print(f"found {len(files)} files")

    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    for f in tqdm(files):
        tracks = np.load(f)["tracks"].astype(np.float64)  # (T, N, 3)
        flat = tracks.reshape(-1, 3)
        total += flat.sum(axis=0)
        total_sq += (flat**2).sum(axis=0)
        count += flat.shape[0]

    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)

    stats = {"mean": mean.tolist(), "std": std.tolist(), "num_files": len(files), "num_points": count}
    print(stats)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
