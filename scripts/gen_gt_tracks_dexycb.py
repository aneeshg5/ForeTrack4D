import argparse
import pickle
import traceback
from pathlib import Path

import yaml
from tqdm import tqdm

from foretrack.data.gt_tracks import build_track_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="DexYCB root (DEX_YCB_DIR)")
    parser.add_argument("--split_file", required=True, help="forehand4d s3_{split}.pkl")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N entries, for smoke testing")
    parser.add_argument("--num_shards", type=int, default=1, help="split the split-file entries across this many parallel workers")
    parser.add_argument("--shard_idx", type=int, default=0, help="which shard this worker processes, in [0, num_shards)")
    args = parser.parse_args()
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError(f"shard_idx {args.shard_idx} must be in [0, {args.num_shards})")

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_file, "rb") as f:
        split = pickle.load(f)
    names, ranges = split["names"], split["ranges"]
    if args.limit is not None:
        names, ranges = names[: args.limit], ranges[: args.limit]

    entries = list(zip(names, ranges))[args.shard_idx :: args.num_shards]

    meta_cache = {}
    failures = []
    skipped = 0

    desc = f"generating GT tracks (shard {args.shard_idx}/{args.num_shards})"
    for name, frame_range in tqdm(entries, desc=desc):
        subject, sequence, serial = name.split("/")
        seq_dir = root / subject / sequence / serial
        out_path = out_dir / subject / sequence / f"{serial}.npz"

        if out_path.exists():
            skipped += 1
            continue

        meta_key = (subject, sequence)
        if meta_key not in meta_cache:
            with open(root / subject / sequence / "meta.yml") as f:
                meta_cache[meta_key] = yaml.safe_load(f)
        num_frames = meta_cache[meta_key]["num_frames"]

        if tuple(frame_range) != (0, num_frames - 1):
            failures.append((name, f"split range {frame_range} != full sequence (0, {num_frames - 1})"))
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            build_track_npz(str(seq_dir), str(out_path), n=args.n)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    succeeded = len(entries) - len(failures) - skipped
    print(f"shard {args.shard_idx}/{args.num_shards} done. {succeeded} succeeded, {skipped} already existed, {len(failures)} failed.")
    if failures:
        fail_log = out_dir / f"failures.shard{args.shard_idx}.txt"
        with open(fail_log, "w") as f:
            for name, err in failures:
                f.write(f"{name}\t{err}\n")
        print(f"failure details written to {fail_log}")


if __name__ == "__main__":
    main()
