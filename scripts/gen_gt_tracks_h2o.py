import argparse
import traceback
from pathlib import Path

from tqdm import tqdm

from foretrack.data.h2o import build_track_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2o_root", required=True, help="<...>/unpack, containing {subject}_ego/ dirs and object/object/{name}/{name}.obj")
    parser.add_argument(
        "--split_dir", required=True,
        help="dir containing action_{train,val,test}.txt (H2O's own official split -- unique "
        "'path' column per take, verified non-overlapping across splits: 114/24/46 takes, "
        "subject-disjoint). Used directly instead of a forehand4d split, same choice as ARCTIC.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N entries, for smoke testing")
    parser.add_argument("--num_shards", type=int, default=1, help="split entries across this many parallel workers")
    parser.add_argument("--shard_idx", type=int, default=0, help="which shard this worker processes, in [0, num_shards)")
    args = parser.parse_args()
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError(f"shard_idx {args.shard_idx} must be in [0, {args.num_shards})")

    h2o_root = Path(args.h2o_root)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)

    for split, fname in [("train", "action_train.txt"), ("val", "action_val.txt"), ("test", "action_test.txt")]:
        with open(split_dir / fname) as f:
            lines = f.readlines()[1:]  # header: id path action_label start_act end_act start_frame end_frame
        entries = sorted({line.split()[1] for line in lines if line.strip()})  # unique take paths

        if args.limit is not None:
            entries = entries[: args.limit]
        shard_entries = entries[args.shard_idx :: args.num_shards]

        split_out_dir = out_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        failures = []
        skipped = 0
        desc = f"generating H2O GT tracks [{split}] (shard {args.shard_idx}/{args.num_shards})"
        for entry in tqdm(shard_entries, desc=desc):
            subject, take, take_idx = entry.split("/")
            seq_dir = h2o_root / f"{subject}_ego" / take / take_idx / "cam4"
            out_path = split_out_dir / subject / take / f"{take_idx}.npz"

            if out_path.exists():
                skipped += 1
                continue
            if not seq_dir.exists():
                failures.append((entry, f"missing take dir: {seq_dir}"))
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                build_track_npz(str(seq_dir), str(out_path), str(h2o_root), n=args.n)
            except Exception as e:
                failures.append((entry, f"{type(e).__name__}: {e}"))
                traceback.print_exc()

        succeeded = len(shard_entries) - len(failures) - skipped
        print(f"[{split}] shard {args.shard_idx}/{args.num_shards} done. {succeeded} succeeded, {skipped} already existed, {len(failures)} failed.")
        if failures:
            fail_log = split_out_dir / f"failures.shard{args.shard_idx}.txt"
            with open(fail_log, "w") as f:
                for entry, err in failures:
                    f.write(f"{entry}\t{err}\n")
            print(f"failure details written to {fail_log}")


if __name__ == "__main__":
    main()
