import argparse
import json
import traceback
from pathlib import Path

from tqdm import tqdm

from foretrack.data.arctic import build_track_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic_root", required=True, help="<...>/unpack/arctic_data/data")
    parser.add_argument(
        "--protocol", default="p2",
        help="ARCTIC's own official split protocol (splits_json/protocol_{p}.json) -- p2 is "
        "the egocentric split. Used directly "
        "instead of forehand4d's separate motion split (which we don't need for ARCTIC: p2 "
        "already gives simple subject/seq_name lists per split, verified against the real "
        "downloaded data -- 267+34+38=339 total, matching the sequence count seen during "
        "download).",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N entries, for smoke testing")
    parser.add_argument("--num_shards", type=int, default=1, help="split entries across this many parallel workers")
    parser.add_argument("--shard_idx", type=int, default=0, help="which shard this worker processes, in [0, num_shards)")
    args = parser.parse_args()
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError(f"shard_idx {args.shard_idx} must be in [0, {args.num_shards})")

    arctic_root = Path(args.arctic_root)
    out_dir = Path(args.out_dir)

    with open(arctic_root / "splits_json" / f"protocol_{args.protocol}.json") as f:
        protocol = json.load(f)

    for split, entries in protocol.items():
        if args.limit is not None:
            entries = entries[: args.limit]
        shard_entries = entries[args.shard_idx :: args.num_shards]

        split_out_dir = out_dir / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        failures = []
        skipped = 0
        desc = f"generating ARCTIC GT tracks [{split}] (shard {args.shard_idx}/{args.num_shards})"
        for entry in tqdm(shard_entries, desc=desc):
            subject, seq_name = entry.split("/")
            seq_object_npy = arctic_root / "raw_seqs" / subject / f"{seq_name}.object.npy"
            out_path = split_out_dir / subject / f"{seq_name}.npz"

            if out_path.exists():
                skipped += 1
                continue
            if not seq_object_npy.exists():
                failures.append((entry, f"missing raw_seqs file: {seq_object_npy}"))
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                build_track_npz(str(seq_object_npy), str(out_path), str(arctic_root), n=args.n)
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
