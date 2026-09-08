import argparse
import hashlib
import json
from pathlib import Path

FINE = "Fine grained action"
COARSE = "Coarse grained action"


def session_split(video_name: str, val_frac: int = 10) -> str:
    # stable per-session hash, so a session's assignment never changes as the corpus grows and
    # no session ever contributes clips to both sides
    h = int(hashlib.md5(video_name.encode()).hexdigest()[:8], 16)
    return "val" if h % val_frac == 0 else "train"


def main():
    p = argparse.ArgumentParser(
        description="Convert HoloAssist's released annotation file into the labels_*_classes.json "
        "plus split lists that data/holoassist.py already reads."
    )
    p.add_argument("--annotation", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--video_root", default=None, help="restrict splits to sessions whose video is on disk")
    p.add_argument("--tag", default="20230000")
    p.add_argument("--min_seconds", type=float, default=1.0)
    p.add_argument("--max_seconds", type=float, default=20.0)
    args = p.parse_args()

    sessions = json.load(open(args.annotation))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    available = None
    if args.video_root:
        root = Path(args.video_root)
        available = {d.name for d in root.iterdir() if (d / "Export_py" / "Video_compress.mp4").exists()}
        print(f"{len(available)} sessions have video on disk")

    task_types = sorted({s.get("taskType", "") for s in sessions})
    fine, coarse, split_lists = {}, {}, {"train": [], "val": []}
    kept = dropped = 0

    for s in sessions:
        name = s["video_name"]
        if available is not None and name not in available:
            continue
        task_id = task_types.index(s.get("taskType", ""))
        for key, bucket in ((FINE, fine), (COARSE, coarse)):
            rows = []
            for e in s.get("events", []):
                if e.get("label") != key or e.get("type") != "range":
                    continue
                t0, t1 = float(e.get("start", 0.0)), float(e.get("end", 0.0))
                if key == FINE:
                    # very short clips carry no forecastable motion, very long ones break the
                    # single-shot assumption MegaSaM needs
                    if not (args.min_seconds <= t1 - t0 <= args.max_seconds):
                        dropped += 1
                        continue
                    kept += 1
                rows.append([t0, t1, [task_id, int(e.get("id", 0))]])
            if rows:
                bucket[name] = rows
        if name in fine:
            split_lists[session_split(name)].append(name)

    classes_path = out / f"labels_{args.tag}_classes.json"
    with open(classes_path, "w") as f:
        json.dump({"fine_grained_action": fine, "coarse_grained_action": coarse}, f)
    for subset, names in split_lists.items():
        (out / f"{subset}_0724.txt").write_text("\n".join(sorted(names)) + "\n")

    print(f"wrote {classes_path}")
    print(f"sessions with usable clips: {len(fine)}  (train {len(split_lists['train'])}, val {len(split_lists['val'])})")
    print(f"fine-grained clips kept {kept}, dropped by duration {dropped}")


if __name__ == "__main__":
    main()
