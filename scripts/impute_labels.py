import argparse

import yaml

from foretrack.data.holoassist import list_annotated_clips
from foretrack.labeling.impute import clip_frame_range, clip_output_path, process_clip
from foretrack.labeling.segment import Sam2ObjectSegmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    segmenter = Sam2ObjectSegmenter(
        cfg["sam2_checkpoint"], cfg["sam2_model_cfg"], device=cfg.get("device", "cuda")
    )

    clips = list_annotated_clips(
        cfg["label_root"], args.split, event_key=cfg.get("event_key", "fine_grained_action")
    )
    accepted = 0
    for video_name, t_start, t_end, _task_id, _label_id in clips:
        query_frame, end_frame = clip_frame_range(t_start, t_end)
        out_path = clip_output_path(cfg["out_dir"], args.split, video_name, query_frame, end_frame)
        if out_path.exists():
            accepted += 1
            continue
        if process_clip(video_name, t_start, t_end, cfg, segmenter, out_path):
            accepted += 1

    rate = 100 * accepted / max(len(clips), 1)
    print(f"accepted {accepted}/{len(clips)} clips ({rate:.1f}%)")


if __name__ == "__main__":
    main()
