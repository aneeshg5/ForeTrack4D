import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="build a TAPIP3D-input npz from a DexYCB sequence + our GT query points, "
        "for cross-checking our mesh-derived tracks against TAPIP3D's independent 3D tracking."
    )
    parser.add_argument("--seq_dir", required=True, help="DexYCB camera-view sequence dir")
    parser.add_argument("--gt_npz", required=True, help="our build_track_npz output for this sequence")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    gt = np.load(args.gt_npz)
    num_frames = gt["tracks"].shape[0]

    video = np.stack(
        [
            cv2.cvtColor(cv2.imread(str(seq_dir / f"color_{t:06d}.jpg")), cv2.COLOR_BGR2RGB)
            for t in range(num_frames)
        ]
    ).astype(np.uint8)

    depths = np.stack(
        [
            cv2.imread(str(seq_dir / f"aligned_depth_to_color_{t:06d}.png"), cv2.IMREAD_UNCHANGED)
            for t in range(num_frames)
        ]
    ).astype(np.float32) / 1000.0  # DexYCB depth is uint16 millimeters -> meters

    intrinsics = np.stack([gt["intrinsics"]] * num_frames).astype(np.float32)

    query_xyz_t0 = gt["query_xyz_t0"]  # (N, 3), camera frame, same convention TAPIP3D uses
    # TAPIP3D query_point format: (N, 4) = [query_frame_idx, x, y, z]
    query_point = np.concatenate(
        [np.zeros((len(query_xyz_t0), 1), dtype=np.float32), query_xyz_t0], axis=1
    ).astype(np.float32)

    # no extrinsics key -> TAPIP3D defaults to identity per frame, matching our
    # own world==camera-frame convention
    np.savez(args.out, video=video, depths=depths, intrinsics=intrinsics, query_point=query_point)
    print(f"wrote {args.out}: video {video.shape}, depths {depths.shape}, query_point {query_point.shape}")


if __name__ == "__main__":
    main()
