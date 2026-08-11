import argparse

import cv2
import imageio
import numpy as np


def project(points, intrinsics):
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    u = (points[:, 0] / points[:, 2] * fx + cx).round().astype(int)
    v = (points[:, 1] / points[:, 2] * fy + cy).round().astype(int)
    return u, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_npz", required=True)
    parser.add_argument("--tapip3d_npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    gt = np.load(args.gt_npz)
    tp = np.load(args.tapip3d_npz)
    gt_tracks, intrinsics, image_paths = gt["tracks"], gt["intrinsics"], gt["image_paths"]
    tp_tracks = tp["coords"]

    frames_out = []
    for t, img_path in enumerate(image_paths):
        frame = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gu, gv = project(gt_tracks[t], intrinsics)
        tu, tv = project(tp_tracks[t], intrinsics)
        for i in range(len(gu)):
            cv2.circle(frame, (gu[i], gv[i]), 5, (0, 255, 0), -1)  # our mesh GT: green
            cv2.circle(frame, (tu[i], tv[i]), 3, (255, 0, 255), -1)  # tapip3d: magenta
            cv2.line(frame, (gu[i], gv[i]), (tu[i], tv[i]), (255, 255, 0), 1)
        frames_out.append(frame)

    imageio.mimwrite(args.out, frames_out, fps=args.fps, quality=8)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
