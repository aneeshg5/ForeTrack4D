import argparse

import cv2
import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    d = np.load(args.npz)
    tracks, vis, intrinsics = d["tracks"], d["visibility"], d["intrinsics"]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    frames_out = []
    for t, img_path in enumerate(d["image_paths"]):
        frame = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        u = (tracks[t, :, 0] / tracks[t, :, 2] * fx + cx).round().astype(int)
        v = (tracks[t, :, 1] / tracks[t, :, 2] * fy + cy).round().astype(int)
        for i in range(len(u)):
            color = (0, 255, 0) if vis[t, i] else (255, 0, 0)
            cv2.circle(frame, (u[i], v[i]), 4, color, -1)
        frames_out.append(frame)

    imageio.mimwrite(args.out, frames_out, fps=args.fps, quality=8)
    print(f"wrote {args.out}, {len(frames_out)} frames")


if __name__ == "__main__":
    main()
