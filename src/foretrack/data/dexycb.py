# dataset loader for our own build_track_npz output (data/gt_tracks.py), not adapted from
# forehand4d's dexycb_dataset.py -- our GT schema is our own. image normalization
# (ImageNet mean/std) matches forehand4d's convention (src/parsers/parser.py,
# img_norm_mean/img_norm_std), used consistently across all their dataset classes including
# their own dexycb_dataset.py, since the ViT backbone is HaMeR-pretrained against that.

import glob
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import normalize_translation

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_transl_stats(path: str) -> tuple:
    with open(path) as f:
        stats = json.load(f)
    return np.array(stats["mean"], dtype=np.float32), np.array(stats["std"], dtype=np.float32)


# HaMeR's ViT was pretrained on tight hand-crops (the hand fills most of the frame), not full
# scenes. DexYCB's object occupies only ~12% x 20% of the raw 640x480 frame -- feeding it
# through unchanged left the object spanning only ~2x3 of the ViT's 16x12 patch grid after the
# old naive full-scene resize, starving the model of visual detail (found while diagnosing a
# training plateau that a learning-rate fix didn't resolve). This crops
# around the object first, matching HaMeR's own hand-crop paradigm, before any resizing.
CROP_TARGET_SIZE = (256, 192)  # (H, W), matches ImageEncoder.vit_input_size exactly
CROP_PADDING_FRAC = 1.0  # each side gets bbox_size * this fraction of extra context
CROP_MIN_SIZE_PX = 150  # floor so small objects (e.g. scissors) don't get a degenerate crop


CROP_MAX_INVALID_FRAC = 0.15  # see `image` param docstring below


def compute_object_crop(
    query_uv: np.ndarray, img_h: int, img_w: int, scale_jitter: float = 1.0, image: np.ndarray = None
) -> tuple:
    """object-centric crop box (x0, y0, x1, y1) in original-image pixel coords: pad around the
    query points' bounding box, force the target aspect ratio, enforce a minimum size, then
    clip to stay within the image. scale_jitter (train-time augmentation, see augm_params)
    multiplies the crop's box size, mirroring forehand4d's `sc` term in augm_params/
    rgb_processing -- 1.0 (no jitter) at eval time.

    image: optional (H,W,3) uint8 source frame. When given, shrinks CROP_PADDING_FRAC (retrying
    up to 4 times, floor 0.1) if the resulting crop would contain more than
    CROP_MAX_INVALID_FRAC near-black pixels. Real, visually-confirmed need: EgoExo4D's Aria
    camera is undistorted from a raw fisheye sensor into a pinhole projection whose corners
    (outside the fisheye's actual circular field of view) are exactly black -- zero-information
    padding the ViT (pretrained on natural hand-crop images) has never seen. The padding factor
    below was tuned against DexYCB's normal-FOV camera, which has no such region at all; a large
    or edge-adjacent query bbox on EgoExo4D can pad straight into it (confirmed via real crop
    inspection: 2/5 sampled EgoExo4D crops had a black-vignette intrusion, one severe at ~40% of
    the frame). No-op for every other dataset in this project (DexYCB/
    ARCTIC/H2O/HoloAssist frames have no black regions of this kind, so the invalid-fraction
    check always passes on the first try there)."""
    x0, y0 = query_uv[:, 0].min(), query_uv[:, 1].min()
    x1, y1 = query_uv[:, 0].max(), query_uv[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bbox_w, bbox_h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    target_aspect = CROP_TARGET_SIZE[1] / CROP_TARGET_SIZE[0]  # W/H

    padding_frac = CROP_PADDING_FRAC
    for _ in range(5):
        w = bbox_w * (1 + 2 * padding_frac) * scale_jitter
        h = bbox_h * (1 + 2 * padding_frac) * scale_jitter
        if w / h > target_aspect:
            h = w / target_aspect
        else:
            w = h * target_aspect
        # bug fix: flooring w and h independently against CROP_MIN_SIZE_PX (a floor on width)
        # and CROP_MIN_SIZE_PX * target_aspect (intended as the matching floor on height) only
        # preserves the aspect ratio when at most one of the two floors actually triggers. When
        # both do (small bounding boxes -- the exact "e.g. scissors" case this constant exists
        # for), the two independent max()s silently INVERT the aspect ratio (target_aspect=0.75,
        # i.e. a portrait 256x192 target, was coming out landscape ~1.33), stretching the object
        # in the final resize. A single shared scale factor keeps the ratio exact regardless of
        # which floor (if any) is binding.
        if w < CROP_MIN_SIZE_PX:
            scale = CROP_MIN_SIZE_PX / w
            w, h = w * scale, h * scale

        cx0, cx1 = cx - w / 2, cx + w / 2
        cy0, cy1 = cy - h / 2, cy + h / 2
        # shift (don't shrink -- would break the aspect ratio) to fit inside the image
        if cx0 < 0:
            cx1 -= cx0
            cx0 = 0
        if cx1 > img_w:
            cx0 -= cx1 - img_w
            cx1 = img_w
        if cy0 < 0:
            cy1 -= cy0
            cy0 = 0
        if cy1 > img_h:
            cy0 -= cy1 - img_h
            cy1 = img_h
        cx0, cy0 = max(cx0, 0), max(cy0, 0)
        cx1, cy1 = min(cx1, img_w), min(cy1, img_h)

        if image is None or padding_frac <= 0.1:
            break
        crop = image[int(round(cy0)) : int(round(cy1)), int(round(cx0)) : int(round(cx1))]
        if crop.size == 0 or (crop.sum(axis=-1) < 10).mean() <= CROP_MAX_INVALID_FRAC:
            break
        padding_frac = max(padding_frac * 0.5, 0.1)

    return cx0, cy0, cx1, cy1


class TrackFramesDataset(Dataset):
    """shared reader for build_track_npz's output schema (tracks, visibility, intrinsics,
    image_paths, query_frame_idx, query_xyz_t0, object_id) -- object-centric crop,
    forehand4d-derived augmentation, and frame padding are all dataset-agnostic once a
    dataset's GT generator produces this schema, so this is factored out here (DexYCBTracks
    was the original, only implementation) rather than duplicated verbatim into ArcticTracks
    (data/arctic.py) -- delicate, previously-debugged crop/augmentation code with a real history
    of subtle bugs is exactly the kind of thing that should have ONE copy,
    not two that can silently drift apart. Subclasses only need to populate self.files."""

    def __init__(
        self,
        files: list,
        n: int = 64,
        t: int = 128,
        transl_stats_path: str = None,
        augment: bool = False,
        noise_factor: float = 0.4,
        scale_factor: float = 0.25,
    ):
        self.n = n
        self.t = t
        # train-time-only augmentation (see augm_params/rgb_processing in forehand4d's
        # common/data_utils.py) -- caller passes augment=False for val/test splits, matching
        # their is_train gating.
        self.augment = augment
        self.noise_factor = noise_factor
        self.scale_factor = scale_factor
        self.files = files

        if transl_stats_path is not None:
            self.transl_mean, self.transl_std = load_transl_stats(transl_stats_path)
        else:
            self.transl_mean, self.transl_std = np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        tracks = d["tracks"].astype(np.float32)  # (T_actual, N_stored, 3), raw metric coords
        visibility = d["visibility"].astype(bool)  # (T_actual, N_stored)
        intrinsics = d["intrinsics"].astype(np.float32)  # (3, 3)
        image_paths = d["image_paths"]
        query_frame_idx = int(d["query_frame_idx"])
        query_xyz_t0_raw = d["query_xyz_t0"].astype(np.float32)  # (N_stored, 3), raw metric coords

        # N ablation: every npz stores N_stored=64 points in fixed
        # farthest-point-sampling order (decision 4) -- a PREFIX of that order is itself a
        # reasonably well-spread subset (FPS greedily picks the most-spread point first), so
        # self.n < 64 just takes the first self.n. self.n > 64 is not supported -- would need
        # the GT tracks regenerated with more query points, not a slicing operation.
        if self.n > tracks.shape[1]:
            raise ValueError(f"requested n={self.n} exceeds stored N={tracks.shape[1]} query points; regenerate GT tracks to support n > {tracks.shape[1]}")
        tracks = tracks[:, : self.n]
        visibility = visibility[:, : self.n]
        query_xyz_t0_raw = query_xyz_t0_raw[: self.n]

        # query 2D projections in the ORIGINAL image (pinhole), for QueryTokenizer's bilinear
        # sampling -- must come from the raw (unnormalized) coords, before any normalization.
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        z = query_xyz_t0_raw[:, 2]
        query_uv = np.stack(
            [query_xyz_t0_raw[:, 0] / z * fx + cx, query_xyz_t0_raw[:, 1] / z * fy + cy], axis=-1
        ).astype(np.float32)

        image = cv2.cvtColor(cv2.imread(str(image_paths[query_frame_idx])), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # crop around the object (see compute_object_crop's docstring for why), then resize the
        # crop to the ViT's exact target size and remap query_uv into that same frame -- must
        # stay in lockstep; a mismatch silently samples the wrong image location.
        # scale jitter (sc) mirrors forehand4d's augm_params: clipped-Gaussian factor around 1.0,
        # applied to the crop box only during training.
        scale_jitter = 1.0
        if self.augment and self.scale_factor > 0:
            scale_jitter = float(np.clip(1 + np.random.randn() * self.scale_factor, 1 - self.scale_factor, 1 + self.scale_factor))
        cx0, cy0, cx1, cy1 = compute_object_crop(query_uv, orig_h, orig_w, scale_jitter=scale_jitter, image=image)
        image = image[int(round(cy0)) : int(round(cy1)), int(round(cx0)) : int(round(cx1))]
        crop_h, crop_w = image.shape[:2]
        image = cv2.resize(image, (CROP_TARGET_SIZE[1], CROP_TARGET_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        query_uv = query_uv - np.array([cx0, cy0], dtype=np.float32)
        query_uv = query_uv * np.array(
            [CROP_TARGET_SIZE[1] / crop_w, CROP_TARGET_SIZE[0] / crop_h], dtype=np.float32
        )
        orig_h, orig_w = CROP_TARGET_SIZE  # image/query_uv are now in this frame, not the raw one

        image = image.astype(np.float32) / 255.0
        # per-channel pixel noise (pn), mirrors forehand4d's rgb_processing: multiplicative
        # factor per RGB channel, train-time only.
        if self.augment and self.noise_factor > 0:
            pn = np.random.uniform(1 - self.noise_factor, 1 + self.noise_factor, size=3).astype(np.float32)
            image = np.clip(image * pn, 0.0, 1.0)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # (3, H, W)

        # DexYCB sequences (~68-74 frames) are shorter than the configured horizon T -- pad by
        # repeating the last valid frame (keeps depth/positions physically
        # sane, unlike zero-padding which would blow up depth_scaled_l2's 1/depth weighting)
        # and provide frame_mask so the training loop can exclude padded frames from the loss.
        t_actual = tracks.shape[0]
        frame_mask = np.zeros(self.t, dtype=bool)
        frame_mask[: min(t_actual, self.t)] = True
        if t_actual < self.t:
            pad = self.t - t_actual
            tracks = np.concatenate([tracks, np.repeat(tracks[-1:], pad, axis=0)], axis=0)
            visibility = np.concatenate([visibility, np.repeat(visibility[-1:], pad, axis=0)], axis=0)
        else:
            tracks = tracks[: self.t]
            visibility = visibility[: self.t]

        depth = tracks[..., 2].copy()  # (T, N), raw metric depth for depth_scaled_l2

        tracks_norm = normalize_translation(tracks, self.transl_mean, self.transl_std)
        query_xyz_t0_norm = normalize_translation(query_xyz_t0_raw, self.transl_mean, self.transl_std)

        return {
            "image": image,
            "query_xyz_t0": torch.from_numpy(query_xyz_t0_norm),
            "query_uv": torch.from_numpy(query_uv),
            "orig_image_size": (orig_h, orig_w),
            "intrinsics": torch.from_numpy(intrinsics),
            "tracks": torch.from_numpy(tracks_norm),
            "visibility": torch.from_numpy(visibility),
            "frame_mask": torch.from_numpy(frame_mask),
            "depth": torch.from_numpy(depth),
            "object_id": int(d["object_id"]),
        }


class DexYCBTracks(TrackFramesDataset):
    def __init__(self, root: str, split: str, **kwargs):
        files = sorted(glob.glob(f"{root}/{split}/**/*.npz", recursive=True))
        if len(files) == 0:
            raise ValueError(f"no npz files found under {root}/{split}")
        super().__init__(files, **kwargs)
