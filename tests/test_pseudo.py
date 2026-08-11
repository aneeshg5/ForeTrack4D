import cv2
import numpy as np

from foretrack.data.pseudo import PseudoTracks


def _write_synthetic_clip(tmp_path, n=8, t=5):
    split_dir = tmp_path / "val"
    split_dir.mkdir()
    img_path = split_dir / "frame_0.png"
    cv2.imwrite(str(img_path), np.full((240, 320, 3), 128, dtype=np.uint8))

    tracks = np.random.randn(t, n, 3).astype(np.float32)
    tracks[..., 2] += 1.0  # positive depth
    visibility = np.ones((t, n), dtype=bool)
    intrinsics = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1]], dtype=np.float32)

    np.savez(
        split_dir / "clip0.npz",
        tracks=tracks,
        visibility=visibility,
        intrinsics=intrinsics,
        image_paths=np.array([str(img_path)] * t),
        query_frame_idx=0,
        query_xyz_t0=tracks[0],
        object_id=0,
        source="pseudo",
    )
    return split_dir


def test_pseudo_tracks_loads_synthetic_clip(tmp_path):
    _write_synthetic_clip(tmp_path)
    ds = PseudoTracks(str(tmp_path), "val", n=8, t=5)
    assert len(ds) == 1
    item = ds[0]
    assert item["tracks"].shape == (5, 8, 3)
    assert item["visibility"].shape == (5, 8)
    assert item["image"].shape[0] == 3


def test_pseudo_tracks_raises_on_missing_split(tmp_path):
    _write_synthetic_clip(tmp_path)
    try:
        PseudoTracks(str(tmp_path), "test")
        assert False, "expected ValueError"
    except ValueError:
        pass
