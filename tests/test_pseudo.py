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


def test_query_relative_norm_round_trips_to_metric(tmp_path):
    _write_synthetic_clip(tmp_path)
    ds = PseudoTracks(str(tmp_path), "val", n=8, t=5, norm_mode="query_relative")
    item = ds[0]
    nm = item["norm_mean"].numpy()
    ns = item["norm_scale"].numpy()
    recovered = item["tracks"].numpy() * ns + nm
    raw = np.load(str(tmp_path / "val" / "clip0.npz"))["tracks"].astype(np.float32)
    np.testing.assert_allclose(recovered, raw, rtol=1e-4, atol=1e-5)


def test_query_relative_norm_is_scene_independent(tmp_path):
    # the same object filmed twice at different distances must produce the same normalized
    # target, which is what keeps an unseen scene inside the training support
    _write_synthetic_clip(tmp_path)
    raw = np.load(str(tmp_path / "val" / "clip0.npz"))
    far = tmp_path / "far"
    (far / "val").mkdir(parents=True)
    img = far / "val" / "frame_0.png"
    cv2.imwrite(str(img), np.full((240, 320, 3), 128, dtype=np.uint8))
    scaled = raw["tracks"] * 3.0
    np.savez(
        far / "val" / "clip0.npz",
        tracks=scaled, visibility=raw["visibility"], intrinsics=raw["intrinsics"],
        image_paths=np.array([str(img)] * scaled.shape[0]), query_frame_idx=0,
        query_xyz_t0=scaled[0], object_id=0, source="pseudo",
    )
    a = PseudoTracks(str(tmp_path), "val", n=8, t=5, norm_mode="query_relative")[0]
    b = PseudoTracks(str(far), "val", n=8, t=5, norm_mode="query_relative")[0]
    np.testing.assert_allclose(a["tracks"].numpy(), b["tracks"].numpy(), rtol=1e-4, atol=1e-5)


def test_dataset_norm_mode_is_scene_dependent(tmp_path):
    # the contrast that motivates the change: dataset-stat normalization moves the target when
    # the scene moves, which is why an unseen scene lands outside the trained range
    _write_synthetic_clip(tmp_path)
    ds = PseudoTracks(str(tmp_path), "val", n=8, t=5, norm_mode="dataset")
    item = ds[0]
    assert item["norm_scale"].numpy().shape == (3,)
    np.testing.assert_allclose(item["norm_mean"].numpy(), np.zeros(3), atol=1e-6)
