import cv2
import numpy as np
import pytest

from foretrack.labeling.segment import sample_query_points_in_mask, save_mask_overlay


def test_sample_query_points_in_mask_shape():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    pts = sample_query_points_in_mask(mask, n=64, seed=0)
    assert pts.shape == (64, 2)
    assert pts.dtype == np.float32


def test_sample_query_points_in_mask_within_bounds():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True
    pts = sample_query_points_in_mask(mask, n=16, seed=1)
    assert (pts[:, 0] >= 10).all() and (pts[:, 0] < 20).all()
    assert (pts[:, 1] >= 10).all() and (pts[:, 1] < 20).all()


def test_sample_query_points_in_mask_small_mask_allows_repeats():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    pts = sample_query_points_in_mask(mask, n=8, seed=0)
    assert pts.shape == (8, 2)


def test_sample_query_points_in_mask_empty_raises():
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError):
        sample_query_points_in_mask(mask, n=4)


def test_save_mask_overlay_writes_readable_image(tmp_path):
    frame = np.full((40, 40, 3), 100, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True
    out_path = tmp_path / "overlay.jpg"
    save_mask_overlay(frame, mask, contact_point=(15.0, 15.0), out_path=out_path)
    assert out_path.exists()
    written = cv2.imread(str(out_path))
    assert written.shape == (40, 40, 3)


def test_save_mask_overlay_tints_masked_region_red():
    frame = np.full((20, 20, 3), 100, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out_path = f"{d}/overlay.jpg"
        save_mask_overlay(frame, mask, contact_point=(2.0, 2.0), out_path=out_path)
        written = cv2.imread(out_path)
        inside_red = written[7, 7, 2]
        outside_red = written[1, 1, 2]
        assert inside_red > outside_red
