import numpy as np

from foretrack.data.dexycb import CROP_MAX_INVALID_FRAC, compute_object_crop


def test_compute_object_crop_basic_no_image():
    query_uv = np.array([[100.0, 100.0], [120.0, 110.0]], dtype=np.float32)
    x0, y0, x1, y1 = compute_object_crop(query_uv, img_h=480, img_w=640)
    assert 0 <= x0 < x1 <= 640
    assert 0 <= y0 < y1 <= 480
    assert abs((x1 - x0) / (y1 - y0) - 192 / 256) < 1e-3


def test_compute_object_crop_clips_to_image_bounds():
    query_uv = np.array([[5.0, 5.0], [8.0, 8.0]], dtype=np.float32)
    x0, y0, x1, y1 = compute_object_crop(query_uv, img_h=480, img_w=640)
    assert x0 >= 0 and y0 >= 0
    assert x1 <= 640 and y1 <= 480


def test_compute_object_crop_no_op_when_image_has_no_black_region():
    query_uv = np.array([[300.0, 300.0], [340.0, 320.0]], dtype=np.float32)
    image = np.full((480, 640, 3), 200, dtype=np.uint8)  # uniformly bright, no invalid pixels
    without_image = compute_object_crop(query_uv, img_h=480, img_w=640)
    with_image = compute_object_crop(query_uv, img_h=480, img_w=640, image=image)
    assert without_image == with_image


def test_compute_object_crop_shrinks_to_avoid_black_vignette():
    img_h, img_w = 1408, 1408
    image = np.full((img_h, img_w, 3), 200, dtype=np.uint8)
    image[:, :450] = 0  # vertical black band, e.g. a vignette region to one side

    query_uv = np.array([[500.0, 500.0], [600.0, 550.0]], dtype=np.float32)  # bbox itself is clear of it

    def invalid_frac_of(x0, y0, x1, y1):
        crop = image[int(round(y0)) : int(round(y1)), int(round(x0)) : int(round(x1))]
        return (crop.sum(axis=-1) < 10).mean()

    naive = compute_object_crop(query_uv, img_h=img_h, img_w=img_w)
    assert invalid_frac_of(*naive) > CROP_MAX_INVALID_FRAC, "test setup should exercise the shrink path"

    shrunk = compute_object_crop(query_uv, img_h=img_h, img_w=img_w, image=image)
    assert shrunk != naive
    assert invalid_frac_of(*shrunk) <= CROP_MAX_INVALID_FRAC + 1e-6


def test_compute_object_crop_gives_up_gracefully_when_black_is_unavoidable():
    img_h, img_w = 400, 400
    image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    query_uv = np.array([[200.0, 200.0], [210.0, 210.0]], dtype=np.float32)
    x0, y0, x1, y1 = compute_object_crop(query_uv, img_h=img_h, img_w=img_w, image=image)
    assert 0 <= x0 < x1 <= img_w
    assert 0 <= y0 < y1 <= img_h
