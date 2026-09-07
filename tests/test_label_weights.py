import json

import numpy as np
import pytest

from foretrack.data.label_weights import horizon_weights, load_calibration

CALIB = {
    "fps": 30.0,
    "seconds": [0.0, 1.0, 2.0],
    "label_err_cm": [0.0, 5.0, 25.0],
}


def test_full_weight_while_label_error_within_tolerance():
    w = horizon_weights(61, 30.0, CALIB, tol_cm=5.0)
    assert np.allclose(w[:31], 1.0)


def test_weight_falls_as_label_error_grows():
    w = horizon_weights(61, 30.0, CALIB, tol_cm=5.0)
    assert w[60] == pytest.approx(5.0 / 25.0, abs=1e-5)
    assert np.all(np.diff(w) <= 1e-6)  # monotonically non-increasing


def test_extrapolates_beyond_calibration_instead_of_holding_flat():
    w = horizon_weights(121, 30.0, CALIB, tol_cm=5.0)
    assert w[120] < w[60]  # 4s is past the 2s calibration, error keeps growing


def test_floor_bounds_the_weight_from_below():
    w = horizon_weights(121, 30.0, CALIB, tol_cm=5.0, floor=0.25)
    assert w.min() >= 0.25


def test_shipped_calibration_matches_measured_experiment(tmp_path):
    calib = load_calibration("calibration/pseudo_label_reliability.json")
    assert calib["seconds"][0] == 0.0
    assert np.all(np.diff(calib["label_err_cm"]) >= 0)  # monotone envelope
    w = horizon_weights(128, 30.0, calib)
    assert w[0] == 1.0
    assert w[36] == 1.0  # measured: labels stay under 5cm for ~1.2s
    assert w[127] < 0.2


def test_gt_datasets_get_uniform_weight(tmp_path):
    from foretrack.data.mixed import pseudo_label_weight
    assert pseudo_label_weight({"data": {}}, 128) is None
    assert pseudo_label_weight({"data": {"pseudo_label_weighting": {"enabled": False}}}, 128) is None
    cal = tmp_path / "c.json"
    cal.write_text(json.dumps(CALIB))
    cfg = {"data": {"pseudo_label_weighting": {"enabled": True, "calibration": str(cal)}}}
    w = pseudo_label_weight(cfg, 128)
    assert w is not None and w.shape == (128,)
