import numpy as np

from foretrack.eval.ood_gate import (
    DISAGREEMENT_THRESHOLD_CM,
    conditioned_disagreement,
    gated_prediction,
)


def test_conditioned_disagreement_zero_when_identical():
    pred = np.random.randn(8, 4, 3).astype(np.float32)
    assert conditioned_disagreement(pred, pred.copy()) == 0.0


def test_conditioned_disagreement_respects_mask():
    pred_cond = np.zeros((8, 4, 3), dtype=np.float32)
    pred_uncond = np.zeros((8, 4, 3), dtype=np.float32)
    pred_uncond[4:] += 1.0  # 1m offset, only in frames excluded by the mask
    mask = np.array([True] * 4 + [False] * 4)
    assert conditioned_disagreement(pred_cond, pred_uncond, mask) == 0.0


def test_gated_prediction_falls_back_to_static_below_threshold():
    pred_cond = np.zeros((8, 4, 3), dtype=np.float32)
    pred_uncond = pred_cond.copy()  # zero disagreement -- below any positive threshold
    pred_static = np.ones((8, 4, 3), dtype=np.float32)
    out = gated_prediction(pred_cond, pred_uncond, pred_static)
    np.testing.assert_array_equal(out, pred_static)


def test_gated_prediction_trusts_conditioned_above_threshold():
    pred_cond = np.zeros((8, 4, 3), dtype=np.float32)
    pred_uncond = np.ones((8, 4, 3), dtype=np.float32) * 10.0  # 10m disagreement, way above threshold
    pred_static = np.ones((8, 4, 3), dtype=np.float32) * -1.0
    out = gated_prediction(pred_cond, pred_uncond, pred_static)
    np.testing.assert_array_equal(out, pred_cond)


def test_gated_prediction_custom_threshold():
    pred_cond = np.zeros((8, 4, 3), dtype=np.float32)
    offset = 0.02 / np.sqrt(3)  # per-axis offset giving an exact 2cm L2 disagreement
    pred_uncond = np.full((8, 4, 3), offset, dtype=np.float32)
    pred_static = np.ones((8, 4, 3), dtype=np.float32)
    assert abs(conditioned_disagreement(pred_cond, pred_uncond) - 2.0) < 1e-3
    # 2cm disagreement: gated with a 5cm threshold, trusted with a 1cm threshold
    np.testing.assert_array_equal(gated_prediction(pred_cond, pred_uncond, pred_static, threshold=5.0), pred_static)
    np.testing.assert_array_equal(gated_prediction(pred_cond, pred_uncond, pred_static, threshold=1.0), pred_cond)


def test_default_threshold_matches_calibrated_value():
    # regression guard: this constant is calibrated against real data --
    # changing it silently would be a real behavior change, not a refactor.
    assert DISAGREEMENT_THRESHOLD_CM == 3.659
