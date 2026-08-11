# Out-of-distribution fallback for the diffusion model: when the model's conditioning
# (image + query tokens) isn't actually being used to distinguish a specific scene, the diffusion
# sampler was found to converge to a confident but essentially uncorrelated-with-truth motion
# prediction rather than falling back to low-confidence/no-motion behavior.
# Detected cheaply at inference time: run the sampler twice, once with real conditioning and once
# with disable_query_cond=True (the same "unconditional" mode classifier-free guidance samples
# from). If the two predictions nearly agree, conditioning isn't contributing meaningfully for
# this input -- fall back to the static (no-motion) baseline, which is the empirically best
# available prediction whenever that happens, rather than trust either diffusion output.
#
# Threshold provenance: 3.659cm, the 5th percentile of the pooled (DexYCB val + HoloAssist pseudo
# val) conditioned-vs-unconditional disagreement distribution (n=843), i.e. the VALUE was
# computed entirely from in-domain data, no EgoExo4D data went into the percentile calculation.
# p5 follows standard anomaly-detection convention; p1/p5/p10 candidates were compared on
# EgoExo4D in the same pass, so the selection was not fully blind to that benchmark.
#
# This constant is specific to the checkpoint it was calibrated against
# (mixed_stage2_diffusion_lowlr_expanded/diffusion/epoch20.pt) and that checkpoint's coordinate
# normalization (downloads/stats/pseudo_holoassist_transl_stats.json) -- the raw disagreement
# scale depends on both the trained weights and the units predictions are denormalized into.
# Retraining the diffusion model, changing its architecture/capacity, or changing normalization
# stats invalidates this number; recalibrate by rerunning the pooled-in-domain-percentile
# procedure above against the new checkpoint before trusting the gate.

import numpy as np

DISAGREEMENT_THRESHOLD_CM = 3.659


def conditioned_disagreement(pred_cond: np.ndarray, pred_uncond: np.ndarray, mask: np.ndarray = None) -> float:
    """mean per-point-per-frame L2 distance between the conditioned and unconditional diffusion
    predictions, in cm. pred_cond, pred_uncond: (T, N, 3). mask: optional (T,) bool."""
    err = np.linalg.norm(pred_cond - pred_uncond, axis=-1)  # (T, N)
    if mask is not None:
        err = err[mask]
    return float(err.mean() * 100)


def gated_prediction(pred_cond: np.ndarray, pred_uncond: np.ndarray, pred_static: np.ndarray, mask: np.ndarray = None, threshold: float = DISAGREEMENT_THRESHOLD_CM) -> np.ndarray:
    """returns pred_static if the conditioned/unconditional disagreement falls below threshold
    (conditioning isn't contributing meaningfully -- likely out-of-distribution input), else
    pred_cond. All arrays (T, N, 3)."""
    if conditioned_disagreement(pred_cond, pred_uncond, mask) < threshold:
        return pred_static
    return pred_cond
