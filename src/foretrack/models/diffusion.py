# cosine schedule + x0 prediction: adapted from forehand4d's
# src/models/mdm/diffusion/gaussian_diffusion.py (get_named_beta_schedule's "cosine" branch,
# training_losses' ModelMeanType.START_X path), itself a port of openai/improved-diffusion.
# forehand4d hardcodes schedule_sampler_type='uniform' (model_.py), whose UniformSampler
# assigns uniform per-timestep weights -- i.e. their default "multi-iteration" loss is a single
# uniformly-sampled timestep per batch with no extra reweighting, which is what's implemented
# here. depth-scaled loss is TAPIP3D's depth-adaptive weighting (decision 6), not from
# forehand4d.

import math

import torch
import torch.nn.functional as F


def cosine_schedule(num_steps: int = 1000, max_beta: float = 0.999, s: float = 0.008) -> torch.Tensor:
    """betas_for_alpha_bar with the cosine alpha_bar (Nichol & Dhariwal), matching forehand4d's
    schedule_name == 'cosine'."""

    def alpha_bar(t):
        return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

    betas = []
    for i in range(num_steps):
        t1 = i / num_steps
        t2 = (i + 1) / num_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


def depth_scaled_l2(pred: torch.Tensor, target: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor = None, depth_scaled: bool = True) -> torch.Tensor:
    """per-point-per-timestep L2 weighted by 1/depth (TAPIP3D's depth-adaptive weighting,
    decision 6) so far points don't dominate. pred, target: (B, T, N, 3). depth: (B, T, N) > 0.
    mask: optional (B, T) bool -- zeroes out padded frames (DexYCBTracks pads short sequences
    by repeating the last frame; without this those repeated frames would still contribute to
    the loss, just with sane rather than exploding values).
    depth_scaled: ablation switch -- False gives plain (still masked) mean L2,
    isolating whether the depth-adaptive weighting itself matters vs. just training at all."""
    sq_err = (pred - target).pow(2).sum(dim=-1)  # (B, T, N)
    weight = 1.0 / depth.clamp(min=1e-3) if depth_scaled else torch.ones_like(depth)
    if mask is not None:
        weight = weight * mask.unsqueeze(-1).to(weight.dtype)
    return (sq_err * weight).sum() / weight.sum().clamp(min=1e-8)


def visibility_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """binary CE for the visibility head (ForeHand4D's alpha_vis=3.0 weighting applied by
    the caller). logits, target: (B, T, N).
    mask: optional (B, T) bool, same padded-frame exclusion as depth_scaled_l2."""
    per_point = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    if mask is not None:
        m = mask.unsqueeze(-1).to(per_point.dtype)
        return (per_point * m).sum() / m.expand_as(per_point).sum().clamp(min=1e-8)
    return per_point.mean()


class GaussianDiffusion:
    """x0-prediction, cosine schedule (decision 6). Fixed (non-learned) variance, matching
    forehand4d's sigma_small config -- no KL/variational-bound terms since those only matter
    for learned variance."""

    def __init__(self, num_steps: int = 1000):
        self.num_steps = num_steps
        # cumprod in float64 to avoid precision loss over 1000 steps, then cast to float32 to
        # match the rest of the (float32) training pipeline -- caught by the smoke test: a
        # stray float64 schedule silently upcast every downstream tensor and crashed the
        # denoiser's float32 linear layers.
        betas = cosine_schedule(num_steps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.betas = betas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()

    def _extract(self, table: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        shape = [x_shape[0]] + [1] * (len(x_shape) - 1)
        return table.to(t.device)[t].reshape(shape)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """forward diffusion: sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omac * noise

    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, query_xyz_t0: torch.Tensor, t_prev: torch.Tensor = None, eta: float = 1.0) -> torch.Tensor:
        """one reverse-diffusion step, driven by the model's x0 prediction. t_prev defaults to
        t-1 (every timestep) but can be set explicitly for strided/respaced sampling (e.g. eval
        time, where running all 1000 steps per sample is too slow).

        DDIM's generalized non-Markovian formula (Song et al. 2020, eq 12), which interpolates
        between fully stochastic ancestral sampling (eta=1, exactly the true DDPM posterior
        q(x_t_prev | x_t, x0=pred_x0), generalized to an arbitrary jump from t to t_prev via
        alpha_eff = alpha_bar_t / alpha_bar_t_prev) and fully deterministic sampling (eta=0, no
        injected noise at all). eta=1 is what this function computed before -- verified
        equivalent by `test_p_sample_matches_standard_ddpm_posterior_at_single_step` and a new
        eta-equivalence test. Tried eta=0 at eval time after the offset-parameterization fix
        (attempt 5) landed diffusion at roughly parity with the static baseline rather than a
        clean win -- This DOES depend on x_t, not just
        pred_x0 -- an earlier version of this function instead re-ran the FORWARD process on
        pred_x0 with fresh noise, which discards x_t entirely and samples from the marginal
        q(x_t_prev | x0) instead of the true conditional."""
        pred_x0, _ = model(x_t, t, cond, query_xyz_t0)  # visibility logits unused at sampling time
        if t_prev is None:
            t_prev = t - 1
        if bool((t_prev < 0).any()):
            return pred_x0

        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
        alpha_eff = (alpha_bar_t / alpha_bar_prev).clamp(max=1.0)

        posterior_var = ((1 - alpha_eff) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)).clamp(min=0)
        sigma = eta * torch.sqrt(posterior_var)

        eps_pred = (x_t - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt((1 - alpha_bar_t).clamp(min=1e-8))
        dir_coeff = (1 - alpha_bar_prev - sigma.pow(2)).clamp(min=0).sqrt()
        mean = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_coeff * eps_pred

        noise = torch.randn_like(x_t)
        return mean + sigma * noise

    def training_losses(self, model, x0: torch.Tensor, cond: torch.Tensor, query_xyz_t0: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor = None, offset_reg_weight: float = 0.0, depth_scaled: bool = True, visibility_gt: torch.Tensor = None, alpha_vis: float = 3.0) -> torch.Tensor:
        """sample a random timestep and noise, denoise, depth-scaled L2 against x0 (decision 6).
        offset_reg_weight: optional penalty matching the BATCH-MEAN of pred_x0 to the batch-mean
        of true x0, discouraging the model from systematically over/under-shooting the dataset's
        true average displacement. Added after diagnosing a real, train-set-present (not a
        generalization gap) ~60% systematic overshoot of the dataset's true mean displacement
        direction, "closing the last bit of the ADE gap".

        Two earlier versions of this term (attempts 6 and 7) penalized PER-EXAMPLE
        |pred_x0 - query_xyz_t0|^2, flat and then confidence-scaled by alpha_bar_t. Both
        conflated the diagnosed bias (a systematic AGGREGATE tendency) with genuine per-example
        motion, since any individual example's large, correct predicted displacement got
        penalized identically to the systematic overshoot -- collapsing diffusion's sample
        diversity from 11-13cm down to 3-4cm as a side effect, confidence-scaling only partially
        recovering it. Matching BATCH means instead only constrains the aggregate/population
        statistic across the examples in a training batch; it places no constraint on any
        individual example's prediction, so per-example (and per-noise-realization) diversity
        is untouched by construction, not by tuning a schedule. No confidence scaling needed for
        the same reason -- the mechanism that protects diversity here is structural."""
        bz = x0.shape[0]
        t = torch.randint(0, self.num_steps, (bz,), device=x0.device)
        x_t = self.q_sample(x0, t)
        pred_x0, vis_logits = model(x_t, t, cond, query_xyz_t0)
        loss = depth_scaled_l2(pred_x0, x0, depth, mask, depth_scaled=depth_scaled)
        if offset_reg_weight > 0:
            mean_bias = (pred_x0.mean(dim=0) - x0.mean(dim=0)).pow(2).mean()
            loss = loss + offset_reg_weight * mean_bias
        if visibility_gt is not None:
            loss = loss + alpha_vis * visibility_bce(vis_logits, visibility_gt, mask)
        return loss
