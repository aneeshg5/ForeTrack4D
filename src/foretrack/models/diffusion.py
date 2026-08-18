# Cosine schedule and x0 prediction adapted from forehand4d's
# src/models/mdm/diffusion/gaussian_diffusion.py. Depth-scaled loss follows TAPIP3D. See NOTICE.md.

import math

import torch
import torch.nn.functional as F


def cosine_schedule(num_steps: int = 1000, max_beta: float = 0.999, s: float = 0.008) -> torch.Tensor:

    def alpha_bar(t):
        return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

    betas = []
    for i in range(num_steps):
        t1 = i / num_steps
        t2 = (i + 1) / num_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


def depth_scaled_l2(pred: torch.Tensor, target: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor = None, depth_scaled: bool = True) -> torch.Tensor:
    sq_err = (pred - target).pow(2).sum(dim=-1)  # (B, T, N)
    weight = 1.0 / depth.clamp(min=1e-3) if depth_scaled else torch.ones_like(depth)
    if mask is not None:
        weight = weight * mask.unsqueeze(-1).to(weight.dtype)
    return (sq_err * weight).sum() / weight.sum().clamp(min=1e-8)


def visibility_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    per_point = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    if mask is not None:
        m = mask.unsqueeze(-1).to(per_point.dtype)
        return (per_point * m).sum() / m.expand_as(per_point).sum().clamp(min=1e-8)
    return per_point.mean()


class GaussianDiffusion:

    def __init__(self, num_steps: int = 1000):
        self.num_steps = num_steps
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
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omac * noise

    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, query_xyz_t0: torch.Tensor, t_prev: torch.Tensor = None, eta: float = 1.0) -> torch.Tensor:
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
