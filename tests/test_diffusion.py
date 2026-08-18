import torch

from foretrack.models.diffusion import GaussianDiffusion, depth_scaled_l2


def test_p_sample_matches_standard_ddpm_posterior_at_single_step():
    torch.manual_seed(0)
    diffusion = GaussianDiffusion(num_steps=50)
    bz = 4
    fixed_pred_x0 = torch.randn(bz, 2, 3, 3)

    def model(x_noisy, timestep, cond, query_xyz_t0):
        return fixed_pred_x0, None

    t = torch.full((bz,), 20, dtype=torch.long)
    x_t = torch.randn(bz, 2, 3, 3)
    query_xyz_t0 = torch.randn(bz, 3, 3)

    torch.manual_seed(1)
    out = diffusion.p_sample(model, x_t, t, cond=None, query_xyz_t0=query_xyz_t0, t_prev=t - 1)

    alpha_bar_t = diffusion.alphas_cumprod[20]
    alpha_bar_prev = diffusion.alphas_cumprod[19]
    beta_t = 1 - alpha_bar_t / alpha_bar_prev

    expected_mean = (
        torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t) * fixed_pred_x0
        + torch.sqrt(1 - beta_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x_t
    )
    expected_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

    torch.manual_seed(1)
    expected_noise = torch.randn_like(x_t)
    expected = expected_mean + torch.sqrt(expected_var) * expected_noise

    assert torch.allclose(out, expected, atol=1e-4)


def test_p_sample_eta_zero_is_deterministic():
    diffusion = GaussianDiffusion(num_steps=50)
    bz = 4
    fixed_pred_x0 = torch.randn(bz, 2, 3, 3)

    def model(x_noisy, timestep, cond, query_xyz_t0):
        return fixed_pred_x0, None

    t = torch.full((bz,), 20, dtype=torch.long)
    x_t = torch.randn(bz, 2, 3, 3)
    query_xyz_t0 = torch.randn(bz, 3, 3)

    torch.manual_seed(1)
    out_a = diffusion.p_sample(model, x_t, t, cond=None, query_xyz_t0=query_xyz_t0, t_prev=t - 1, eta=0.0)
    torch.manual_seed(999)
    out_b = diffusion.p_sample(model, x_t, t, cond=None, query_xyz_t0=query_xyz_t0, t_prev=t - 1, eta=0.0)

    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_p_sample_variance_shrinks_relative_to_forward_marginal():
    diffusion = GaussianDiffusion(num_steps=1000)
    t, t_prev = 200, 100
    alpha_bar_t = diffusion.alphas_cumprod[t]
    alpha_bar_prev = diffusion.alphas_cumprod[t_prev]
    alpha_eff = alpha_bar_t / alpha_bar_prev
    posterior_var = (1 - alpha_eff) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
    forward_marginal_var = 1 - alpha_bar_prev
    assert posterior_var < forward_marginal_var


def test_depth_scaled_l2_matches_manual_weighted_mean():
    pred = torch.randn(2, 3, 4, 3)
    target = torch.randn(2, 3, 4, 3)
    depth = torch.rand(2, 3, 4) + 0.5
    loss = depth_scaled_l2(pred, target, depth)

    sq_err = (pred - target).pow(2).sum(dim=-1)
    weight = 1.0 / depth
    expected = (sq_err * weight).sum() / weight.sum()
    assert torch.isclose(loss, expected, atol=1e-5)


def test_depth_scaled_l2_mask_excludes_padded_frames():
    pred = torch.randn(1, 4, 2, 3)
    target = pred.clone()
    pred[:, 2:] += 1000.0
    depth = torch.ones(1, 4, 2)
    mask = torch.tensor([[True, True, False, False]])

    loss_masked = depth_scaled_l2(pred, target, depth, mask=mask)
    assert torch.isclose(loss_masked, torch.tensor(0.0), atol=1e-5), "masked-out garbage frames leaked into the loss"

    loss_unmasked = depth_scaled_l2(pred, target, depth)
    assert loss_unmasked > 100, "sanity check: without the mask this should NOT be near zero"


def test_training_losses_with_mask_end_to_end():
    diffusion = GaussianDiffusion(num_steps=100)
    bz, t_len, n = 2, 5, 3

    def identity_model(x_noisy, timestep, cond, query_xyz_t0):
        return torch.zeros_like(x_noisy), None

    x0 = torch.randn(bz, t_len, n, 3)
    cond = torch.zeros(1, bz, 8)
    query_xyz_t0 = torch.randn(bz, n, 3)
    depth = torch.rand(bz, t_len, n) + 0.5
    mask = torch.ones(bz, t_len, dtype=torch.bool)
    mask[:, -2:] = False

    loss = diffusion.training_losses(identity_model, x0, cond, query_xyz_t0, depth, mask=mask)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_offset_reg_weight_ignores_diverse_but_unbiased_predictions():
    torch.manual_seed(0)
    bz, t_len, n = 8, 2, 3
    x0 = torch.randn(bz, t_len, n, 3)
    perturbation = torch.randn(bz, t_len, n, 3) * 5.0
    perturbation = perturbation - perturbation.mean(dim=0, keepdim=True)
    pred_x0_diverse_but_unbiased = x0 + perturbation

    mean_bias = (pred_x0_diverse_but_unbiased.mean(dim=0) - x0.mean(dim=0)).pow(2).mean()
    assert mean_bias.item() < 1e-4, f"diverse-but-unbiased predictions should get ~0 penalty, got {mean_bias.item()}"


def test_offset_reg_weight_penalizes_systematic_batch_bias():
    torch.manual_seed(0)
    bz, t_len, n = 8, 2, 3
    x0 = torch.randn(bz, t_len, n, 3)
    pred_x0_biased = x0 + 3.0

    mean_bias = (pred_x0_biased.mean(dim=0) - x0.mean(dim=0)).pow(2).mean()
    assert mean_bias.item() > 1.0, f"a real systematic bias should produce a large mean_bias term, got {mean_bias.item()}"


def test_training_losses_offset_reg_weight_end_to_end():
    diffusion = GaussianDiffusion(num_steps=100)
    bz, t_len, n = 8, 2, 3
    torch.manual_seed(0)
    x0 = torch.randn(bz, t_len, n, 3)
    cond = torch.zeros(1, bz, 8)
    query_xyz_t0 = torch.zeros(bz, n, 3)
    depth = torch.ones(bz, t_len, n)

    def biased_model(x_noisy, timestep, cond, query_xyz_t0):
        return x0 + 3.0, None

    def unbiased_model(x_noisy, timestep, cond, query_xyz_t0):
        return x0.clone(), None

    torch.manual_seed(1)
    loss_biased = diffusion.training_losses(biased_model, x0, cond, query_xyz_t0, depth, offset_reg_weight=1.0)
    torch.manual_seed(1)
    loss_unbiased = diffusion.training_losses(unbiased_model, x0, cond, query_xyz_t0, depth, offset_reg_weight=1.0)

    assert loss_biased.item() > loss_unbiased.item() + 1.0
