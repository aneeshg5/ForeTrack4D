import torch

from foretrack.models.conditioning import ImageEncoder, QueryTokenizer
from foretrack.models.denoiser import TrackDenoiser
from foretrack.models.diffusion import GaussianDiffusion


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    bz, n, t_len, latent_dim = 2, 64, 8, 64
    img_h, img_w = 256, 256

    image_encoder = ImageEncoder(vit_init="random").to(device)
    query_tokenizer = QueryTokenizer(latent_dim=latent_dim, dropout_prob=0.1, vit_feat_dim=1280).to(device)
    denoiser = TrackDenoiser(n=n, t=t_len, latent_dim=latent_dim, num_layers=2, num_heads=2, ff_size=128).to(device)
    diffusion = GaussianDiffusion(num_steps=1000)

    images = torch.randn(bz, 3, img_h, img_w, device=device)
    query_xyz_t0 = torch.randn(bz, n, 3, device=device)
    query_uv = torch.rand(bz, n, 2, device=device) * torch.tensor([img_w - 1, img_h - 1], device=device)

    patch_feats = image_encoder(images)
    print(f"patch_feats: {tuple(patch_feats.shape)}")

    cond = query_tokenizer(query_xyz_t0, patch_feats, query_uv, orig_image_size=(img_h, img_w))
    print(f"cond tokens: {tuple(cond.shape)}")
    cond = cond.transpose(0, 1)  # (N, B, D) for the transformer's seqlen-first convention

    x0 = torch.randn(bz, t_len, n, 3, device=device)
    x0[..., 2] = x0[..., 2].abs() + 0.5  # plausible positive depth

    def model_fn(x_noisy, timestep, cond, query_xyz_t0):
        return denoiser(x_noisy, timestep, cond, query_xyz_t0)

    t = torch.randint(0, 1000, (bz,), device=device)
    x_t = diffusion.q_sample(x0, t)
    print(f"q_sample output: {tuple(x_t.shape)}")

    pred, vis_logits = model_fn(x_t, t, cond, query_xyz_t0)
    print(f"denoiser output: {tuple(pred.shape)}, visibility logits: {tuple(vis_logits.shape)}")
    assert pred.shape == x0.shape, f"shape mismatch: {pred.shape} vs {x0.shape}"
    assert vis_logits.shape == (bz, t_len, n), f"visibility shape mismatch: {vis_logits.shape}"

    depth = x0[..., 2].clone()
    visibility_gt = torch.randint(0, 2, (bz, t_len, n), device=device).bool()
    loss = diffusion.training_losses(model_fn, x0, cond, query_xyz_t0, depth, visibility_gt=visibility_gt)
    print(f"training loss (with visibility term): {loss.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"

    loss.backward()
    grad_found = any(p.grad is not None and p.grad.abs().sum() > 0 for p in denoiser.parameters())
    print(f"gradients flowed into denoiser: {grad_found}")
    assert grad_found, "no gradients reached the denoiser"

    opt = torch.optim.SGD(denoiser.parameters(), lr=0.1)
    opt.step()
    opt.zero_grad()
    query_tokenizer.zero_grad()

    patch_feats2 = image_encoder(images)
    cond2 = query_tokenizer(query_xyz_t0, patch_feats2, query_uv, orig_image_size=(img_h, img_w)).transpose(0, 1)
    loss2 = diffusion.training_losses(model_fn, x0, cond2, query_xyz_t0, depth)
    loss2.backward()
    grad_found_cond = any(p.grad is not None and p.grad.abs().sum() > 0 for p in query_tokenizer.parameters())
    print(f"gradients flowed into query_tokenizer (after 1 head update): {grad_found_cond}")
    assert grad_found_cond, "no gradients reached the query tokenizer even after the head moved off zero-init"

    sample = diffusion.p_sample(model_fn, x_t, t, cond, query_xyz_t0)  # returns tracks only, visibility discarded at sampling time
    print(f"p_sample output: {tuple(sample.shape)}")
    assert sample.shape == x0.shape

    print("SMOKE_TEST_PASSED")


if __name__ == "__main__":
    main()
