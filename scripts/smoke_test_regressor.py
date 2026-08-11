import torch

from foretrack.models.conditioning import ImageEncoder, QueryTokenizer
from foretrack.models.regressor import TrackRegressor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    bz, n, t_len, latent_dim = 2, 64, 8, 64
    img_h, img_w = 256, 256

    image_encoder = ImageEncoder(vit_init="random").to(device)
    query_tokenizer = QueryTokenizer(latent_dim=latent_dim, dropout_prob=0.1, vit_feat_dim=1280).to(device)
    regressor = TrackRegressor(n=n, t=t_len, latent_dim=latent_dim, num_layers=2, num_heads=2, ff_size=128).to(device)

    images = torch.randn(bz, 3, img_h, img_w, device=device)
    query_xyz_t0 = torch.randn(bz, n, 3, device=device)
    query_uv = torch.rand(bz, n, 2, device=device) * torch.tensor([img_w - 1, img_h - 1], device=device)

    patch_feats = image_encoder(images)
    cond = query_tokenizer(query_xyz_t0, patch_feats, query_uv, orig_image_size=(img_h, img_w))
    cond = cond.transpose(0, 1)  # (N, B, D)

    pred, vis_logits = regressor(cond, query_xyz_t0)
    print(f"regressor output: {tuple(pred.shape)}, visibility logits: {tuple(vis_logits.shape)}")
    assert pred.shape == (bz, t_len, n, 3), f"shape mismatch: {pred.shape}"
    assert vis_logits.shape == (bz, t_len, n), f"visibility shape mismatch: {vis_logits.shape}"

    target = torch.randn(bz, t_len, n, 3, device=device)
    loss = (pred - target).pow(2).mean()
    loss.backward()

    grad_reg = any(p.grad is not None and p.grad.abs().sum() > 0 for p in regressor.parameters())
    print(f"gradients flowed into regressor: {grad_reg}")
    assert grad_reg

    # TrackHead.proj is zero-initialized (predicts an offset from query_xyz_t0, see
    # track_head.py) -- at exact init, d(loss)/d(tokens) = weight^T @ ... is exactly zero, so
    # only the head's own weight/bias get gradient on this very first step (ControlNet's "zero
    # convolution" pattern, intentional). Take one step to move the head off zero before
    # checking gradient flow to the conditioning pathway.
    opt = torch.optim.SGD(regressor.parameters(), lr=0.1)
    opt.step()
    opt.zero_grad()
    query_tokenizer.zero_grad()

    # cond's (and patch_feats') graph was already consumed by the first backward() -- recompute
    # the whole conditioning chain fresh rather than reuse either tensor.
    patch_feats2 = image_encoder(images)
    cond2 = query_tokenizer(query_xyz_t0, patch_feats2, query_uv, orig_image_size=(img_h, img_w)).transpose(0, 1)
    pred2, _ = regressor(cond2, query_xyz_t0)
    loss2 = (pred2 - target).pow(2).mean()
    loss2.backward()
    grad_cond = any(p.grad is not None and p.grad.abs().sum() > 0 for p in query_tokenizer.parameters())
    print(f"gradients flowed into query_tokenizer (after 1 head update): {grad_cond}")
    assert grad_cond, "no gradients reached the query tokenizer even after the head moved off zero-init"

    print("SMOKE_TEST_PASSED")


if __name__ == "__main__":
    main()
