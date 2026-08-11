import argparse

import torch
from torch.utils.data import DataLoader

from foretrack.data.dexycb import DexYCBTracks
from foretrack.models.conditioning import ImageEncoder, QueryTokenizer
from foretrack.models.denoiser import TrackDenoiser
from foretrack.models.diffusion import GaussianDiffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--transl_stats", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ds = DexYCBTracks(args.root, args.split, n=64, t=128, transl_stats_path=args.transl_stats)
    print(f"dataset size: {len(ds)}")

    item = ds[0]
    for k, v in item.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v}")

    assert item["tracks"].shape == (128, 64, 3)
    assert item["visibility"].shape == (128, 64)
    assert item["frame_mask"].shape == (128,)
    assert item["depth"].shape == (128, 64)
    assert item["query_xyz_t0"].shape == (64, 3)
    assert item["query_uv"].shape == (64, 2)
    assert item["image"].shape[0] == 3
    assert item["frame_mask"].sum() > 0 and item["frame_mask"].sum() <= 128
    assert torch.isfinite(item["tracks"]).all()
    assert torch.isfinite(item["depth"]).all()
    assert (item["depth"] > 0).all(), "depth must be positive (camera-frame Z)"

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"batch tracks: {tuple(batch['tracks'].shape)}")
    assert batch["tracks"].shape == (4, 128, 64, 3)

    bz, n, latent_dim = 4, 64, 64
    image_encoder = ImageEncoder(vit_init="random").to(device)
    query_tokenizer = QueryTokenizer(latent_dim=latent_dim, dropout_prob=0.1, vit_feat_dim=1280).to(device)
    denoiser = TrackDenoiser(n=n, t=128, latent_dim=latent_dim, num_layers=2, num_heads=2, ff_size=128).to(device)
    diffusion = GaussianDiffusion(num_steps=1000)

    images = batch["image"].to(device)
    query_xyz_t0 = batch["query_xyz_t0"].to(device)
    query_uv = batch["query_uv"].to(device)
    orig_h = batch["orig_image_size"][0][0].item()
    orig_w = batch["orig_image_size"][1][0].item()
    tracks = batch["tracks"].to(device)
    depth = batch["depth"].to(device)

    patch_feats = image_encoder(images)
    cond = query_tokenizer(query_xyz_t0, patch_feats, query_uv, orig_image_size=(orig_h, orig_w))
    cond = cond.transpose(0, 1)

    def model_fn(x_noisy, timestep, cond, query_xyz_t0):
        return denoiser(x_noisy, timestep, cond, query_xyz_t0)

    loss = diffusion.training_losses(model_fn, tracks, cond, query_xyz_t0, depth)
    print(f"end-to-end training loss on REAL data: {loss.item():.4f}")
    assert torch.isfinite(loss)
    loss.backward()

    print("SMOKE_TEST_PASSED")


if __name__ == "__main__":
    main()
