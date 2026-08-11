# ViT backbone (HaMeR-init) + per-query-point tokenizer, adapted from forehand4d's
# src/models/mdm/model/encoder.py (ImageEncoder, SpatialConditioner). SpatialConditioner turns
# every ViT patch into a token; decision 2 instead samples the patch grid at each of our 64
# query points' 2D projections, concatenated with a Fourier position embedding, one token per
# query point rather than one per patch.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import vit


class ImageEncoder(nn.Module):
    def __init__(self, vit_init: str = "hamer"):
        super().__init__()
        self.vit_init = vit_init
        self.vit_input_size = (256, 192)  # matches HaMeR's pretrained input resolution
        self.backbone = vit(cfg=None, img_size=self.vit_input_size)
        if vit_init == "hamer":
            self._load_hamer_weights()

    def _load_hamer_weights(self):
        ckpt_path = os.path.join(os.environ["DOWNLOADS_DIR"], "model", "hamer", "hamer.ckpt")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
        state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(state_dict, strict=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: (B, 3, H, W) -> (B, C, Hp, Wp) patch feature grid. DexYCBTracks now crops
        # around the object before this point (see compute_object_crop's docstring), so the
        # input already has the right 256:192 aspect -- a direct resize is correct here. The
        # old resize-to-square-then-crop-32px-each-side was a hack specifically for full-scene
        # (uncropped) input and would distort an already object-centric crop, so it's gone.
        if image.shape[-2:] != self.vit_input_size:
            image = F.interpolate(image, size=self.vit_input_size, mode="bilinear", align_corners=False)
        return self.backbone(image)


def fourier_embed(x: torch.Tensor, freq: int = 8) -> torch.Tensor:
    """NeRF-style positional encoding: sin/cos at increasing frequencies.
    x: (..., C) -> (..., C * freq * 2)"""
    freq_bands = 2 ** torch.arange(freq, device=x.device, dtype=x.dtype)
    x_expand = x.unsqueeze(-1) * freq_bands  # (..., C, freq)
    return torch.stack([x_expand.sin(), x_expand.cos()], dim=-1).flatten(-3)


def project_to_vit_frame(uv: torch.Tensor, orig_image_size: tuple, vit_input_size: tuple = (256, 192)) -> torch.Tensor:
    """ImageEncoder resizes the input to a `max(vit_input_size)` square, then crops 32px off
    each side of the width -- so patch_feats' spatial grid represents that resized+cropped
    frame, not the original image. Query pixel coords must go through the identical transform
    before being used to sample patch_feats, or they land on the wrong patch entirely.
    uv: (..., 2) pixel coords in the ORIGINAL image. -> (..., 2) in the (256, 192) ViT frame."""
    orig_h, orig_w = orig_image_size
    resize_to = max(vit_input_size)
    scale_x = resize_to / orig_w
    scale_y = resize_to / orig_h
    crop = (resize_to - min(vit_input_size)) // 2  # 32 for (256, 192)
    u = uv[..., 0] * scale_x - crop
    v = uv[..., 1] * scale_y
    return torch.stack([u, v], dim=-1)


def sample_patch_features(patch_feats: torch.Tensor, uv_vit_frame: torch.Tensor, vit_input_size: tuple) -> torch.Tensor:
    """bilinearly sample the ViT patch feature grid at 2D pixel coords already expressed in the
    ViT's own (post resize+crop) frame -- see project_to_vit_frame.
    patch_feats: (B, C, Hp, Wp). uv_vit_frame: (B, N, 2). -> (B, N, C)."""
    h, w = vit_input_size
    u_norm = uv_vit_frame[..., 0] / (w - 1) * 2 - 1
    v_norm = uv_vit_frame[..., 1] / (h - 1) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(1)  # (B, 1, N, 2)
    sampled = F.grid_sample(patch_feats, grid, mode="bilinear", align_corners=True)  # (B, C, 1, N)
    return sampled.squeeze(2).transpose(1, 2)  # (B, N, C)


class QueryTokenizer(nn.Module):
    """per query point: concat [Fourier(xyz_t0), ViT patch feature at 2D projection] -> linear
    -> one conditioning token each (decision 2). independent 0.1 dropout per (batch, query
    point), matching forehand4d's mask_spatial exactly (not a single shared mask for the whole
    block)."""

    def __init__(self, latent_dim: int = 512, dropout_prob: float = 0.1, fourier_freq: int = 8, vit_feat_dim: int = 1280, vit_input_size: tuple = (256, 192), disable_query_cond: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob
        self.fourier_freq = fourier_freq
        self.vit_input_size = vit_input_size
        # query-conditioning ablation: unlike dropout_prob, this zeros
        # tokens unconditionally in BOTH train and eval mode -- dropout_prob alone only masks
        # during .train(), so a model trained at dropout_prob=1.0 would still see live,
        # never-before-seen query tokens at eval time, which isn't a true "conditioning off"
        # test. query_xyz_t0 still reaches TrackHead directly (decision 3's offset-from-query
        # design), so this ablates the image+Fourier conditioning specifically, not the
        # architecture's baseline query-position wiring.
        self.disable_query_cond = disable_query_cond
        in_dim = 3 * fourier_freq * 2 + vit_feat_dim
        self.proj = nn.Linear(in_dim, latent_dim)

    def forward(self, query_xyz_t0: torch.Tensor, patch_feats: torch.Tensor, query_uv: torch.Tensor, orig_image_size: tuple) -> torch.Tensor:
        # query_xyz_t0: (B, N, 3) camera-frame, normalized. patch_feats: (B, C, Hp, Wp).
        # query_uv: (B, N, 2) pixel coords in the ORIGINAL (unresized) image.
        if self.disable_query_cond:
            return torch.zeros(query_xyz_t0.shape[0], query_xyz_t0.shape[1], self.latent_dim, device=query_xyz_t0.device, dtype=query_xyz_t0.dtype)
        pos_embed = fourier_embed(query_xyz_t0, freq=self.fourier_freq)  # (B, N, 3*freq*2)
        uv_vit_frame = project_to_vit_frame(query_uv, orig_image_size, self.vit_input_size)
        img_feat = sample_patch_features(patch_feats, uv_vit_frame, self.vit_input_size)  # (B, N, C)
        tokens = self.proj(torch.cat([pos_embed, img_feat], dim=-1))  # (B, N, D)

        if self.training and self.dropout_prob > 0:
            bz, n = tokens.shape[:2]
            mask = torch.bernoulli(torch.full((bz, n, 1), self.dropout_prob, device=tokens.device))
            tokens = tokens * (1.0 - mask)

        return tokens
