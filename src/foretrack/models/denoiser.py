# adapted from forehand4d's src/models/mdm/model/mdm.py (MDM class, arch='trans_enc'),
# hand-specific (MANO/rotation) terms removed per decision 5. Per-timestep token = linear
# projection of the flattened (N, 3) noised coords, mirroring MDM's InputProcess.poseEmbedding
# almost exactly (192 dims here vs. their 198-dim hand token) -- decision 3. Also keeps their
# 2D-projection Fourier term (see project_angle_fourier below), and the TimestepEmbedder trick
# of reusing the sinusoidal position table as a timestep-indexed lookup.

import math

import torch
import torch.nn as nn

from .heads.track_head import TrackHead
from .heads.visibility_head import VisibilityHead


class PositionalEncoding(nn.Module):
    """sinusoidal table, used both as a per-position embedding added to the transformer input
    and (via TimestepEmbedder) as a lookup table for the diffusion timestep embedding -- mirrors
    MDM's dual use of the same table."""

    def __init__(self, latent_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, D)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim: int, pos_encoder: PositionalEncoding):
        super().__init__()
        self.pos_encoder = pos_encoder
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos_encoder.pe[timesteps, 0]).unsqueeze(0)  # (1, B, D)


def project_angle_fourier(xyz: torch.Tensor, freq: int = 4) -> torch.Tensor:
    """equivalent to forehand4d's compute_kpe_enc (project to pixels with intrinsics, then
    arctan2 the offset from the principal point) but derived directly from the 3D point: for an
    ideal pinhole camera, arctan2(fx*X/Z, fx) == arctan2(X/Z, 1) == arctan2(X, Z) -- the focal
    length cancels, so routing through pixel space is a no-op once we already have the 3D point.
    Robust to points behind/outside the frame by construction; this term helps the model
    reason about out-of-frame points."""
    x_angle = torch.atan2(xyz[..., 0], xyz[..., 2].clamp(min=1e-6))
    y_angle = torch.atan2(xyz[..., 1], xyz[..., 2].clamp(min=1e-6))
    angle = torch.stack([x_angle, y_angle], dim=-1)  # (..., 2)
    freq_bands = 2 ** torch.arange(freq, device=xyz.device, dtype=xyz.dtype)
    angle_expand = angle.unsqueeze(-1) * freq_bands  # (..., 2, freq)
    return torch.stack([angle_expand.sin(), angle_expand.cos()], dim=-1).flatten(-3)  # (..., 2*freq*2)


class TrackDenoiser(nn.Module):
    def __init__(
        self,
        n: int = 64,
        t: int = 128,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_size: int = 1024,
        dropout: float = 0.1,
        kpe_freq: int = 4,
    ):
        super().__init__()
        self.n = n
        self.t = t
        self.latent_dim = latent_dim

        self.input_feats = n * 3  # decision 3: 192-dim flattened coords for n=64
        self.coord_embed = nn.Linear(self.input_feats, latent_dim)
        self.kpe_freq = kpe_freq
        self.kpe_embed = nn.Linear(n * 2 * kpe_freq * 2, latent_dim)  # 2D-projection Fourier term

        self.pos_encoder = PositionalEncoding(latent_dim)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.pos_encoder)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = TrackHead(latent_dim=latent_dim, n=n)
        # visibility is not part of the diffusion state (it isn't noised/denoised), so it's a
        # side-output only used
        # for the auxiliary training loss -- p_sample discards it during sampling.
        self.visibility_head = VisibilityHead(latent_dim=latent_dim, n=n)

    def forward(self, x_noisy: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor, query_xyz_t0: torch.Tensor) -> tuple:
        """x_noisy: (B, T, N, 3) noised, normalized coords. timestep: (B,). cond: (num_cond, B, D)
        conditioning tokens (query tokens etc), dropout already applied by the caller.
        query_xyz_t0: (B, N, 3) normalized -- passed straight through to TrackHead, which adds
        it back as the offset's reference point. Returns (tracks, visibility_logits)."""
        bz, t_dim, n, _ = x_noisy.shape

        coord_tok = self.coord_embed(x_noisy.reshape(bz, t_dim, n * 3))
        kpe = project_angle_fourier(x_noisy, freq=self.kpe_freq)  # (B, T, N, 2*freq*2)
        kpe_tok = self.kpe_embed(kpe.reshape(bz, t_dim, -1))
        per_timestep_tok = (coord_tok + kpe_tok).transpose(0, 1)  # (T, B, D)

        emb = self.embed_timestep(timestep)  # (1, B, D)
        xseq = torch.cat([emb, cond, per_timestep_tok], dim=0)  # (1+num_cond+T, B, D)
        xseq = xseq + self.pos_encoder.pe[: xseq.shape[0]]

        out = self.transformer(xseq)[-t_dim:].transpose(0, 1)  # (B, T, D)
        return self.head(out, query_xyz_t0), self.visibility_head(out)  # (B, T, N, 3), (B, T, N)
