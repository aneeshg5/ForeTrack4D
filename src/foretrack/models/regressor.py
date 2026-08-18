# Adapted from forehand4d's src/models/mdm_ff/model.py. See NOTICE.md.

import torch
import torch.nn as nn

from .denoiser import PositionalEncoding
from .heads.track_head import TrackHead
from .heads.visibility_head import VisibilityHead


class TrackRegressor(nn.Module):
    def __init__(
        self,
        n: int = 64,
        t: int = 128,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_size: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.t = t
        self.latent_dim = latent_dim

        self.pos_encoder = PositionalEncoding(latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = TrackHead(latent_dim=latent_dim, n=n)
        self.visibility_head = VisibilityHead(latent_dim=latent_dim, n=n)  # see denoiser.py's comment

    def forward(self, cond: torch.Tensor, query_xyz_t0: torch.Tensor) -> tuple:
        bz = cond.shape[1]
        query_pos = torch.zeros(self.t, bz, self.latent_dim, device=cond.device, dtype=cond.dtype)
        query_pos = query_pos + self.pos_encoder.pe[: self.t]

        xseq = torch.cat([cond, query_pos], dim=0)  # (num_cond+T, B, D)
        cond_dim = cond.shape[0]
        out = self.transformer(xseq)[cond_dim:].transpose(0, 1)  # (B, T, D)
        return self.head(out, query_xyz_t0), self.visibility_head(out)  # (B, T, N, 3), (B, T, N)
