import torch
import torch.nn as nn


class TrackHead(nn.Module):

    def __init__(self, latent_dim: int = 512, n: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.n = n
        self.proj = nn.Linear(latent_dim, n * 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, query_xyz_t0: torch.Tensor) -> torch.Tensor:
        bz, t_dim = tokens.shape[:2]
        offset = self.proj(tokens).reshape(bz, t_dim, self.n, 3)
        return offset + query_xyz_t0.unsqueeze(1)
