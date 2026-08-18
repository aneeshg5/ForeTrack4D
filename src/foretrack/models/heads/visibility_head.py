import torch
import torch.nn as nn


class VisibilityHead(nn.Module):

    def __init__(self, latent_dim: int = 512, n: int = 64):
        super().__init__()
        self.n = n
        self.proj = nn.Linear(latent_dim, n)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)
