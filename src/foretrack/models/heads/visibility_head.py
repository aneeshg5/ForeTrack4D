import torch
import torch.nn as nn


class VisibilityHead(nn.Module):
    """per-point-per-timestep binary visibility logit, from the same transformer tokens
    TrackHead consumes -- kept as its own module (decision 5) alongside TrackHead, so it can be
    dropped or swapped without touching the denoiser/regressor backbone."""

    def __init__(self, latent_dim: int = 512, n: int = 64):
        super().__init__()
        self.n = n
        self.proj = nn.Linear(latent_dim, n)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D) -> (B, T, N) logits (unnormalized -- caller applies BCEWithLogits)
        return self.proj(tokens)
