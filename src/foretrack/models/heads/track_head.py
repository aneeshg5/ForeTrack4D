import torch
import torch.nn as nn


class TrackHead(nn.Module):
    """final per-timestep projection to an (n, 3) OFFSET from query_xyz_t0, added back before
    returning -- not an absolute position. Diagnosed after 4 training attempts: predicting
    absolute coordinates gave the network no inductive bias toward "stay near the known t0
    position" (which query_xyz_t0 already tells it), so it had to rediscover that purely from
    data starting from a random init, and it never did -- a post-hoc shrinkage sweep on a
    trained checkpoint showed the model's raw predictions overshoot real displacement by ~2.4x
    with no alpha giving meaningfully-better-than-static ADE. Zero-initializing proj means
    training starts at exactly
    query_xyz_t0 for every frame -- "no motion" is free, the model only has to learn to add
    displacement on top, instead of learning the whole absolute position from scratch. Kept as
    a separate module (decision 5) so a joint hand+object variant can swap in a different head
    later without touching the denoiser."""

    def __init__(self, latent_dim: int = 512, n: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.n = n
        self.proj = nn.Linear(latent_dim, n * 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, query_xyz_t0: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, D). query_xyz_t0: (B, N, 3), same normalized space as the offset ->
        # broadcast-added over T. (B, T, N, 3)
        bz, t_dim = tokens.shape[:2]
        offset = self.proj(tokens).reshape(bz, t_dim, self.n, 3)
        return offset + query_xyz_t0.unsqueeze(1)
