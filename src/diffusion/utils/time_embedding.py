import torch
import math
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps (from DDPM paper)"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer("emb", torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, t):
        """t: (batch,) -> (batch, dim)"""
        emb = t[:, None] * self.emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
