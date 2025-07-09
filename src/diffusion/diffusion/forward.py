import torch
from diffusion.utils.schedulers import linear_beta_schedule


class ForwardDiffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        """Corrupt x_start with noise at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(
            -1, 1, 1, 1
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, arr, t, x_shape):
        """Extract values from array for timestep t"""
        return arr[t].view(-1, 1, 1, 1)
