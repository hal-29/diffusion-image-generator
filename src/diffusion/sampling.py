import torch
from tqdm import tqdm


class DDIMSampler:
    def __init__(self, model, diffusion, eta=0.0):
        self.model = model
        self.diffusion = diffusion
        self.eta = eta

    @torch.no_grad()
    def sample(self, shape, labels, num_steps=50):
        """Accelerated sampling with DDIM"""
        x = torch.randn(shape, device=self.model.device)
        timesteps = torch.linspace(
            self.diffusion.timesteps - 1, 0, num_steps, dtype=torch.long
        )

        for t in tqdm(timesteps):
            # Predict noise
            t_tensor = torch.full((shape[0],), t, device=x.device)
            pred_noise = self.model(x, t_tensor, labels)

            # DDIM update rule
            alpha = self.diffusion.alphas_cumprod[t]
            alpha_prev = self.diffusion.alphas_cumprod[t - 1] if t > 0 else 1.0
            sigma = self.eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
            )

            x0_pred = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
            x = (
                torch.sqrt(alpha_prev) * x0_pred
                + torch.sqrt(1 - alpha_prev - sigma**2) * pred_noise
                + sigma * torch.randn_like(x)
            )

        return x.clamp(-1, 1)
