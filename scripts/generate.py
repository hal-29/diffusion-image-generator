import torch
from diffusion.models.unet import UNet
from diffusion.diffusion.forward import ForwardDiffusion
from diffusion.utils.schedulers import cosine_beta_schedule
import matplotlib.pyplot as plt
from diffusion.sampling import DDIMSampler


@torch.no_grad()
def p_sample(model, x, t, t_index, labels):
    betas_t = diffusion.betas[t_index].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(
        1.0 - diffusion.alphas_cumprod[t_index]
    ).view(-1, 1, 1, 1)
    sqrt_recip_alphas_t = torch.sqrt(1.0 / diffusion.alphas[t_index]).view(-1, 1, 1, 1)

    pred_noise = model(x, t, labels)

    mean = sqrt_recip_alphas_t * (
        x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
    )
    if t_index == 0:
        return mean
    else:
        posterior_variance_t = diffusion.posterior_variance[t_index].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return mean + torch.sqrt(posterior_variance_t) * noise


def generate_digit(digit, num_samples=1, use_ddim=True, steps=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(...).to(device)
    model.load_state_dict(torch.load("checkpoints/unet_class_conditional.pth"))
    model.eval()

    labels = torch.full((num_samples,), digit, device=device)

    if use_ddim:
        sampler = DDIMSampler(model, diffusion, eta=0.0)
        x = sampler.sample(
            shape=(num_samples, 1, 32, 32), labels=labels, num_steps=steps
        )
    else:
        x = torch.randn((num_samples, 1, 32, 32), device=device)
        for t in reversed(range(diffusion.timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device)
            x = p_sample(model, x, t_tensor, t, labels)

    plt.imshow(x[0, 0].cpu().numpy(), cmap="gray")
    plt.title(
        f"Generated Digit: {digit} (DDIM, {steps} steps)"
        if use_ddim
        else f"Digit: {digit} (DDPM)"
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    diffusion = ForwardDiffusion(timesteps=1000)
    generate_digit(digit=3, use_ddim=True, steps=20)
