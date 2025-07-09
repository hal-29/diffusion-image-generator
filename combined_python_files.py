### __init__.py



### loaders.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(batch_size=128, img_size=32):
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


### __init__.py



### attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention layer for global context"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)

        # Attention scores
        scale = C**-0.5
        attn = torch.einsum("bci,bcj->bij", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = torch.einsum("bij,bcj->bci", attn, v)
        out = out.view(B, C, H, W)
        return x + self.proj(out)


### unet.py

import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion.utils.time_embedding import SinusoidalTimeEmbedding
from diffusion.utils.class_embedding import ClassEmbedding
from diffusion.models.attention import SelfAttention


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Time embedding
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        # Class embedding
        self.class_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(class_emb_dim, out_channels)
        )

    def forward(self, x, time_emb, class_emb):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))

        # Add time and class embeddings
        time_out = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        class_out = self.class_mlp(class_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_out + class_out

        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        class_emb_dim=64,
        num_classes=10,
    ):
        super().__init__()
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class embedding
        self.class_embed = ClassEmbedding(num_classes, class_emb_dim)

        self.down1 = ResidualBlock(in_channels, 64, time_emb_dim, class_emb_dim)
        self.attn1 = SelfAttention(64)
        self.down2 = ResidualBlock(64, 128, time_emb_dim, class_emb_dim)

        self.mid_block1 = ResidualBlock(128, 128, time_emb_dim, class_emb_dim)
        self.mid_attn = SelfAttention(128)
        self.mid_block2 = ResidualBlock(128, 128, time_emb_dim, class_emb_dim)

        self.up1 = ResidualBlock(128 + 128, 64, time_emb_dim, class_emb_dim)
        self.attn2 = SelfAttention(64)
        self.up2 = ResidualBlock(64 + 64, 32, time_emb_dim, class_emb_dim)

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x, t, labels):
        # Time and class embedding (unchanged)
        t_emb = self.time_mlp(self.time_embed(t))
        class_emb = self.class_embed(labels)

        # Encoder
        x1 = self.down1(x, t_emb, class_emb)
        x1 = self.attn1(x1)
        x2 = self.down2(x1, t_emb, class_emb)

        # Bottleneck
        x_mid = self.mid_block1(x2, t_emb, class_emb)
        x_mid = self.mid_attn(x_mid)
        x_mid = self.mid_block2(x_mid, t_emb, class_emb)

        # Decoder
        x_up1 = self.up1(torch.cat([x_mid, x2], dim=1), t_emb, class_emb)
        x_up1 = self.attn2(x_up1)
        x_up2 = self.up2(torch.cat([x_up1, x1], dim=1), t_emb, class_emb)

        return self.out(x_up2)


### __init__.py



### forward.py

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


### __init__.py



### schedulers.py

import torch
import math


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise scheduling from DDPM paper"""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from Improved DDPM"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


### time_embedding.py

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


### class_embedding.py

import torch.nn as nn


class ClassEmbedding(nn.Module):
    """Embed class labels into a latent space"""

    def __init__(self, num_classes, emb_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_classes, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, labels):
        """labels: (batch,) -> (batch, emb_dim)"""
        return self.embedding(labels)


### config.py

import torch


class Config:
    # Diffusion
    TIMESTEPS = 1000
    IMG_SIZE = 32
    BETA_START = 0.0001
    BETA_END = 0.02

    # Training
    BATCH_SIZE = 128
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()


### __init__.py



### sampling.py

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


### train.py

import torch
import torch.nn.functional as F
from diffusion.models.unet import UNet
from diffusion.data.loaders import get_mnist_loaders
from diffusion.diffusion.forward import ForwardDiffusion
from tqdm import tqdm
import os


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(time_emb_dim=128, class_emb_dim=64, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    diffusion = ForwardDiffusion(timesteps=1000)
    dataloader = get_mnist_loaders(batch_size=128)

    for epoch in range(1):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for x, labels in progress_bar:
            x = x.to(device)
            labels = labels.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            noisy_x = diffusion.q_sample(x, t, noise)

            pred_noise = model(noisy_x, t, labels)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "checkpoints/unet_class_conditional.pth")


if __name__ == "__main__":
    train()


### combined_python_files.py



