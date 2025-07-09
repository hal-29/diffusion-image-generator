import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion.utils.time_embedding import SinusoidalTimeEmbedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Time embedding projection
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

    def forward(self, x, time_emb):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))

        # Add time embedding (broadcasted)
        time_out = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_out

        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.down1 = ResidualBlock(in_channels, 64, time_emb_dim)
        self.down2 = ResidualBlock(64, 128, time_emb_dim)

        # Bottleneck
        self.mid = ResidualBlock(128, 128, time_emb_dim)

        # Decoder
        self.up1 = ResidualBlock(128 + 128, 64, time_emb_dim)
        self.up2 = ResidualBlock(64 + 64, 32, time_emb_dim)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x1 = self.down1(x, t_emb)
        x2 = self.down2(x1, t_emb)

        # Bottleneck
        x_mid = self.mid(x2, t_emb)

        # Decoder
        x_up1 = self.up1(torch.cat([x_mid, x2], dim=1), t_emb)
        x_up2 = self.up2(torch.cat([x_up1, x1], dim=1), t_emb)

        return self.out(x_up2)
