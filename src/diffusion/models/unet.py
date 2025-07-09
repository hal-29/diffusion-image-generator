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
