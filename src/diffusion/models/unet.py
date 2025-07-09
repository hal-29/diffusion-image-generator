import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder
        self.down1 = ResidualBlock(in_channels, 64)
        self.down2 = ResidualBlock(64, 128)
        # Bottleneck
        self.mid = ResidualBlock(128, 128)
        # Decoder (with skip connections)
        self.up1 = ResidualBlock(128 + 128, 64)
        self.up2 = ResidualBlock(64 + 64, 32)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        # Bottleneck
        x_mid = self.mid(x2)
        # Decoder
        x_up1 = self.up1(torch.cat([x_mid, x2], dim=1))
        x_up2 = self.up2(torch.cat([x_up1, x1], dim=1))
        return self.out(x_up2)
