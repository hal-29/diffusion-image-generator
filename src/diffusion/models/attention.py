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
