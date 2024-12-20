"""vision transformer components"""

import torch.nn.functional as F
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn = None

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)

        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        self.attn = attn.detach()
        y = attn @ v

        y = y.transpose(1, 2).view(B, T, C)
        return self.out(y)


class PatchEmbedding(nn.Module):
    """Patch embedding module"""

    pass
