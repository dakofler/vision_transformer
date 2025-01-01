"""vision transformer module"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Patch embedding module"""

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """MLP module"""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.up_proj = nn.Linear(in_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, in_dim)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        x = F.gelu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = self.down_proj(x)
        x = F.dropout(x, self.dropout, self.training)
        return x


class MSA(nn.Module):
    """Multi-head self-attention module"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape

        qkv = self.in_proj(x)
        q, k, v = qkv.split(D, dim=2)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_w = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = F.dropout(attn_w, self.dropout, self.training)
        y = attn_w @ v

        y = y.transpose(1, 2).contiguous().flatten(2)
        y = self.out_proj(y)
        y = F.dropout(y, self.dropout, self.training)
        return y


class TransformerBlock(nn.Module):
    """Transformer block module"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.msa = MSA(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, 4 * embed_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer module"""

    def __init__(
        self,
        in_channels: int,
        num_patches: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.empty(1, num_patches + 1, embed_dim).normal_(std=0.02)
        )
        self.class_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        patch_embed = self.patch_embed(x)
        class_embed = self.class_embed.expand(x.shape[0], 1, self.embed_dim)
        x = torch.concat([class_embed, patch_embed], dim=1) + self.pos_embed
        x = F.dropout(x, self.dropout)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])  # classification token
        x = self.head(x)
        return x
