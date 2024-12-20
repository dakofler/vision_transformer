"""vision transformer module"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Patch embedding module"""

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = x.flatten(2).transpose(1, 2)
        return x


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

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class MLP(nn.Module):
    """MLP module"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(in_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        x = F.relu(x)
        x = self.down_proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block module"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, 4 * embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
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
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_embed = nn.Embedding(num_patches + 1, embed_dim)
        self.class_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        patch_embed = self.patch_embed(x)
        class_embed = self.class_embed.expand(x.shape[0], 1, self.embed_dim)
        x = torch.concat([class_embed, patch_embed], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x
