"""vision transformer module"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import MultiHeadAttention, PatchEmbedding


class MLP(nn.Module):
    """MLP module"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(in_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, 4 * embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int, num_layers: int):
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
