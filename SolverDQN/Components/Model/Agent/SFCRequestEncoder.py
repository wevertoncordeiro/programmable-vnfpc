import numpy
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

HIDDEN_DIM = 256
NUM_GNN_LAYERS = 1
NUM_TRANSFORMER_LAYERS = 1
NUM_ATTENTION_HEADS = 4
DROPOUT_RATE = 0.1

class SFCRequestEncoder(nn.Module):
    """Deeper Transformer-based encoder for SFC requests"""

    def __init__(self, node_feat_dim: int,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(node_feat_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        seq_len = x.size(0)
        h = self.input_projection(x).unsqueeze(0)
        h = h + self.pos_embedding[:, :seq_len, :]

        for block in self.blocks:
            h = block(h)

        global_rep, _ = self.global_attention(self.global_query, h, h)
        h = self.norm_out(h)
        return h.squeeze(0), global_rep.squeeze(0).squeeze(0)


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm and gated residual"""
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.res_gate1 = nn.Parameter(torch.ones(1))
        self.res_gate2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.res_gate1 * attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.res_gate2 * ffn_out
        return x
