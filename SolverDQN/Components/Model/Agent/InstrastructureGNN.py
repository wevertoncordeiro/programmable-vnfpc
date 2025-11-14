import logging

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch.nn import LayerNorm

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sfc_dqn_execution.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class InfrastructureGNN(nn.Module):
    """Enhanced GNN for encoding physical infrastructure (powerful Transformer-style architecture)"""

    def __init__(self, node_feat_dim: int, hidden_dim: int = 128,
                 num_layers: int = 4, dropout: float = 0.1, drop_path_rate: float = 0.05):
        super().__init__()

        self.input_layer = nn.Linear(node_feat_dim, hidden_dim)

        # Layer-wise stochastic depth (DropPath)
        self.drop_path_rates = torch.linspace(0, drop_path_rate, 2).tolist()

        self.layers = nn.ModuleList([
            GNNBlock(hidden_dim, dropout=dropout, drop_path=self.drop_path_rates[i])
            for i in range(2)
        ])

        self.norm_out = LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        h = self.input_layer(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        h = self.norm_out(h)
        return h


class GNNBlock(nn.Module):
    """Transformer-like GNN block with pre-norm, attention, FFN, and gated residuals"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(hidden_dim)
        self.attn = TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False,
                                    dropout=dropout, beta=True)
        self.norm2 = LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Gated residual connection (like in Gated Transformer)
        self.res_gate1 = nn.Parameter(torch.ones(1))
        self.res_gate2 = nn.Parameter(torch.ones(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, edge_index):
        # Pre-norm + attention
        h = self.norm1(x)
        h_attn = self.attn(h, edge_index)
        x = x + self.drop_path(self.res_gate1 * h_attn)

        # Pre-norm + FFN
        h = self.norm2(x)
        h_ffn = self.ffn(h)
        x = x + self.drop_path(self.res_gate2 * h_ffn)
        return x


class DropPath(nn.Module):
    """Stochastic Depth / DropPath"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
