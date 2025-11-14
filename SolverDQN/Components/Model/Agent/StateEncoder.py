import numpy
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

from Components.Model.Agent.InstrastructureGNN import InfrastructureGNN
from Components.Model.Agent.SFCRequestEncoder import SFCRequestEncoder

HIDDEN_DIM = 256
NUM_GNN_LAYERS = 1
NUM_TRANSFORMER_LAYERS = 1
NUM_ATTENTION_HEADS = 4
DROPOUT_RATE = 0.1


class StateEncoder(nn.Module):
    def __init__(self, infra_feat_dim: int, sfc_feat_dim: int,
                 hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT_RATE):
        super().__init__()

        self.infra_gnn = InfrastructureGNN(infra_feat_dim, hidden_dim)
        self.sfc_encoder = SFCRequestEncoder(sfc_feat_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Projeção global
        self.global_mlp = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, infra_data, sfc_data, num_npops):
        # Codificação da infraestrutura e do SFC
        h_infra = self.infra_gnn(infra_data.x, infra_data.edge_index)
        h_sfc, sfc_global = self.sfc_encoder(sfc_data.x)

        # Fusão direta entre SFC e infraestrutura
        infra_context = h_infra.mean(dim=0, keepdim=True).expand_as(h_sfc)
        h_sfc_attended = self.fusion(torch.cat([h_sfc, infra_context], dim=-1))

        # Estado global
        global_infra = h_infra.mean(dim=0, keepdim=True)
        global_state = self.global_mlp(
            torch.cat([global_infra, sfc_global.unsqueeze(0)], dim=-1)
        )

        # Seleção dos NPoPs
        h_npops = h_infra[:num_npops]

        return h_npops, h_sfc_attended, global_state
