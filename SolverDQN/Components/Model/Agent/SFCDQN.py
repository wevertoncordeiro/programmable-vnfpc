import torch
import torch.nn as nn
from torch_geometric.data import Data
from Components.Model.Agent import StateEncoder

HIDDEN_DIM = 256
NUM_ATTENTION_HEADS = 4
DROPOUT_RATE = 0.1


class SFCDQN(nn.Module):
    """Deep Q-Network (DQN) for SFC placement decisions, with improved stability and expressiveness."""

    def __init__(self, state_encoder: StateEncoder,
                 hidden_dim: int = HIDDEN_DIM,
                 dropout: float = DROPOUT_RATE):
        super().__init__()
        self.state_encoder = state_encoder

        # Camada de projeção do estado concatenado (infra + vnf + global)
        self.placement_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Inicialização de pesos (muito importante para estabilidade)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, infra_data, sfc_data, vnf_idx, num_npops, action_mask=None):
        """
        Args:
            infra_data (torch_geometric.data.Data): grafo da infraestrutura
            sfc_data (torch_geometric.data.Data): grafo da requisição SFC
            vnf_idx (int): índice da VNF atual na sequência
            num_npops (int): número de NPoPs disponíveis
            action_mask (torch.BoolTensor, opcional): máscara de ações válidas
        Returns:
            torch.Tensor: vetor de Q-values (tamanho = num_npops)
        """
        # Codifica estados
        h_npops, h_sfc, global_state = self.state_encoder(infra_data, sfc_data, num_npops)

        # Representação da VNF corrente
        vnf_embedding = h_sfc[vnf_idx].unsqueeze(0).expand(num_npops, -1)

        # Expande o estado global para combinar com todos os NPoPs
        global_expanded = global_state.expand(num_npops, -1)

        # Combina todas as representações em um único tensor (vetorizado)
        combined = torch.cat([h_npops, vnf_embedding, global_expanded], dim=-1)

        # Q-values preditos
        q_values = self.placement_net(combined).squeeze(-1)

        # Aplica máscara de ações (se existir)
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, float('-inf'))

        return q_values
