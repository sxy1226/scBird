from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class GeneGraph(nn.Module):
    def __init__(
        self,
        x_shape,
        hidden_dim: int = 128,
        latent_dim: int = 15,
        relation_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        input_dim = int(x_shape[0])
        self.dropout = float(dropout)
        self.conv1 = TransformerConv(input_dim, hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, latent_dim)
        self.relation_projection = nn.Linear(latent_dim, relation_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode_graph(self, z: torch.Tensor) -> torch.Tensor:
        h = self.relation_projection(z)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = h @ h.transpose(0, 1) / math.sqrt(max(h.shape[1], 1))
        return torch.sigmoid(logits)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        z = self.encode(x, edge_index)
        return self.decode_graph(z), z
