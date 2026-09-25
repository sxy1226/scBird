from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.CellGraph import CellGraph
from model.GeneGraph import GeneGraph
from module.ClusteringLayer import ClusteringLayer


class HomoGraph(nn.Module):
    """scGCOT backbone with EMA cell/gene teachers and prototype clustering."""

    def __init__(
        self,
        x_shape,
        n_clusters: int,
        hidden_dim: int = 128,
        latent_dim: int = 15,
        relation_dim: int = 32,
        dropout: float = 0.2,
        prototype_temperature: float = 0.5,
    ):
        super().__init__()
        n_genes = int(x_shape[1])

        self.cell_encoder = CellGraph(
            x_shape,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            relation_dim=relation_dim,
            dropout=dropout,
        )
        self.gene_encoder = GeneGraph(
            x_shape,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            relation_dim=relation_dim,
            dropout=dropout,
        )

        self.cell_teacher = copy.deepcopy(self.cell_encoder)
        self.gene_teacher = copy.deepcopy(self.gene_encoder)
        self._freeze_teacher(self.cell_teacher)
        self._freeze_teacher(self.gene_teacher)

        self.pi_decoder = nn.Linear(n_genes, n_genes)
        self.disp_decoder = nn.Linear(n_genes, n_genes)
        self.mean_decoder = nn.Linear(n_genes, n_genes)
        self.cluster_model = ClusteringLayer(
            n_clusters=n_clusters,
            latent_dim=latent_dim,
            temperature=prototype_temperature,
        )

    @staticmethod
    def _freeze_teacher(module: nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.cell_teacher.eval()
        self.gene_teacher.eval()
        return self

    def forward(self, cell_graph, gene_graph) -> Dict[str, torch.Tensor]:
        cell_decoder, z_cell = self.cell_encoder(cell_graph.x, cell_graph.edge_index)
        gene_decoder, z_gene = self.gene_encoder(gene_graph.x, gene_graph.edge_index)

        reconstructed = z_cell @ z_gene.transpose(0, 1)
        pi = torch.sigmoid(self.pi_decoder(reconstructed))
        disp = torch.clamp(F.softplus(self.disp_decoder(reconstructed)), min=1e-4, max=1e4)
        mean = torch.clamp(F.softplus(self.mean_decoder(reconstructed)), min=1e-5, max=1e6)

        return {
            "cell_decoder": cell_decoder,
            "gene_decoder": gene_decoder,
            "reconstructed": reconstructed,
            "z_cell": z_cell,
            "z_gene": z_gene,
            "pi": pi,
            "disp": disp,
            "mean": mean,
        }

    @torch.no_grad()
    def teacher_embeddings(self, cell_graph, gene_graph) -> Tuple[torch.Tensor, torch.Tensor]:
        z_cell = self.cell_teacher.encode(cell_graph.x, cell_graph.edge_index)
        z_gene = self.gene_teacher.encode(gene_graph.x, gene_graph.edge_index)
        return z_cell, z_gene

    @torch.no_grad()
    def update_ema(self, momentum: float) -> None:
        momentum = float(momentum)
        for teacher, student in (
            (self.cell_teacher, self.cell_encoder),
            (self.gene_teacher, self.gene_encoder),
        ):
            for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
                teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1.0 - momentum)
            for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
                teacher_buffer.copy_(student_buffer)

    def cluster_probabilities(self, z: torch.Tensor, detach_prototypes: bool = False) -> torch.Tensor:
        return self.cluster_model(z, detach_prototypes=detach_prototypes)

    @torch.no_grad()
    def initialize_prototypes(self, centers: torch.Tensor) -> None:
        self.cluster_model.initialize(centers)
