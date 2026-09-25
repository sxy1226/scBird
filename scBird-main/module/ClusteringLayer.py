from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusteringLayer(nn.Module):
    """Cosine prototype clustering with gradient-updated prototypes."""

    def __init__(self, n_clusters: int, latent_dim: int, temperature: float = 0.5):
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.latent_dim = int(latent_dim)
        self.temperature = float(temperature)
        self.clusters = nn.Parameter(torch.empty(self.n_clusters, self.latent_dim))
        nn.init.xavier_uniform_(self.clusters)

    @torch.no_grad()
    def initialize(self, centers: torch.Tensor) -> None:
        if centers.shape != self.clusters.shape:
            raise ValueError(
                f"Prototype shape mismatch: expected {tuple(self.clusters.shape)}, got {tuple(centers.shape)}."
            )
        self.clusters.copy_(centers.to(self.clusters.device, dtype=self.clusters.dtype))

    def forward(self, inputs: torch.Tensor, detach_prototypes: bool = False) -> torch.Tensor:
        centers = self.clusters.detach() if detach_prototypes else self.clusters
        inputs = F.normalize(inputs, p=2, dim=1)
        centers = F.normalize(centers, p=2, dim=1)
        logits = inputs @ centers.transpose(0, 1)
        logits = logits / max(self.temperature, 1e-6)
        return torch.softmax(logits, dim=1)
