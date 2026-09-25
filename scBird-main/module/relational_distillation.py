from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def normalize_cell_gene_bridge(x_bridge: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric degree normalization for the non-negative cell-gene matrix."""
    x_bridge = torch.clamp(x_bridge, min=0.0)
    row_degree = torch.clamp(x_bridge.sum(dim=1, keepdim=True), min=eps)
    col_degree = torch.clamp(x_bridge.sum(dim=0, keepdim=True), min=eps)
    return x_bridge / torch.sqrt(row_degree * col_degree)


def _unique_nonself_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    source, target = edge_index[0], edge_index[1]
    mask = source < target
    filtered = edge_index[:, mask]
    if filtered.shape[1] == 0:
        mask = source != target
        filtered = edge_index[:, mask]
    return filtered


def edge_relation_scores(features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    edge_index = _unique_nonself_edge_index(edge_index)
    if edge_index.shape[1] == 0:
        return features.new_zeros((0,))
    normalized = F.normalize(features, p=2, dim=1)
    source, target = edge_index[0], edge_index[1]
    return (normalized[source] * normalized[target]).sum(dim=1)


def bidirectional_relational_distillation(
    z_cell_student: torch.Tensor,
    z_gene_student: torch.Tensor,
    z_cell_teacher: torch.Tensor,
    z_gene_teacher: torch.Tensor,
    bridge: torch.Tensor,
    cell_edge_index: torch.Tensor,
    gene_edge_index: torch.Tensor,
    reverse_weight: float,
) -> Dict[str, torch.Tensor]:
    """Transfer graph relations through the cell-gene incidence matrix.

    Cell teacher knowledge is projected to gene descriptors by X^T Z_c^T;
    gene teacher knowledge is projected to cell descriptors by X Z_g^T.
    The student is supervised on edge-wise cosine relations, avoiding dense
    N-by-N cross-view transport matrices.
    """
    with torch.no_grad():
        induced_gene = bridge.transpose(0, 1) @ F.normalize(z_cell_teacher, p=2, dim=1)
        induced_cell = bridge @ F.normalize(z_gene_teacher, p=2, dim=1)
        gene_target = edge_relation_scores(induced_gene, gene_edge_index)
        cell_target = edge_relation_scores(induced_cell, cell_edge_index)

    gene_student = edge_relation_scores(z_gene_student, gene_edge_index)
    cell_student = edge_relation_scores(z_cell_student, cell_edge_index)

    zero = z_cell_student.new_zeros(())
    c_to_g = F.mse_loss(gene_student, gene_target) if gene_student.numel() else zero
    g_to_c = F.mse_loss(cell_student, cell_target) if cell_student.numel() else zero
    total = c_to_g + float(reverse_weight) * g_to_c
    return {"relation": total, "c_to_g": c_to_g, "g_to_c": g_to_c}
