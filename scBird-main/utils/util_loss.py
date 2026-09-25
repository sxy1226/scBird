from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from config import TrainConfig
from module.ZINBLoss import ZINBLoss


def compute_base_losses(
    outputs: Dict[str, torch.Tensor],
    cell_adjacency: torch.Tensor,
    gene_adjacency: torch.Tensor,
    x_graph: torch.Tensor,
    x_count: torch.Tensor,
    size_factors: torch.Tensor,
    config: TrainConfig,
) -> Dict[str, torch.Tensor]:
    cell_loss = F.mse_loss(outputs["cell_decoder"], cell_adjacency)
    gene_loss = F.mse_loss(outputs["gene_decoder"], gene_adjacency)
    expression_loss = F.mse_loss(outputs["reconstructed"], x_graph)
    zinb_loss = ZINBLoss()(
        x=x_count,
        mean=outputs["mean"],
        disp=outputs["disp"],
        pi=outputs["pi"],
        scale_factor=size_factors,
    )

    base = (
        config.cell_graph_weight * cell_loss
        + config.gene_graph_weight * gene_loss
        + config.expression_weight * expression_loss
       + config.zinb_weight * zinb_loss
    )
    return {
        "cell_graph": cell_loss,
        "gene_graph": gene_loss,
        "expression": expression_loss,
        "zinb": zinb_loss,
        "base": base,
    }


def prototype_pseudo_label_loss(
    student_probabilities: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    confidence_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_probabilities = teacher_probabilities.detach()
    confidence = teacher_probabilities.max(dim=1).values
    mask = confidence >= float(confidence_threshold)

    # Numerical guard: an uninitialized split must still produce a valid loss.
    if not torch.any(mask):
        keep = max(1, int(round(0.05 * confidence.numel())))
        top_indices = torch.topk(confidence, k=keep, largest=True).indices
        mask = torch.zeros_like(confidence, dtype=torch.bool)
        mask[top_indices] = True

    selected_student = torch.clamp(student_probabilities[mask], min=1e-8)
    selected_teacher = teacher_probabilities[mask]
    loss = -(selected_teacher * selected_student.log()).sum(dim=1).mean()
    return loss, mask.float().mean(), confidence.mean()
