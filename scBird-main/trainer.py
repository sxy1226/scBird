from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from config import TrainConfig
from model.HomoGraph import HomoGraph
from module.dataloader import prepare_cell_gene_graphs
from module.relational_distillation import (
    bidirectional_relational_distillation,
    normalize_cell_gene_bridge,
)
from utils.util_loss import compute_base_losses, prototype_pseudo_label_loss
from utils.util_print import cal_cluster_metrics, format_metrics


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_float_dict(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in losses.items():
        output[key] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    return output


def _optimizer(model: HomoGraph, lr: float, config: TrainConfig):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.Adam(parameters, lr=lr, weight_decay=config.weight_decay)


def _optimization_step(
    model: HomoGraph,
    optimizer: torch.optim.Optimizer,
    total_loss: torch.Tensor,
    config: TrainConfig,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=config.grad_clip,
        )
    optimizer.step()
    model.update_ema(config.ema_momentum)


@torch.no_grad()
def _teacher_predictions(model: HomoGraph, cell_graph, gene_graph):
    model.eval()
    z_cell_teacher, z_gene_teacher = model.teacher_embeddings(cell_graph, gene_graph)
    probabilities = model.cluster_probabilities(z_cell_teacher, detach_prototypes=True)
    predictions = probabilities.argmax(dim=1)
    return z_cell_teacher, z_gene_teacher, probabilities, predictions


def _initialize_prototypes(
    model: HomoGraph,
    cell_graph,
    gene_graph,
    n_clusters: int,
    config: TrainConfig,
) -> None:
    with torch.no_grad():
        z_cell_teacher, _ = model.teacher_embeddings(cell_graph, gene_graph)
    embedding = z_cell_teacher.detach().cpu().numpy()
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=config.kmeans_n_init,
        random_state=config.seed,
    )
    kmeans.fit(embedding)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=z_cell_teacher.dtype, device=z_cell_teacher.device)
    model.initialize_prototypes(centers)


def train_model(
    x_graph: np.ndarray,
    x_bridge: np.ndarray,
    x_count: np.ndarray,
    size_factors: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    output_dir: str,
    cell_adjacency: Optional[np.ndarray] = None,
):
    seed_all(config.seed)
    os.makedirs(output_dir, exist_ok=True)

    n_clusters = int(np.unique(labels).size)
    cell_graph, cell_target, gene_graph, gene_target = prepare_cell_gene_graphs(
        x_graph=x_graph,
        k=config.knn_k,
        pca_dim=config.pca_dim,
        cell_adjacency=cell_adjacency,
    )
    cell_graph = cell_graph.to(device)
    gene_graph = gene_graph.to(device)

    cell_target_t = torch.tensor(cell_target, dtype=torch.float32, device=device)
    gene_target_t = torch.tensor(gene_target, dtype=torch.float32, device=device)
    x_graph_t = torch.tensor(x_graph, dtype=torch.float32, device=device)
    x_count_t = torch.tensor(x_count, dtype=torch.float32, device=device)
    size_factors_t = torch.tensor(size_factors, dtype=torch.float32, device=device)
    bridge_t = normalize_cell_gene_bridge(
        torch.tensor(x_bridge, dtype=torch.float32, device=device)
    )

    model = HomoGraph(
        x_shape=x_graph.shape,
        n_clusters=n_clusters,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        relation_dim=config.relation_dim,
        dropout=config.dropout,
        prototype_temperature=config.prototype_temperature,
    ).to(device)

    history = []

    # Stage 1: learn the two graph views and count distribution without distillation.
    optimizer = _optimizer(model, config.pretrain_lr, config)
    for epoch in tqdm(range(1, config.pretrain_epochs + 1), desc="Stage 1/3 base pretraining"):
        model.train()
        outputs = model(cell_graph, gene_graph)
        losses = compute_base_losses(
            outputs,
            cell_target_t,
            gene_target_t,
            x_graph_t,
            x_count_t,
            size_factors_t,
            config,
        )
        total = losses["base"]
        _optimization_step(model, optimizer, total, config)

        record = {"stage": "pretrain", "epoch": epoch, **_to_float_dict(losses), "total": float(total.detach().cpu())}
        history.append(record)

    # Stage 2: curriculum bidirectional relational distillation.
    for epoch in tqdm(range(1, config.distill_epochs + 1), desc="Stage 2/3 relational distillation"):
        model.train()
        outputs = model(cell_graph, gene_graph)
        with torch.no_grad():
            z_cell_teacher, z_gene_teacher = model.teacher_embeddings(cell_graph, gene_graph)

        progress = epoch / max(config.distill_epochs, 1)
        reverse_weight = config.reverse_relation_max * progress
        relation_weight = config.relation_weight * progress
        relation_losses = bidirectional_relational_distillation(
            z_cell_student=outputs["z_cell"],
            z_gene_student=outputs["z_gene"],
            z_cell_teacher=z_cell_teacher,
            z_gene_teacher=z_gene_teacher,
            bridge=bridge_t,
            cell_edge_index=cell_graph.edge_index,
            gene_edge_index=gene_graph.edge_index,
            reverse_weight=reverse_weight,
        )
        base_losses = compute_base_losses(
            outputs,
            cell_target_t,
            gene_target_t,
            x_graph_t,
            x_count_t,
            size_factors_t,
            config,
        )
        total = base_losses["base"] + relation_weight * relation_losses["relation"]
        _optimization_step(model, optimizer, total, config)

        combined = {**base_losses, **relation_losses}
        record = {
            "stage": "distill",
            "epoch": epoch,
            **_to_float_dict(combined),
            "reverse_weight": reverse_weight,
            "relation_schedule": relation_weight,
            "total": float(total.detach().cpu()),
        }
        history.append(record)

    # Stage 3: initialize K prototypes from the EMA cell teacher and optimize soft pseudo labels.
    _initialize_prototypes(model, cell_graph, gene_graph, n_clusters, config)
    optimizer = _optimizer(model, config.cluster_lr, config)

    for epoch in tqdm(range(1, config.cluster_epochs + 1), desc="Stage 3/3 prototype clustering"):
        model.train()
        outputs = model(cell_graph, gene_graph)
        with torch.no_grad():
            z_cell_teacher, z_gene_teacher = model.teacher_embeddings(cell_graph, gene_graph)
            teacher_probabilities = model.cluster_probabilities(
                z_cell_teacher,
                detach_prototypes=True,
            )

        relation_losses = bidirectional_relational_distillation(
            z_cell_student=outputs["z_cell"],
            z_gene_student=outputs["z_gene"],
            z_cell_teacher=z_cell_teacher,
            z_gene_teacher=z_gene_teacher,
            bridge=bridge_t,
            cell_edge_index=cell_graph.edge_index,
            gene_edge_index=gene_graph.edge_index,
            reverse_weight=config.reverse_relation_max,
        )
        student_probabilities = model.cluster_probabilities(outputs["z_cell"])
        prototype_loss, selected_ratio, mean_confidence = prototype_pseudo_label_loss(
            student_probabilities,
            teacher_probabilities,
            confidence_threshold=config.pseudo_confidence,
        )
        base_losses = compute_base_losses(
            outputs,
            cell_target_t,
            gene_target_t,
            x_graph_t,
            x_count_t,
            size_factors_t,
            config,
        )
        total = (
            base_losses["base"]
            + config.relation_weight * relation_losses["relation"]
           # + config.prototype_weight * prototype_loss
        )
        _optimization_step(model, optimizer, total, config)

        combined = {
            **base_losses,
            **relation_losses,
            "prototype": prototype_loss,
            "selected_ratio": selected_ratio,
            "mean_confidence": mean_confidence,
        }
        record = {
            "stage": "cluster",
            "epoch": epoch,
            **_to_float_dict(combined),
            "total": float(total.detach().cpu()),
        }
        history.append(record)

        if epoch == 1 or epoch % config.eval_interval == 0 or epoch == config.cluster_epochs:
            _, _, _, predictions = _teacher_predictions(model, cell_graph, gene_graph)
            print(format_metrics("cluster", epoch, labels, predictions.cpu().numpy(), record))

    z_cell, z_gene, probabilities, predictions = _teacher_predictions(model, cell_graph, gene_graph)
    final_predictions = predictions.cpu().numpy()
    acc, nmi, ari = cal_cluster_metrics(labels, final_predictions)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "n_clusters": n_clusters,
            "x_shape": tuple(x_graph.shape),
        },
        os.path.join(output_dir, "model_final.pt"),
    )
    np.savetxt(os.path.join(output_dir, "cell_embedding.txt"), z_cell.cpu().numpy(), fmt="%.8f")
    np.savetxt(os.path.join(output_dir, "gene_embedding.txt"), z_gene.cpu().numpy(), fmt="%.8f")
    pd.DataFrame({"pred": final_predictions}).to_csv(
        os.path.join(output_dir, "pred_labels.csv"), index=False
    )
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"ACC": acc, "NMI": nmi, "ARI": ari}, handle, ensure_ascii=False, indent=2)

    return model, {"ACC": acc, "NMI": nmi, "ARI": ari}
