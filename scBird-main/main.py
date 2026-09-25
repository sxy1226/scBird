from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from config import TrainConfig
from module.dataloader import load_expression_data
from trainer import train_model

#Klein    Muraro     Quake_Smart-seq2_Heart   Adam   Pollen
#Quake_10x_Limb_Muscle    mouse_ES_cell   Plasschaert   Young     mouse_ES_cell

def parse_args():
    parser = argparse.ArgumentParser(
        description="BRD: bidirectional cell-gene relational distillation for scRNA-seq clustering"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--dataset", default="mouse_ES_cell")   #Final: ACC=0.98270, NMI=0.93169, ARI=0.96940, time=23.56s
    parser.add_argument("--hvg", type=int, default=2000)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=4396)
    parser.add_argument("--pretrain-epochs", type=int, default=400)
    parser.add_argument("--distill-epochs", type=int, default=600)
    parser.add_argument("--cluster-epochs", type=int, default=600) 
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--relation-weight", type=float, default=1)
    parser.add_argument("--zinb-weight", type=float, default=0.6)
    parser.add_argument("--reverse-relation-max", type=float, default=0.3)
    parser.add_argument("--prototype-weight", type=float, default=0)
    parser.add_argument("--prototype-temperature", type=float, default=0.2)
    parser.add_argument("--pseudo-confidence", type=float, default=0.8)
    parser.add_argument(
        "--h5-key-format",
        action="store_true",
        help="Read data.h5 using pandas key='data'.",
    )
    parser.add_argument(
        "--use-ne",
        action="store_true",
        help="For datasets with data.tsv.",
    )
    return parser.parse_args()


def maybe_load_ne_adjacency(args, labels) -> np.ndarray | None:
    if not args.use_ne:
        return None
    tsv_path = os.path.join(args.data_root, args.dataset, "data.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"--use-ne was set, but {tsv_path} does not exist.")
    from utils.utils_gac import load_data

    adjacency, _, _, _ = load_data(
        tsv_path,
        args.dataset,
        512,
        True,
        int(np.unique(labels).size),
        args.knn_k,
        seed=args.seed,
    )
    return adjacency


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")

    config = TrainConfig(
        seed=args.seed,
        hvg=args.hvg,
        knn_k=args.knn_k,
        pretrain_epochs=args.pretrain_epochs,
        distill_epochs=args.distill_epochs,
        cluster_epochs=args.cluster_epochs,
        ema_momentum=args.ema_momentum,
        relation_weight=args.relation_weight,
        zinb_weight=args.zinb_weight,
        reverse_relation_max=args.reverse_relation_max,
        prototype_weight=args.prototype_weight,
        prototype_temperature=args.prototype_temperature,
        pseudo_confidence=args.pseudo_confidence,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    start = time.time()

    data = load_expression_data(
        dataname=args.dataset,
        highly_genes=config.hvg,
        data_root=args.data_root,
        h5_key_format=args.h5_key_format,
    )
    cell_adjacency = maybe_load_ne_adjacency(args, data.labels)

    output_dir = os.path.join(args.output_root, args.dataset)
    _, metrics = train_model(
        x_graph=data.x_graph,
        x_bridge=data.x_bridge,
        x_count=data.x_count,
        size_factors=data.size_factors,
        labels=data.labels,
        config=config,
        device=device,
        output_dir=output_dir,
        cell_adjacency=cell_adjacency,
    )

    elapsed = time.time() - start
    print(
        f"Final: ACC={metrics['ACC']:.5f}, NMI={metrics['NMI']:.5f}, "
        f"ARI={metrics['ARI']:.5f}, time={elapsed:.2f}s"
    )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
