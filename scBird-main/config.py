from dataclasses import asdict, dataclass
from typing import Dict, Any


@dataclass
class TrainConfig:
    """Training configuration for -BRD.

    The defaults keep the original  scale (400 total epochs), while
    replacing the OT fine-tuning stage with relational distillation and
    prototype clustering.
    """

    seed: int = 6487
    hvg: int = 500
    knn_k: int = 15
    pca_dim: int = 50

    hidden_dim: int = 128
    latent_dim: int = 15
    relation_dim: int = 32
    dropout: float = 0.2

    pretrain_epochs: int = 200
    distill_epochs: int = 200
    cluster_epochs: int = 100
    pretrain_lr: float = 5e-4
    cluster_lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 5.0

    cell_graph_weight: float = 1.
    gene_graph_weight: float = 0.2
    expression_weight: float = 0.3
    

    # cell_graph_weight: float = 1.1 #Klein
    # gene_graph_weight: float = 0.6
    # expression_weight: float = 0.5


    zinb_weight: float = 1.0
    relation_weight: float = 1.0
    prototype_weight: float = 1.0

    ema_momentum: float = 0.99
    reverse_relation_max: float = 0.3
    relation_temperature: float = 1.0

    prototype_temperature: float = 0.2
    pseudo_confidence: float = 0.8
    kmeans_n_init: int = 20

    eval_interval: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
