from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def cluster_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must contain the same number of samples.")
    size = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    confusion = np.zeros((size, size), dtype=np.int64)
    for predicted, actual in zip(y_pred, y_true):
        confusion[predicted, actual] += 1
    row, col = linear_sum_assignment(confusion.max() - confusion)
    return float(confusion[row, col].sum() / max(y_true.size, 1))


def cal_cluster_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = cluster_accuracy(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    return acc, nmi, ari


def format_metrics(stage: str, epoch: int, y_true, y_pred, losses=None) -> str:
    acc, nmi, ari = cal_cluster_metrics(y_true, y_pred)
    parts = [f"[{stage}] epoch={epoch:04d}"]
    if losses:
        for key, value in losses.items():
            if key in {"total", "base", "relation", "prototype"}:
                parts.append(f"{key}={float(value):.4f}")
    parts.extend([f"ACC={acc:.4f}", f"NMI={nmi:.4f}", f"ARI={ari:.4f}"])
    return " ".join(parts)
