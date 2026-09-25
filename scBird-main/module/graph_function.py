from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def degree_power(adjacency: np.ndarray, power: float) -> np.ndarray:
    degrees = np.power(np.asarray(adjacency.sum(axis=1)).reshape(-1), power)
    degrees[~np.isfinite(degrees)] = 0.0
    return np.diag(degrees)


def norm_adj(adjacency: np.ndarray) -> np.ndarray:
    d_inv_sqrt = degree_power(adjacency, -0.5)
    return d_inv_sqrt @ adjacency @ d_inv_sqrt


def _pca_if_needed(matrix: np.ndarray, pca_dim: int | None) -> np.ndarray:
    if not pca_dim:
        return matrix
    max_dim = min(matrix.shape[0] - 1, matrix.shape[1])
    if max_dim <= 1:
        return matrix
    n_components = min(int(pca_dim), max_dim)
    if matrix.shape[1] <= n_components:
        return matrix
    return PCA(n_components=n_components, random_state=0).fit_transform(matrix)


def get_adj(
    count: np.ndarray,
    k: int = 15,
    pca: int | None = 50,
    mode: str = "connectivity",
):
    """Construct a symmetric KNN graph and its normalized adjacency target."""
    matrix = np.asarray(count, dtype=np.float32)
    n_nodes = matrix.shape[0]
    if n_nodes < 2:
        adjacency = np.eye(n_nodes, dtype=np.float32)
        return adjacency, adjacency

    features = _pca_if_needed(matrix, pca)
    n_neighbors = max(1, min(int(k), n_nodes - 1))
    sparse_adj = kneighbors_graph(
        features,
        n_neighbors=n_neighbors,
        mode=mode,
        metric="euclidean",
        include_self=True,
        n_jobs=-1,
    )
    adjacency = sparse_adj.toarray().astype(np.float32)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    normalized = norm_adj(adjacency).astype(np.float32)
    return adjacency, normalized


def sanitize_external_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("External cell adjacency must be a square matrix.")
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    adjacency[adjacency < 0] = 0.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency
