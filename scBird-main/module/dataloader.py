from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from module.graph_function import get_adj, norm_adj, sanitize_external_adjacency
from utils.preprocess import prepro, read_h5


@dataclass
class ExpressionData:
    x_graph: np.ndarray
    x_bridge: np.ndarray
    x_count: np.ndarray
    size_factors: np.ndarray
    labels: np.ndarray
    gene_names: np.ndarray


def _dense_float32(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)




import os

import h5py
import numpy as np


def _decode_h5_labels(values: np.ndarray) -> np.ndarray:
    """Decode byte-string labels stored in an HDF5 dataset."""
    values = np.asarray(values)

    if values.dtype.kind in {"S", "O"}:
        decoded = []

        for value in values.reshape(-1):
            if isinstance(value, bytes):
                decoded.append(value.decode("utf-8"))
            else:
                decoded.append(str(value))

        return np.asarray(decoded)

    return values


def _read_xy_h5(path: str):
    """Read an HDF5 dataset whose top-level keys are X and Y.

    X:
        Expression matrix. The expected final orientation is cells × genes.

    Y:
        Cell labels. Supports one-dimensional labels, column-vector labels,
        row-vector labels, and one-hot encoded labels.
    """
    with h5py.File(path, "r") as handle:
        available_keys = list(handle.keys())

        if "X" not in handle or "Y" not in handle:
            raise KeyError(
                "The HDF5 file does not contain both 'X' and 'Y'.\n"
                f"Available keys: {available_keys}"
            )

        x_object = handle["X"]

        # Most X/Y datasets store X directly as a dense HDF5 dataset.
        if isinstance(x_object, h5py.Dataset):
            raw_x = x_object[...]

        # Also support AnnData-style sparse matrix groups if encountered.
        elif isinstance(x_object, h5py.Group):
            sparse_keys = {"data", "indices", "indptr", "shape"}

            if not sparse_keys.issubset(set(x_object.keys())):
                raise ValueError(
                    "The 'X' object is an HDF5 group, but its sparse matrix "
                    "structure is not recognized.\n"
                    f"Keys under X: {list(x_object.keys())}"
                )

            import scipy.sparse as sp

            data = x_object["data"][...]
            indices = x_object["indices"][...]
            indptr = x_object["indptr"][...]
            shape = tuple(x_object["shape"][...].astype(int))

            encoding_type = x_object.attrs.get(
                "encoding-type",
                x_object.attrs.get("h5sparse_format", "csr_matrix"),
            )

            if isinstance(encoding_type, bytes):
                encoding_type = encoding_type.decode("utf-8")

            if "csc" in str(encoding_type).lower():
                raw_x = sp.csc_matrix(
                    (data, indices, indptr),
                    shape=shape,
                ).toarray()
            else:
                raw_x = sp.csr_matrix(
                    (data, indices, indptr),
                    shape=shape,
                ).toarray()

        else:
            raise TypeError(
                f"Unsupported HDF5 object type for X: {type(x_object)}"
            )

        labels = handle["Y"][...]

    raw_x = np.asarray(raw_x)
    labels = _decode_h5_labels(labels)

    if raw_x.ndim != 2:
        raise ValueError(
            f"X must be a two-dimensional expression matrix, "
            f"but its shape is {raw_x.shape}."
        )

    # Convert Y to a one-dimensional label vector.
    labels = np.asarray(labels)

    if labels.ndim == 0:
        labels = labels.reshape(1)

    elif labels.ndim == 1:
        labels = labels.reshape(-1)

    elif labels.ndim == 2:
        # Shape: cells × 1 or 1 × cells
        if labels.shape[1] == 1 or labels.shape[0] == 1:
            labels = labels.reshape(-1)

        # Shape: cells × classes, one-hot or probability labels
        elif labels.shape[0] == raw_x.shape[0]:
            labels = np.argmax(labels, axis=1)

        # Shape: classes × cells
        elif labels.shape[1] == raw_x.shape[0]:
            labels = np.argmax(labels, axis=0)

        else:
            raise ValueError(
                "Unable to determine the meaning of the two-dimensional "
                "Y matrix.\n"
                f"X shape: {raw_x.shape}\n"
                f"Y shape: {labels.shape}"
            )

    else:
        raise ValueError(
            f"Y must be one- or two-dimensional, but its shape is "
            f"{labels.shape}."
        )

    # Ensure the expression matrix orientation is cells × genes.
    if raw_x.shape[0] != labels.shape[0]:
        if raw_x.shape[1] == labels.shape[0]:
            print(
                "[Data loader] X is stored as genes × cells. "
                "Transposing it to cells × genes."
            )
            raw_x = raw_x.T
        else:
            raise ValueError(
                "The label count does not match either dimension of X.\n"
                f"X shape: {raw_x.shape}\n"
                f"Number of labels: {labels.shape[0]}"
            )

    print(
        "[Data loader] Loaded the dataset directly from HDF5 keys X and Y:\n"
        f"  X shape: {raw_x.shape}\n"
        f"  Y shape: {labels.shape}"
    )

    return raw_x, labels


def load_expression_data(
    dataname: str,
    highly_genes: int = 500,
    data_root: str = "data",
    h5_key_format: bool = False,
) -> ExpressionData:
    """Load counts and return aligned matrices for graph learning and ZINB.

    x_graph:
        Scaled log-normalized HVG matrix used by graph encoders.

    x_bridge:
        Non-negative log-normalized HVG matrix used as the cell-gene bridge.

    x_count:
        Raw HVG counts used only by the ZINB likelihood.
    """
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "scanpy and anndata are required for preprocessing. "
            "Install requirements.txt first."
        ) from exc

    path = os.path.join(
        data_root,
        dataname,
        "data.h5",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    # ============================================================
    # Select the reader according to the actual HDF5 structure.
    # ============================================================
    try:
        with h5py.File(path, "r") as handle:
            available_keys = list(handle.keys())
    except OSError as exc:
        raise OSError(
            f"Unable to open the HDF5 file: {path}"
        ) from exc

    print(
        f"[Data loader] HDF5 top-level keys: {available_keys}"
    )

    # Current mouse_ES_cell file: top-level keys are X and Y.
    if "X" in available_keys and "Y" in available_keys:
        raw_x, labels = _read_xy_h5(path)

    # Pandas HDFStore format, usually containing data and label keys.
    elif h5_key_format:
        try:
            raw_x, labels = read_h5(path)
        except ImportError as exc:
            raise ImportError(
                "This dataset uses the Pandas HDFStore format, which "
                "requires PyTables. Install it with:\n"
                "  conda install pytables\n"
                "or:\n"
                "  pip install tables"
            ) from exc

    # Original scGCOT format containing obs_names, var_names and exprs.
    else:
        raw_x, labels = prepro(path)

    # Convert sparse matrices to dense matrices if necessary.
    if hasattr(raw_x, "toarray"):
        raw_x = raw_x.toarray()

    raw_x = np.asarray(
        raw_x,
        dtype=np.float32,
    )

    labels = np.asarray(labels).reshape(-1)

    if raw_x.ndim != 2:
        raise ValueError(
            f"The expression matrix must be two-dimensional, "
            f"but got shape {raw_x.shape}."
        )

    # Final orientation check.
    if raw_x.shape[0] != labels.shape[0]:
        if raw_x.shape[1] == labels.shape[0]:
            raw_x = raw_x.T
        else:
            raise ValueError(
                "The number of labels does not match the number of cells.\n"
                f"Expression shape: {raw_x.shape}\n"
                f"Number of labels: {labels.shape[0]}"
            )

    # Preserve the original preprocessing behavior.
    raw_x = np.ceil(raw_x).astype(np.float32)

    # Convert arbitrary labels into consecutive integer class IDs.
    try:
        labels = labels.astype(np.int64)
    except (TypeError, ValueError):
        _, labels = np.unique(
            labels.astype(str),
            return_inverse=True,
        )
        labels = labels.astype(np.int64)

    adata = ad.AnnData(raw_x)

    adata.obs["label"] = labels

    adata.var_names = np.asarray(
        [
            f"gene_{idx}"
            for idx in range(adata.n_vars)
        ],
        dtype=str,
    )

    # Remove genes and cells with zero total counts.
    sc.pp.filter_genes(
        adata,
        min_counts=1,
    )
    sc.pp.filter_cells(
        adata,
        min_counts=1,
    )

    labels = np.asarray(
        adata.obs["label"],
        dtype=np.int64,
    )

    # Save filtered raw counts before normalization.
    adata.layers["counts"] = adata.X.copy()

    totals = np.asarray(
        adata.X.sum(axis=1)
    ).reshape(-1).astype(np.float32)

    positive_totals = totals[totals > 0]

    median_total = (
        float(np.median(positive_totals))
        if positive_totals.size > 0
        else 1.0
    )

    size_factors = np.maximum(
        totals / max(median_total, 1e-8),
        1e-4,
    ).astype(np.float32)

    # Log-normalized expression.
    sc.pp.normalize_total(
        adata,
        target_sum=1e4,
    )
    sc.pp.log1p(adata)

    # Select highly variable genes.
    n_top = min(
        int(highly_genes),
        adata.n_vars,
    )

    if n_top <= 0:
        raise ValueError(
            f"highly_genes must be positive, but got {highly_genes}."
        )

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top,
        flavor="seurat",
        subset=False,
    )

    hvg_mask = np.asarray(
        adata.var["highly_variable"],
        dtype=bool,
    )

    # Variance-based fallback.
    if hvg_mask.sum() == 0:
        variances = np.var(
            _dense_float32(adata.X),
            axis=0,
        )

        indices = np.argsort(
            variances
        )[-n_top:]

        hvg_mask = np.zeros(
            adata.n_vars,
            dtype=bool,
        )

        hvg_mask[indices] = True

    adata = adata[:, hvg_mask].copy()

    # Non-negative log-normalized matrix for cross-view projection.
    x_bridge = _dense_float32(
        adata.X
    )

    # Raw HVG counts for the ZINB likelihood.
    x_count = _dense_float32(
        adata.layers["counts"]
    )

    gene_names = np.asarray(
        adata.var_names,
        dtype=str,
    )

    # Scaled matrix for graph encoders and graph construction.
    graph_adata = adata.copy()

    sc.pp.scale(
        graph_adata,
        zero_center=True,
        max_value=10,
    )

    x_graph = _dense_float32(
        graph_adata.X
    )

    # Check that all three matrices remain aligned.
    if not (
        x_graph.shape
        == x_bridge.shape
        == x_count.shape
    ):
        raise RuntimeError(
            "The processed matrices are not aligned.\n"
            f"x_graph shape: {x_graph.shape}\n"
            f"x_bridge shape: {x_bridge.shape}\n"
            f"x_count shape: {x_count.shape}"
        )

    if x_graph.shape[0] != labels.shape[0]:
        raise RuntimeError(
            "The number of cells does not match the label count.\n"
            f"Cell count: {x_graph.shape[0]}\n"
            f"Label count: {labels.shape[0]}"
        )

    if size_factors.shape[0] != x_graph.shape[0]:
        raise RuntimeError(
            "The size-factor count does not match the cell count.\n"
            f"Size-factor count: {size_factors.shape[0]}\n"
            f"Cell count: {x_graph.shape[0]}"
        )

    print(
        "[Data loader] Preprocessing completed:\n"
        f"  cells: {x_graph.shape[0]}\n"
        f"  selected genes: {x_graph.shape[1]}\n"
        f"  clusters: {len(np.unique(labels))}\n"
        f"  x_graph shape: {x_graph.shape}\n"
        f"  x_bridge shape: {x_bridge.shape}\n"
        f"  x_count shape: {x_count.shape}"
    )

    return ExpressionData(
        x_graph=x_graph,
        x_bridge=x_bridge,
        x_count=x_count,
        size_factors=size_factors,
        labels=labels,
        gene_names=gene_names,
    )





def _edge_index_from_adjacency(adjacency: np.ndarray) -> torch.Tensor:
    rows, cols = np.nonzero(adjacency > 0)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)


def make_graph_data(
    matrix: np.ndarray,
    k: int = 15,
    pca_dim: int = 50,
    adjacency_override: Optional[np.ndarray] = None,
):
    if adjacency_override is None:
        adjacency, adjacency_target = get_adj(matrix, k=k, pca=pca_dim)
    else:
        adjacency = sanitize_external_adjacency(adjacency_override)
        if adjacency.shape[0] != matrix.shape[0]:
            raise ValueError(
                f"External adjacency has {adjacency.shape[0]} nodes, but the feature matrix has {matrix.shape[0]}."
            )
        adjacency_target = norm_adj(adjacency).astype(np.float32)

    graph = Data(
        x=torch.tensor(matrix, dtype=torch.float32),
        edge_index=_edge_index_from_adjacency(adjacency),
    )
    return graph, adjacency_target


def prepare_cell_gene_graphs(
    x_graph: np.ndarray,
    k: int = 15,
    pca_dim: int = 50,
    cell_adjacency: Optional[np.ndarray] = None,
):
    cell_graph, cell_target = make_graph_data(
        x_graph,
        k=k,
        pca_dim=pca_dim,
        adjacency_override=cell_adjacency,
    )
    gene_graph, gene_target = make_graph_data(
        x_graph.T,
        k=k,
        pca_dim=pca_dim,
        adjacency_override=None,
    )
    return cell_graph, cell_target, gene_graph, gene_target


# Compatibility wrappers retained for older scripts.
def DataLoaderMatrix(matrix: np.ndarray, k: int = 15, pca_dim: int = 50):
    return make_graph_data(matrix, k=k, pca_dim=pca_dim)


def HomoData(matrix: np.ndarray, adjacency: np.ndarray):
    graph, _ = make_graph_data(matrix, adjacency_override=adjacency)
    return graph
