from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _decode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else item
            for item in values
        ])
    return values


def read_clean(data: np.ndarray) -> Any:
    if data.dtype.type is np.bytes_ or data.dtype.kind == "S":
        data = _decode(data)
    if data.size == 1:
        return data.flat[0]
    return data


def dict_from_group(group: h5py.Group) -> Dict[str, Any]:
    output: Dict[str, Any] = DotDict()
    for key in group:
        value = group[key]
        output[key] = dict_from_group(value) if isinstance(value, h5py.Group) else read_clean(value[...])
    return output


def read_data(filename: str, sparsify: bool = False, skip_exprs: bool = False):
    with h5py.File(filename, "r") as handle:
        obs_names = _decode(handle["obs_names"][...])
        var_names = _decode(handle["var_names"][...])
        obs = pd.DataFrame(dict_from_group(handle["obs"]), index=obs_names)
        var = pd.DataFrame(dict_from_group(handle["var"]), index=var_names)
        uns = dict_from_group(handle["uns"])

        if skip_exprs:
            matrix = sp.csr_matrix((obs.shape[0], var.shape[0]))
        else:
            exprs = handle["exprs"]
            if isinstance(exprs, h5py.Group):
                matrix = sp.csr_matrix(
                    (exprs["data"][...], exprs["indices"][...], exprs["indptr"][...]),
                    shape=exprs["shape"][...],
                )
            else:
                matrix = exprs[...].astype(np.float32)
                if sparsify:
                    matrix = sp.csr_matrix(matrix)
    return matrix, obs, var, uns


def prepro(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    matrix, obs, _, _ = read_data(filename, sparsify=False, skip_exprs=False)
    x = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    if "cell_type1" not in obs:
        raise KeyError("The input H5 file must contain obs['cell_type1'] labels.")
    cell_names = np.asarray(obs["cell_type1"])
    _, labels = np.unique(cell_names, return_inverse=True)
    return x.astype(np.float32), labels.astype(np.int64)


def read_h5(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_hdf(filename, key="data")
    x = np.asarray(data.values, dtype=np.float32)
    _, labels = np.unique(data.index, return_inverse=True)
    return x, labels.astype(np.int64)
