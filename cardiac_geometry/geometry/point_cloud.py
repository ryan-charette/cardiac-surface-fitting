"""Point-cloud geometry reference functions."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_points(name: str, value: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (num_points, 3)")
    if points.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def _one_sided_min_distances(
    source: np.ndarray,
    target: np.ndarray,
    *,
    squared: bool,
    chunk_size: int,
) -> np.ndarray:
    mins = np.empty(source.shape[0], dtype=np.float64)
    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start : start + chunk_size]
        diff = chunk[:, None, :] - target[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        mins[start : start + chunk.shape[0]] = np.min(dist2, axis=1)
    if not squared:
        mins = np.sqrt(mins)
    return mins


def chamfer_distance_numpy(
    source: np.ndarray | Iterable[Iterable[float]],
    target: np.ndarray | Iterable[Iterable[float]],
    *,
    squared: bool = True,
    chunk_size: int = 4096,
    return_parts: bool = False,
) -> float | tuple[float, float, float]:
    """Compute a symmetric Chamfer distance reference in NumPy."""

    src = _as_points("source", source)
    tgt = _as_points("target", target)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    forward = float(
        np.mean(_one_sided_min_distances(src, tgt, squared=squared, chunk_size=chunk_size))
    )
    backward = float(
        np.mean(_one_sided_min_distances(tgt, src, squared=squared, chunk_size=chunk_size))
    )
    total = forward + backward
    if return_parts:
        return total, forward, backward
    return total

