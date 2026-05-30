"""Point-cloud parameterization helpers for surface fitting."""

from __future__ import annotations

import numpy as np


def principal_components(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centered points, descending eigenvalues, and PCA basis vectors."""

    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points must contain only finite coordinates")

    centered = arr - arr.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(arr.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return centered, eigenvalues[order], eigenvectors[:, order]


def normalize_columns(values: np.ndarray) -> np.ndarray:
    """Normalize each column to [0, 1], preserving finite shape."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must be a 2D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must contain only finite entries")

    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    return (arr - mins) / spans


def pca_parameterize(points: np.ndarray, axes: tuple[int, int] = (0, 1)) -> np.ndarray:
    """Planar PCA parameterization used as the original unorganized fallback."""

    centered, _, basis = principal_components(points)
    if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("axes must contain two distinct PCA axis indices")
    return normalize_columns(centered @ basis[:, list(axes)])


def spherical_parameterize(
    points: np.ndarray,
    *,
    pole_axis: int = 0,
    seam: str = "largest_gap",
) -> np.ndarray:
    """Map a curved point cloud to angular coordinates around its centroid.

    This parameterization is intended for sparse, curved, roughly star-shaped
    anatomical point clouds where planar PCA projection folds distant surface
    regions onto the same ``(u, v)`` coordinates. Holes remain empty regions in
    angular space, which is usually preferable to forcing a dense rectangular
    projection through missing anatomy.
    """

    if pole_axis not in (0, 1, 2):
        raise ValueError("pole_axis must be 0, 1, or 2")
    centered, _, basis = principal_components(points)
    coords = centered @ basis
    equator_axes = [axis for axis in (0, 1, 2) if axis != pole_axis]
    x = coords[:, equator_axes[0]]
    y = coords[:, equator_axes[1]]
    z = coords[:, pole_axis]
    radius = np.linalg.norm(coords, axis=1)
    safe_radius = np.where(radius > 0.0, radius, 1.0)

    theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    theta = np.mod(theta - _seam_angle(theta, mode=seam), 2.0 * np.pi)
    phi = np.arccos(np.clip(z / safe_radius, -1.0, 1.0))
    return np.column_stack([theta / (2.0 * np.pi), phi / np.pi])


def _seam_angle(theta: np.ndarray, *, mode: str) -> float:
    if mode == "zero":
        return 0.0
    if mode != "largest_gap":
        raise ValueError("seam must be 'largest_gap' or 'zero'")
    if theta.size < 2:
        return 0.0

    sorted_theta = np.sort(theta)
    wrapped = np.concatenate([sorted_theta, sorted_theta[:1] + 2.0 * np.pi])
    gaps = np.diff(wrapped)
    gap_index = int(np.argmax(gaps))
    return float(np.mod(sorted_theta[gap_index] + 0.5 * gaps[gap_index], 2.0 * np.pi))
