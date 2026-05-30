"""Robust local B-spline basis evaluation.

The original project used a direct recursive Cox-de Boor implementation. This
module keeps the same local-knot-vector convention, but uses an iterative
implementation with explicit zero-denominator handling. That matters for GPU
ports because recursion and NaN-producing divisions are both poor fits for
kernel code.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def validate_local_knot_vectors(
    knots: np.ndarray | Iterable[Iterable[float]],
    degree: int | None = None,
    *,
    name: str = "knots",
) -> tuple[np.ndarray, int]:
    """Return validated local knot vectors and the inferred degree.

    The repository's T-spline representation stores one local knot vector per
    control point. A degree-p local basis uses p + 2 knot entries.
    """

    arr = np.asarray(knots, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if arr.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two knots per row")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")

    inferred_degree = arr.shape[1] - 2
    if degree is None:
        degree = inferred_degree
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if arr.shape[1] != degree + 2:
        raise ValueError(
            f"{name} has width {arr.shape[1]}, expected degree + 2 = {degree + 2}"
        )
    if np.any(np.diff(arr, axis=1) < 0.0):
        raise ValueError(f"{name} rows must be nondecreasing")
    if np.any(arr[:, 0] >= arr[:, -1]):
        raise ValueError(f"{name} rows must have nonzero support")
    return np.ascontiguousarray(arr), degree


def _basis_one_unchecked(x: float, knots: np.ndarray, degree: int) -> float:
    """Evaluate one local basis function using Cox-de Boor iteration."""

    x_eval = float(x)
    if not np.isfinite(x_eval):
        raise ValueError("parameter samples must be finite")

    # Return the left-limit at the right endpoint. This avoids a common surface
    # sampling trap where np.linspace includes the upper bound and every
    # half-open basis interval evaluates to zero.
    if x_eval == knots[-1]:
        x_eval = float(np.nextafter(x_eval, -np.inf))

    values = np.zeros(degree + 1, dtype=np.float64)
    for i in range(degree + 1):
        if knots[i] <= x_eval < knots[i + 1]:
            values[i] = 1.0

    for p in range(1, degree + 1):
        next_values = np.zeros(degree + 1 - p, dtype=np.float64)
        for i in range(degree + 1 - p):
            left_den = knots[i + p] - knots[i]
            right_den = knots[i + p + 1] - knots[i + 1]

            left = 0.0
            if left_den != 0.0:
                left = ((x_eval - knots[i]) / left_den) * values[i]

            right = 0.0
            if right_den != 0.0:
                right = ((knots[i + p + 1] - x_eval) / right_den) * values[i + 1]

            next_values[i] = left + right
        values = next_values

    return float(values[0])


def bspline_basis_one(
    x: float,
    knots: np.ndarray | Iterable[float],
    degree: int | None = None,
) -> float:
    """Evaluate a single local B-spline basis value."""

    rows, degree = validate_local_knot_vectors(knots, degree)
    return _basis_one_unchecked(x, rows[0], degree)


def basis_matrix(
    x: np.ndarray | Iterable[float],
    local_knots: np.ndarray | Iterable[Iterable[float]],
    degree: int | None = None,
) -> np.ndarray:
    """Evaluate all local basis functions for a vector of samples.

    Returns an array with shape ``x.shape + (num_control_points,)``.
    """

    knots, degree = validate_local_knot_vectors(local_knots, degree)
    samples = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(samples)):
        raise ValueError("parameter samples must be finite")

    flat = samples.reshape(-1)
    out = np.empty((flat.shape[0], knots.shape[0]), dtype=np.float64)
    for sample_index, sample in enumerate(flat):
        for control_index in range(knots.shape[0]):
            out[sample_index, control_index] = _basis_one_unchecked(
                float(sample), knots[control_index], degree
            )
    return out.reshape(samples.shape + (knots.shape[0],))


def tensor_product_basis(
    u: np.ndarray | Iterable[float],
    v: np.ndarray | Iterable[float],
    knots_u: np.ndarray | Iterable[Iterable[float]],
    knots_v: np.ndarray | Iterable[Iterable[float]],
    *,
    degree: int | None = None,
    normalize: bool = False,
    zero_tol: float = 0.0,
) -> np.ndarray:
    """Evaluate tensor-product local basis values.

    ``u`` and ``v`` may both be 1D vectors, producing an output shaped
    ``(num_u, num_v, num_control_points)``, or both be 2D parameter meshes with
    identical shape, producing ``u.shape + (num_control_points,)``.
    """

    ku, degree_u = validate_local_knot_vectors(knots_u, degree, name="knots_u")
    kv, degree_v = validate_local_knot_vectors(knots_v, degree, name="knots_v")
    if ku.shape[0] != kv.shape[0]:
        raise ValueError("knots_u and knots_v must have the same number of rows")
    if degree_u != degree_v:
        raise ValueError("knots_u and knots_v must have the same degree")

    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    if u_arr.ndim == 1 and v_arr.ndim == 1:
        bu = basis_matrix(u_arr, ku, degree_u)
        bv = basis_matrix(v_arr, kv, degree_v)
        basis = bu[:, None, :] * bv[None, :, :]
    elif u_arr.ndim == 2 and v_arr.ndim == 2:
        if u_arr.shape != v_arr.shape:
            raise ValueError("2D u and v parameter meshes must have the same shape")
        bu = basis_matrix(u_arr, ku, degree_u)
        bv = basis_matrix(v_arr, kv, degree_v)
        basis = bu * bv
    else:
        raise ValueError("u and v must either both be 1D or both be 2D")

    if normalize:
        basis_sum = np.sum(basis, axis=-1)
        zero_mask = np.abs(basis_sum) <= zero_tol
        if np.any(zero_mask):
            raise FloatingPointError(
                f"cannot normalize basis at {int(np.count_nonzero(zero_mask))} "
                "sample(s) with zero support"
            )
        basis = basis / basis_sum[..., None]

    return basis

