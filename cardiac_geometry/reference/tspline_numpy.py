"""NumPy reference evaluator for rational T-spline surfaces."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from cardiac_geometry.reference.bspline_basis import (
    basis_matrix,
    tensor_product_basis,
    validate_local_knot_vectors,
)


def _validate_inputs(
    knots_u: np.ndarray | Iterable[Iterable[float]],
    knots_v: np.ndarray | Iterable[Iterable[float]],
    control_points: np.ndarray | Iterable[Iterable[float]],
    weights: np.ndarray | Iterable[float],
    degree: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    ku, degree_u = validate_local_knot_vectors(knots_u, degree, name="knots_u")
    kv, degree_v = validate_local_knot_vectors(knots_v, degree, name="knots_v")
    if degree_u != degree_v:
        raise ValueError("knots_u and knots_v must describe the same degree")
    if ku.shape[0] != kv.shape[0]:
        raise ValueError("knots_u and knots_v row counts must match")

    points = np.asarray(control_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("control_points must have shape (num_control_points, 3)")
    if points.shape[0] != ku.shape[0]:
        raise ValueError("control_points row count must match knot vector rows")
    if not np.all(np.isfinite(points)):
        raise ValueError("control_points must contain only finite values")

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != points.shape[0]:
        raise ValueError("weights must have shape (num_control_points,)")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must contain only finite values")

    return ku, kv, points, w, degree_u


def _check_denominator(denominator: np.ndarray, zero_tol: float) -> None:
    zero_mask = np.abs(denominator) <= zero_tol
    if np.any(zero_mask):
        first = np.argwhere(zero_mask)[0].tolist()
        raise FloatingPointError(
            "rational T-spline denominator is zero at "
            f"{int(np.count_nonzero(zero_mask))} sample(s); first index {first}"
        )


def evaluate_tspline_numpy(
    u: np.ndarray | Iterable[float],
    v: np.ndarray | Iterable[float],
    knots_u: np.ndarray | Iterable[Iterable[float]],
    knots_v: np.ndarray | Iterable[Iterable[float]],
    control_points: np.ndarray | Iterable[Iterable[float]],
    weights: np.ndarray | Iterable[float],
    *,
    degree: int | None = None,
    normalize_basis: bool = False,
    zero_denominator_tol: float = 0.0,
) -> np.ndarray:
    """Evaluate a rational T-spline surface on a parameter grid.

    ``u`` and ``v`` may be 1D vectors or matching 2D parameter meshes. The
    returned array has shape ``(len(u), len(v), 3)`` for vector inputs or
    ``u.shape + (3,)`` for mesh inputs.
    """

    ku, kv, points, w, degree = _validate_inputs(
        knots_u, knots_v, control_points, weights, degree
    )
    basis = tensor_product_basis(
        u,
        v,
        ku,
        kv,
        degree=degree,
        normalize=normalize_basis,
        zero_tol=zero_denominator_tol,
    )
    weighted_basis = basis * w
    denominator = np.sum(weighted_basis, axis=-1)
    _check_denominator(denominator, zero_denominator_tol)

    numerator = np.einsum("...c,cd->...d", weighted_basis, points)
    return numerator / denominator[..., None]


def evaluate_tspline_numpy_fused(
    u: np.ndarray | Iterable[float],
    v: np.ndarray | Iterable[float],
    knots_u: np.ndarray | Iterable[Iterable[float]],
    knots_v: np.ndarray | Iterable[Iterable[float]],
    control_points: np.ndarray | Iterable[Iterable[float]],
    weights: np.ndarray | Iterable[float],
    *,
    degree: int | None = None,
    zero_denominator_tol: float = 0.0,
) -> np.ndarray:
    """Loop-based evaluator matching the fused GPU kernel design.

    This avoids materializing the ``U x V x C`` basis tensor. It is slower than
    the vectorized NumPy reference, but it mirrors the one-output-sample-per-
    thread CUDA/Triton mapping and is useful for correctness tests.
    """

    ku, kv, points, w, degree = _validate_inputs(
        knots_u, knots_v, control_points, weights, degree
    )
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    if u_arr.ndim == 1 and v_arr.ndim == 1:
        uu, vv = np.meshgrid(u_arr, v_arr, indexing="ij")
    elif u_arr.ndim == 2 and v_arr.ndim == 2 and u_arr.shape == v_arr.shape:
        uu, vv = u_arr, v_arr
    else:
        raise ValueError("u and v must either both be 1D or matching 2D meshes")

    out = np.empty(uu.shape + (3,), dtype=np.float64)
    for idx in np.ndindex(uu.shape):
        bu = basis_matrix(np.array([uu[idx]]), ku, degree).reshape(-1)
        bv = basis_matrix(np.array([vv[idx]]), kv, degree).reshape(-1)
        weighted = bu * bv * w
        denominator = np.sum(weighted)
        _check_denominator(np.array([denominator]), zero_denominator_tol)
        out[idx] = weighted @ points / denominator
    return out


def S(u, v, s, t, P, w, norm: bool = False, plot: bool = False):
    """Compatibility wrapper for the original ``tsplines.S`` API."""

    if plot:
        raise ValueError("plotting has moved out of the reference evaluator")
    return evaluate_tspline_numpy(u, v, s, t, P, w, normalize_basis=norm)


def S2(u, v, s, t, P, w, norm: bool = False, plot: bool = False):
    """Compatibility wrapper for the original ``tsplines.S2`` API."""

    if plot:
        raise ValueError("plotting has moved out of the reference evaluator")
    return evaluate_tspline_numpy(u, v, s, t, P, w, normalize_basis=norm)

