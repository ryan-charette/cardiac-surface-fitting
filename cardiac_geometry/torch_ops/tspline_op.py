"""PyTorch T-spline evaluator.

This module intentionally imports PyTorch lazily so the NumPy reference package
remains usable on machines without a GPU stack.
"""

from __future__ import annotations

from pathlib import Path


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for cardiac_geometry.torch_ops. "
            "Install with `pip install -e .[torch]`."
        ) from exc
    return torch


def _torch_basis_matrix(x, knots, degree):
    torch = _require_torch()
    flat = x.reshape(-1)
    out = []
    for cp in range(knots.shape[0]):
        row = knots[cp]
        x_eval = torch.where(
            flat == row[-1],
            torch.nextafter(flat, torch.full_like(flat, float("-inf"))),
            flat,
        )
        values = []
        for i in range(degree + 1):
            values.append(((row[i] <= x_eval) & (x_eval < row[i + 1])).to(x.dtype))
        work = torch.stack(values, dim=1)
        for p in range(1, degree + 1):
            pieces = []
            for i in range(degree + 1 - p):
                left_den = row[i + p] - row[i]
                right_den = row[i + p + 1] - row[i + 1]
                left = torch.zeros_like(x_eval)
                if float(left_den.detach().cpu()) != 0.0:
                    left = ((x_eval - row[i]) / left_den) * work[:, i]
                right = torch.zeros_like(x_eval)
                if float(right_den.detach().cpu()) != 0.0:
                    right = ((row[i + p + 1] - x_eval) / right_den) * work[:, i + 1]
                pieces.append(left + right)
            work = torch.stack(pieces, dim=1)
        out.append(work[:, 0])
    return torch.stack(out, dim=1).reshape(x.shape + (knots.shape[0],))


def evaluate_tspline_torch(
    u,
    v,
    knots_u,
    knots_v,
    control_points,
    weights,
    *,
    degree: int | None = None,
    normalize_basis: bool = False,
    zero_denominator_tol: float = 0.0,
):
    """Differentiable PyTorch evaluator using framework-native ops."""

    torch = _require_torch()
    points = torch.as_tensor(control_points)
    dtype = points.dtype if points.is_floating_point() else torch.float32
    device = points.device
    points = points.to(device=device, dtype=dtype)
    ku = torch.as_tensor(knots_u, device=device, dtype=dtype)
    kv = torch.as_tensor(knots_v, device=device, dtype=dtype)
    w = torch.as_tensor(weights, device=device, dtype=dtype)
    u_t = torch.as_tensor(u, device=device, dtype=dtype)
    v_t = torch.as_tensor(v, device=device, dtype=dtype)
    if degree is None:
        degree = int(ku.shape[1]) - 2

    if u_t.ndim == 1 and v_t.ndim == 1:
        bu = _torch_basis_matrix(u_t, ku, degree)
        bv = _torch_basis_matrix(v_t, kv, degree)
        basis = bu[:, None, :] * bv[None, :, :]
    elif u_t.ndim == 2 and v_t.ndim == 2 and u_t.shape == v_t.shape:
        basis = _torch_basis_matrix(u_t, ku, degree) * _torch_basis_matrix(v_t, kv, degree)
    else:
        raise ValueError("u and v must either both be 1D or matching 2D tensors")

    if normalize_basis:
        basis_sum = basis.sum(dim=-1)
        if torch.any(torch.abs(basis_sum) <= zero_denominator_tol):
            raise FloatingPointError("cannot normalize basis where support is zero")
        basis = basis / basis_sum.unsqueeze(-1)

    weighted = basis * w
    denominator = weighted.sum(dim=-1)
    if torch.any(torch.abs(denominator) <= zero_denominator_tol):
        raise FloatingPointError("rational T-spline denominator is zero")
    numerator = weighted @ points
    return numerator / denominator.unsqueeze(-1)


def load_cuda_extension(name: str = "cardiac_geometry_cuda"):
    """Build and load the CUDA extension with ``torch.utils.cpp_extension``."""

    torch = _require_torch()
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    return load(
        name=name,
        sources=[
            str(root / "cuda" / "bindings.cpp"),
            str(root / "cuda" / "tspline_eval_kernel.cu"),
            str(root / "cuda" / "chamfer_distance_kernel.cu"),
        ],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

