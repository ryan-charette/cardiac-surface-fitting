"""PyTorch Chamfer-distance helpers."""

from __future__ import annotations


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for cardiac_geometry.torch_ops. "
            "Install with `pip install -e .[torch]`."
        ) from exc
    return torch


def chamfer_distance_torch(source, target, *, squared: bool = True):
    """Symmetric Chamfer distance using torch.cdist."""

    torch = _require_torch()
    src = torch.as_tensor(source)
    tgt = torch.as_tensor(target, device=src.device, dtype=src.dtype)
    if src.ndim != 2 or tgt.ndim != 2 or src.shape[1] != 3 or tgt.shape[1] != 3:
        raise ValueError("source and target must have shape (num_points, 3)")
    distances = torch.cdist(src, tgt, p=2)
    if squared:
        distances = distances * distances
    return distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()

