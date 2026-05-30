"""PyTorch evaluator for fitted FasTFit patch networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any

import numpy as np


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for cardiac_geometry.torch_ops. "
            "Install with `pip install -e .[torch]`."
        ) from exc
    return torch


@dataclass
class TorchFastFitSurface:
    """Framework-native evaluator for a fixed fitted Bezier patch network."""

    degree: int
    patch_bounds: Any
    control_points: Any
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    @classmethod
    def from_fastfit_surface(cls, surface, *, device=None, dtype=None):
        torch = _require_torch()
        side = surface.degree + 1
        bounds = np.asarray(
            [
                [patch.u_min, patch.u_max, patch.v_min, patch.v_max]
                for patch in surface.patches
            ],
            dtype=np.float64,
        )
        controls = np.stack(
            [patch.control_points.reshape(side, side, 3) for patch in surface.patches],
            axis=0,
        )
        bounds_t = torch.as_tensor(bounds, device=device, dtype=dtype or torch.float64)
        controls_t = torch.as_tensor(
            controls, device=device, dtype=dtype or torch.float64
        )
        return cls(
            degree=surface.degree,
            patch_bounds=bounds_t,
            control_points=controls_t,
            diagnostics=dict(surface.diagnostics),
        )

    @property
    def num_patches(self) -> int:
        return int(self.patch_bounds.shape[0])

    @property
    def num_control_points(self) -> int:
        side = self.degree + 1
        return self.num_patches * side * side

    def evaluate(self, u, v):
        """Evaluate on vector or matching mesh parameter tensors."""

        torch = _require_torch()
        u_t = torch.as_tensor(
            u, device=self.control_points.device, dtype=self.control_points.dtype
        )
        v_t = torch.as_tensor(
            v, device=self.control_points.device, dtype=self.control_points.dtype
        )
        if u_t.ndim == 1 and v_t.ndim == 1:
            uu, vv = torch.meshgrid(u_t, v_t, indexing="ij")
        elif u_t.ndim == 2 and v_t.ndim == 2 and u_t.shape == v_t.shape:
            uu, vv = u_t, v_t
        else:
            raise ValueError("u and v must either both be 1D or matching 2D tensors")
        params = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        return self.evaluate_points(params).reshape(uu.shape + (3,))

    def evaluate_points(self, parameters):
        """Evaluate at paired ``(u, v)`` samples with autograd support."""

        torch = _require_torch()
        params = torch.as_tensor(
            parameters,
            device=self.control_points.device,
            dtype=self.control_points.dtype,
        )
        if params.ndim != 2 or params.shape[1] != 2:
            raise ValueError("parameters must have shape (num_samples, 2)")
        if bool(torch.any(~torch.isfinite(params)).detach().cpu()):
            raise ValueError("parameters must contain only finite values")

        patch_indices = self._locate_patches(params)
        missing = patch_indices < 0
        if bool(torch.any(missing).detach().cpu()):
            count = int(torch.count_nonzero(missing).detach().cpu())
            raise ValueError(f"{count} evaluation sample(s) fell outside all patches")

        out = torch.empty((params.shape[0], 3), device=params.device, dtype=params.dtype)
        for patch_index in range(self.num_patches):
            mask = patch_indices == patch_index
            if bool(torch.any(mask).detach().cpu()):
                out[mask] = self._evaluate_patch(
                    patch_index, params[mask, 0], params[mask, 1]
                )
        return out

    def _locate_patches(self, params):
        torch = _require_torch()
        indices = torch.full(
            (params.shape[0],), -1, device=params.device, dtype=torch.long
        )
        for patch_index in range(self.num_patches):
            u_min, u_max, v_min, v_max = self.patch_bounds[patch_index]
            mask = (
                (params[:, 0] >= u_min)
                & (params[:, 0] <= u_max)
                & (params[:, 1] >= v_min)
                & (params[:, 1] <= v_max)
                & (indices < 0)
            )
            indices = torch.where(mask, torch.full_like(indices, patch_index), indices)
        return indices

    def _evaluate_patch(self, patch_index: int, u, v):
        bounds = self.patch_bounds[patch_index]
        u_min, u_max, v_min, v_max = bounds
        u_local = (u - u_min) / (u_max - u_min)
        v_local = (v - v_min) / (v_max - v_min)
        bu = _bernstein_basis_torch(u_local, self.degree)
        bv = _bernstein_basis_torch(v_local, self.degree)
        return _require_torch().einsum(
            "...i,...j,ijc->...c", bu, bv, self.control_points[patch_index]
        )


def _bernstein_basis_torch(x, degree: int):
    if degree < 0:
        raise ValueError("degree must be non-negative")
    values = x.clamp(0.0, 1.0)
    basis = []
    for index in range(degree + 1):
        basis.append(
            comb(degree, index)
            * (values**index)
            * ((1.0 - values) ** (degree - index))
        )
    return _require_torch().stack(basis, dim=-1)

