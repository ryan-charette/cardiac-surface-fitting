"""Parallel FasTFit-style T-spline fitting.

This module implements the split-fit part of Feng and Taguchi's FasTFit
algorithm for fixed 2D-parameterized point clouds. The returned model is a
piecewise Bezier patch network. The adaptive split stage fits independent
candidate regions in parallel, then the default output is globally refitted
with hard C1 continuity constraints across shared patch edges. The independent
full-multiplicity baseline remains available with ``continuity="none"``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import comb
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FastFitOptions:
    """Configuration for the parallel FasTFit implementation."""

    degree: int = 3
    max_error: float = 0.02
    initial_splits: tuple[int, int] = (4, 4)
    max_depth: int = 10
    smoothing: float = 0.0
    min_points_per_patch: int | None = None
    workers: int | None = None
    keep_unsplittable: bool = True
    min_region_fill_ratio: float = 0.0
    support_grid: int = 8
    fairing: float | None = None
    sparse_anchor_weight: float = 0.1
    continuity: str = "c1"

    def control_count(self) -> int:
        return (self.degree + 1) * (self.degree + 1)

    def effective_min_points(self) -> int:
        if self.min_points_per_patch is not None:
            return self.min_points_per_patch
        return self.control_count()

    def fairing_weight(self) -> float:
        """Return the curvature penalty used to suppress control-net oscillation."""

        if self.fairing is not None:
            return self.fairing
        return 50.0 * self.smoothing


@dataclass(frozen=True)
class _Region:
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    depth: int = 0

    def contains(self, params: np.ndarray, *, include_upper: bool = True) -> np.ndarray:
        if include_upper:
            return (
                (params[:, 0] >= self.u_min)
                & (params[:, 0] <= self.u_max)
                & (params[:, 1] >= self.v_min)
                & (params[:, 1] <= self.v_max)
            )
        return (
            (params[:, 0] >= self.u_min)
            & (params[:, 0] < self.u_max)
            & (params[:, 1] >= self.v_min)
            & (params[:, 1] < self.v_max)
        )

    def split(self) -> tuple["_Region", "_Region"]:
        u_span = self.u_max - self.u_min
        v_span = self.v_max - self.v_min
        if u_span >= v_span:
            mid = 0.5 * (self.u_min + self.u_max)
            return (
                _Region(self.u_min, mid, self.v_min, self.v_max, self.depth + 1),
                _Region(mid, self.u_max, self.v_min, self.v_max, self.depth + 1),
            )
        mid = 0.5 * (self.v_min + self.v_max)
        return (
            _Region(self.u_min, self.u_max, self.v_min, mid, self.depth + 1),
            _Region(self.u_min, self.u_max, mid, self.v_max, self.depth + 1),
        )


@dataclass(frozen=True)
class _SharedBoundary:
    """A positive-length edge shared by two fitted rectangular patches."""

    lower_index: int
    upper_index: int
    axis: str
    start: float
    end: float


@dataclass(frozen=True)
class BezierPatch:
    """One fitted Bezier patch over a rectangular parameter domain."""

    u_min: float
    u_max: float
    v_min: float
    v_max: float
    degree: int
    control_points: np.ndarray
    rmse: float
    max_error: float
    num_points: int
    rank: int
    depth: int

    def evaluate(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate this patch for same-shaped parameter arrays."""

        u_local = _normalize_parameter(u, self.u_min, self.u_max)
        v_local = _normalize_parameter(v, self.v_min, self.v_max)
        bu = bernstein_basis(u_local, self.degree)
        bv = bernstein_basis(v_local, self.degree)
        controls = self.control_points.reshape(
            self.degree + 1, self.degree + 1, 3
        )
        return np.einsum("...i,...j,ijc->...c", bu, bv, controls)

    def local_knot_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return local knot vectors/control points for the full-multiplicity form."""

        ku = []
        kv = []
        anchors = []
        points = []
        for i in range(self.degree + 1):
            u_knots = np.array(
                [self.u_min] * (self.degree + 1 - i)
                + [self.u_max] * (i + 1),
                dtype=np.float64,
            )
            for j in range(self.degree + 1):
                v_knots = np.array(
                    [self.v_min] * (self.degree + 1 - j)
                    + [self.v_max] * (j + 1),
                    dtype=np.float64,
                )
                ku.append(u_knots)
                kv.append(v_knots)
                anchors.append(
                    [
                        self.u_min + (self.u_max - self.u_min) * i / self.degree,
                        self.v_min + (self.v_max - self.v_min) * j / self.degree,
                    ]
                )
                points.append(self.control_points[i * (self.degree + 1) + j])
        return (
            np.asarray(ku),
            np.asarray(kv),
            np.asarray(points),
            np.asarray(anchors),
        )


@dataclass(frozen=True)
class FastFitSurface:
    """A fitted piecewise Bezier/T-spline surface."""

    patches: tuple[BezierPatch, ...]
    degree: int
    domain: tuple[float, float, float, float]
    options: FastFitOptions
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def num_control_points(self) -> int:
        return len(self.patches) * (self.degree + 1) * (self.degree + 1)

    def evaluate(
        self,
        u: np.ndarray | Iterable[float],
        v: np.ndarray | Iterable[float],
        *,
        workers: int | None = None,
    ) -> np.ndarray:
        """Evaluate the fitted surface on vector or mesh parameters."""

        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        vector_input = u_arr.ndim == 1 and v_arr.ndim == 1
        if vector_input:
            uu, vv = np.meshgrid(u_arr, v_arr, indexing="ij")
        elif u_arr.ndim == 2 and v_arr.ndim == 2 and u_arr.shape == v_arr.shape:
            uu, vv = u_arr, v_arr
        else:
            raise ValueError("u and v must either both be 1D or matching 2D arrays")

        params = np.column_stack([uu.reshape(-1), vv.reshape(-1)])
        out = np.full((params.shape[0], 3), np.nan, dtype=np.float64)
        patch_indices = self._locate_patches(params)

        def eval_patch(patch_index: int) -> tuple[np.ndarray, np.ndarray]:
            mask = patch_indices == patch_index
            patch = self.patches[patch_index]
            values = patch.evaluate(params[mask, 0], params[mask, 1])
            return np.flatnonzero(mask), values

        active = [idx for idx in range(len(self.patches)) if np.any(patch_indices == idx)]
        if workers is None:
            workers = self.options.workers
        if workers is not None and workers > 1 and len(active) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for indices, values in pool.map(eval_patch, active):
                    out[indices] = values
        else:
            for patch_index in active:
                indices, values = eval_patch(patch_index)
                out[indices] = values

        if np.any(~np.isfinite(out)):
            missing = int(np.count_nonzero(~np.isfinite(out[:, 0])))
            raise ValueError(f"{missing} evaluation sample(s) fell outside all patches")
        return out.reshape(uu.shape + (3,))

    def evaluate_points(
        self,
        parameters: np.ndarray | Iterable[Iterable[float]],
        *,
        workers: int | None = None,
    ) -> np.ndarray:
        """Evaluate this model at paired ``(u, v)`` parameter samples."""

        params = np.asarray(parameters, dtype=np.float64)
        if params.ndim != 2 or params.shape[1] != 2:
            raise ValueError("parameters must have shape (num_samples, 2)")
        if not np.all(np.isfinite(params)):
            raise ValueError("parameters must contain only finite values")

        out = np.full((params.shape[0], 3), np.nan, dtype=np.float64)
        patch_indices = self._locate_patches(params)

        def eval_patch(patch_index: int) -> tuple[np.ndarray, np.ndarray]:
            mask = patch_indices == patch_index
            patch = self.patches[patch_index]
            values = patch.evaluate(params[mask, 0], params[mask, 1])
            return np.flatnonzero(mask), values

        active = [idx for idx in range(len(self.patches)) if np.any(patch_indices == idx)]
        if workers is None:
            workers = self.options.workers
        if workers is not None and workers > 1 and len(active) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for indices, values in pool.map(eval_patch, active):
                    out[indices] = values
        else:
            for patch_index in active:
                indices, values = eval_patch(patch_index)
                out[indices] = values

        if np.any(~np.isfinite(out)):
            missing = int(np.count_nonzero(~np.isfinite(out[:, 0])))
            raise ValueError(f"{missing} evaluation sample(s) fell outside all patches")
        return out

    def to_local_knot_surface(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a full-multiplicity local-knot representation of all patches."""

        knots_u = []
        knots_v = []
        points = []
        anchors = []
        for patch in self.patches:
            ku, kv, cp, an = patch.local_knot_vectors()
            knots_u.append(ku)
            knots_v.append(kv)
            points.append(cp)
            anchors.append(an)
        return (
            np.vstack(knots_u),
            np.vstack(knots_v),
            np.vstack(points),
            np.vstack(anchors),
        )

    def to_torch(self, *, device=None, dtype=None):
        """Return a PyTorch-native evaluator for this fixed patch network."""

        from cardiac_geometry.torch_ops.fastfit_surface import TorchFastFitSurface

        return TorchFastFitSurface.from_fastfit_surface(self, device=device, dtype=dtype)

    def _locate_patches(self, params: np.ndarray) -> np.ndarray:
        indices = np.full(params.shape[0], -1, dtype=np.int64)
        # Choose the first matching patch in deterministic top-left order. Shared
        # boundaries are therefore assigned to exactly one full-multiplicity patch.
        for patch_index, patch in enumerate(self.patches):
            region = _Region(patch.u_min, patch.u_max, patch.v_min, patch.v_max)
            mask = region.contains(params) & (indices < 0)
            indices[mask] = patch_index
        return indices


def bernstein_basis(x: np.ndarray | Iterable[float], degree: int) -> np.ndarray:
    """Evaluate all Bernstein basis polynomials of ``degree`` at ``x``."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("parameter samples must be finite")
    arr = np.clip(arr, 0.0, 1.0)
    out = np.empty(arr.shape + (degree + 1,), dtype=np.float64)
    for i in range(degree + 1):
        out[..., i] = comb(degree, i) * (arr**i) * ((1.0 - arr) ** (degree - i))
    return out


def fit_fastfit_surface(
    points: np.ndarray | Iterable[Iterable[float]],
    *,
    parameters: np.ndarray | Iterable[Iterable[float]] | None = None,
    mask: np.ndarray | Iterable[bool] | None = None,
    options: FastFitOptions | None = None,
) -> FastFitSurface:
    """Fit a piecewise T-spline/Bezier surface using FasTFit's split stage.

    ``points`` can be either an organized array shaped ``(M, N, 3)`` or a flat
    array shaped ``(P, 3)`` with matching ``parameters`` shaped ``(P, 2)``.
    Organized inputs receive a uniform fixed parameterization over ``[0, 1]^2``.
    """

    opts = options or FastFitOptions()
    continuity = _validate_continuity(opts.continuity)
    params, xyz = _prepare_parameterized_points(points, parameters, mask)
    domain = (
        float(np.min(params[:, 0])),
        float(np.max(params[:, 0])),
        float(np.min(params[:, 1])),
        float(np.max(params[:, 1])),
    )
    regions = _initial_regions(domain, opts.initial_splits)
    patches, fitted_regions, skipped_regions, stabilized_regions = _adaptive_fit_regions(
        params, xyz, regions, opts
    )
    patches = tuple(
        sorted(patches, key=lambda p: (p.u_min, p.v_min, p.u_max, p.v_max, p.depth))
    )
    if not patches:
        raise RuntimeError("FasTFit did not produce any valid patches")

    if continuity == "c1":
        patches, boundary_diagnostics = _refit_patches_with_c1_continuity(
            params, xyz, patches, opts
        )
    else:
        boundary_diagnostics = _boundary_diagnostics(patches, opts.degree)

    diagnostics = {
        "patches": len(patches),
        "control_points": len(patches) * opts.control_count(),
        "fitted_regions": fitted_regions,
        "skipped_regions": skipped_regions,
        "stabilized_sparse_regions": stabilized_regions,
        "mean_patch_rmse": float(np.mean([p.rmse for p in patches])),
        "max_patch_error": float(np.max([p.max_error for p in patches])),
        "continuity": continuity,
        **boundary_diagnostics,
    }
    return FastFitSurface(patches, opts.degree, domain, opts, diagnostics)


def fit_bezier_patch(
    parameters: np.ndarray,
    points: np.ndarray,
    region: tuple[float, float, float, float],
    *,
    degree: int = 3,
    smoothing: float = 0.0,
    depth: int = 0,
    anchor_weight: float = 0.0,
    anchor_grid: int | None = None,
    fairing: float = 0.0,
) -> BezierPatch | None:
    """Fit one Bezier patch by least squares as in FasTFit's Eq. (2).

    ``smoothing`` is the original Tikhonov term. ``fairing`` penalizes second
    differences in the Bezier control net, which suppresses off-data oscillation
    without changing the measured data residual. ``anchor_weight`` is an
    optional sparse-support stabilizer: nearest-neighbor pseudo-samples can tie
    unsupported parts of a low-fill patch to nearby data when the adaptive split
    stage cannot refine the region any further.
    """

    if smoothing < 0.0:
        raise ValueError("smoothing must be non-negative")
    if anchor_weight < 0.0:
        raise ValueError("anchor_weight must be non-negative")
    if fairing < 0.0:
        raise ValueError("fairing must be non-negative")

    reg = _Region(*region)
    selected = reg.contains(parameters)
    params = parameters[selected]
    xyz = points[selected]
    control_count = (degree + 1) * (degree + 1)
    regularized = smoothing > 0.0 or anchor_weight > 0.0 or fairing > 0.0
    if xyz.shape[0] == 0:
        return None
    if xyz.shape[0] < control_count and not regularized:
        return None

    design = _bezier_design_matrix(params, reg, degree)
    rhs = xyz
    design_blocks = [design]
    rhs_blocks = [rhs]

    if anchor_weight > 0.0:
        anchor_design, anchor_rhs = _nearest_anchor_constraints(
            params, xyz, reg, degree, anchor_grid or degree + 1
        )
        scale = np.sqrt(anchor_weight)
        design_blocks.append(scale * anchor_design)
        rhs_blocks.append(scale * anchor_rhs)

    if smoothing > 0.0:
        penalty = np.sqrt(smoothing) * np.eye(control_count, dtype=np.float64)
        design_blocks.append(penalty)
        rhs_blocks.append(np.zeros((control_count, 3), dtype=np.float64))

    if fairing > 0.0:
        fairing_matrix = _bezier_fairing_matrix(degree)
        if fairing_matrix.shape[0] > 0:
            design_blocks.append(np.sqrt(fairing) * fairing_matrix)
            rhs_blocks.append(np.zeros((fairing_matrix.shape[0], 3), dtype=np.float64))

    design_solve = np.vstack(design_blocks)
    rhs_solve = np.vstack(rhs_blocks)

    try:
        control_points, _, rank, _ = np.linalg.lstsq(design_solve, rhs_solve, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < min(control_count, design_solve.shape[0]) and not regularized:
        return None

    residual = design @ control_points - xyz
    distances = np.linalg.norm(residual, axis=1)
    rmse = float(np.sqrt(np.mean(distances * distances))) if distances.size else 0.0
    max_error = float(np.max(distances)) if distances.size else 0.0
    return BezierPatch(
        reg.u_min,
        reg.u_max,
        reg.v_min,
        reg.v_max,
        degree,
        control_points,
        rmse,
        max_error,
        int(xyz.shape[0]),
        int(rank),
        depth,
    )


def _prepare_parameterized_points(
    points: np.ndarray | Iterable[Iterable[float]],
    parameters: np.ndarray | Iterable[Iterable[float]] | None,
    mask: np.ndarray | Iterable[bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] == 3:
        m, n, _ = arr.shape
        u = np.linspace(0.0, 1.0, m)
        v = np.linspace(0.0, 1.0, n)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        params = np.column_stack([uu.reshape(-1), vv.reshape(-1)])
        xyz = arr.reshape(-1, 3)
        valid = np.all(np.isfinite(xyz), axis=1)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool).reshape(-1)
        return params[valid], xyz[valid]

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must have shape (M, N, 3) or (P, 3)")
    if parameters is None:
        raise ValueError("flat points require parameters with shape (P, 2)")
    params = np.asarray(parameters, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != 2 or params.shape[0] != arr.shape[0]:
        raise ValueError("parameters must have shape (P, 2)")
    valid = np.all(np.isfinite(arr), axis=1) & np.all(np.isfinite(params), axis=1)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    return params[valid], arr[valid]


def _normalize_parameter(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    span = max_value - min_value
    if span <= 0.0:
        raise ValueError("patch parameter domain has zero width")
    return (np.asarray(values, dtype=np.float64) - min_value) / span


def _bezier_design_matrix(params: np.ndarray, region: _Region, degree: int) -> np.ndarray:
    control_count = (degree + 1) * (degree + 1)
    if params.shape[0] == 0:
        return np.empty((0, control_count), dtype=np.float64)
    u_local = _normalize_parameter(params[:, 0], region.u_min, region.u_max)
    v_local = _normalize_parameter(params[:, 1], region.v_min, region.v_max)
    bu = bernstein_basis(u_local, degree)
    bv = bernstein_basis(v_local, degree)
    return (bu[:, :, None] * bv[:, None, :]).reshape(params.shape[0], control_count)


def _nearest_anchor_constraints(
    params: np.ndarray,
    points: np.ndarray,
    region: _Region,
    degree: int,
    grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pseudo-samples tying sparse patch interiors to nearby data."""

    if grid <= 0:
        raise ValueError("anchor_grid must be positive")
    if params.shape[0] == 0:
        raise ValueError("anchor constraints require at least one point")

    if grid == 1:
        u_values = np.array([0.5 * (region.u_min + region.u_max)], dtype=np.float64)
        v_values = np.array([0.5 * (region.v_min + region.v_max)], dtype=np.float64)
    else:
        u_values = np.linspace(region.u_min, region.u_max, grid)
        v_values = np.linspace(region.v_min, region.v_max, grid)
    uu, vv = np.meshgrid(u_values, v_values, indexing="ij")
    anchors = np.column_stack([uu.reshape(-1), vv.reshape(-1)])

    local_params = np.column_stack(
        [
            _normalize_parameter(params[:, 0], region.u_min, region.u_max),
            _normalize_parameter(params[:, 1], region.v_min, region.v_max),
        ]
    )
    local_anchors = np.column_stack(
        [
            _normalize_parameter(anchors[:, 0], region.u_min, region.u_max),
            _normalize_parameter(anchors[:, 1], region.v_min, region.v_max),
        ]
    )
    distances2 = np.sum(
        (local_anchors[:, None, :] - local_params[None, :, :]) ** 2, axis=2
    )
    nearest = np.argmin(distances2, axis=1)
    return _bezier_design_matrix(anchors, region, degree), points[nearest]


def _bezier_fairing_matrix(degree: int) -> np.ndarray:
    """Second-difference penalty matrix for a tensor-product Bezier net."""

    side = degree + 1
    control_count = side * side
    if degree < 2:
        return np.empty((0, control_count), dtype=np.float64)

    rows = []
    for i in range(side - 2):
        for j in range(side):
            row = np.zeros(control_count, dtype=np.float64)
            row[i * side + j] = 1.0
            row[(i + 1) * side + j] = -2.0
            row[(i + 2) * side + j] = 1.0
            rows.append(row)
    for i in range(side):
        for j in range(side - 2):
            row = np.zeros(control_count, dtype=np.float64)
            row[i * side + j] = 1.0
            row[i * side + j + 1] = -2.0
            row[i * side + j + 2] = 1.0
            rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _validate_continuity(value: str) -> str:
    continuity = value.lower()
    if continuity not in {"none", "c1"}:
        raise ValueError("continuity must be either 'none' or 'c1'")
    return continuity


def _refit_patches_with_c1_continuity(
    params: np.ndarray,
    points: np.ndarray,
    patches: tuple[BezierPatch, ...],
    opts: FastFitOptions,
) -> tuple[tuple[BezierPatch, ...], dict[str, float | int]]:
    if len(patches) <= 1:
        return patches, _boundary_diagnostics(patches, opts.degree)

    control_count = opts.control_count()
    total_control_points = len(patches) * control_count
    design_blocks: list[np.ndarray] = []
    rhs_blocks: list[np.ndarray] = []
    fairing_matrix = _bezier_fairing_matrix(opts.degree)

    for patch_index, patch in enumerate(patches):
        region = _Region(patch.u_min, patch.u_max, patch.v_min, patch.v_max)
        selected = region.contains(params)
        patch_params = params[selected]
        patch_points = points[selected]

        design = _bezier_design_matrix(patch_params, region, opts.degree)
        design_blocks.append(
            _embed_patch_design(
                design, patch_index, control_count, total_control_points
            )
        )
        rhs_blocks.append(patch_points)

        sparse_region = (
            opts.min_region_fill_ratio > 0.0
            and _region_fill_ratio(params, region, opts.support_grid)
            < opts.min_region_fill_ratio
        )
        if sparse_region and opts.sparse_anchor_weight > 0.0 and patch_params.shape[0] > 0:
            anchor_design, anchor_rhs = _nearest_anchor_constraints(
                patch_params,
                patch_points,
                region,
                opts.degree,
                opts.degree + 1,
            )
            scale = np.sqrt(opts.sparse_anchor_weight)
            design_blocks.append(
                _embed_patch_design(
                    scale * anchor_design,
                    patch_index,
                    control_count,
                    total_control_points,
                )
            )
            rhs_blocks.append(scale * anchor_rhs)

        if opts.smoothing > 0.0:
            penalty = np.sqrt(opts.smoothing) * np.eye(control_count, dtype=np.float64)
            design_blocks.append(
                _embed_patch_design(
                    penalty, patch_index, control_count, total_control_points
                )
            )
            rhs_blocks.append(np.zeros((control_count, 3), dtype=np.float64))

        fairing = opts.fairing_weight()
        if fairing > 0.0 and fairing_matrix.shape[0] > 0:
            design_blocks.append(
                _embed_patch_design(
                    np.sqrt(fairing) * fairing_matrix,
                    patch_index,
                    control_count,
                    total_control_points,
                )
            )
            rhs_blocks.append(np.zeros((fairing_matrix.shape[0], 3), dtype=np.float64))

    design_solve = np.vstack(design_blocks)
    rhs_solve = np.vstack(rhs_blocks)
    constraints = _c1_continuity_matrix(patches, opts.degree)
    control_points, rank = _solve_homogeneous_constrained_lstsq(
        design_solve, rhs_solve, constraints
    )

    refitted = []
    for patch_index, patch in enumerate(patches):
        start = patch_index * control_count
        stop = start + control_count
        controls = control_points[start:stop]
        region = _Region(patch.u_min, patch.u_max, patch.v_min, patch.v_max)
        selected = region.contains(params)
        design = _bezier_design_matrix(params[selected], region, opts.degree)
        residual = design @ controls - points[selected]
        distances = np.linalg.norm(residual, axis=1)
        rmse = float(np.sqrt(np.mean(distances * distances))) if distances.size else 0.0
        max_error = float(np.max(distances)) if distances.size else 0.0
        refitted.append(
            BezierPatch(
                patch.u_min,
                patch.u_max,
                patch.v_min,
                patch.v_max,
                patch.degree,
                controls,
                rmse,
                max_error,
                int(np.count_nonzero(selected)),
                int(rank),
                patch.depth,
            )
        )

    diagnostics = _boundary_diagnostics(
        tuple(refitted), opts.degree, boundary_constraints=constraints.shape[0]
    )
    return tuple(refitted), diagnostics


def _embed_patch_design(
    design: np.ndarray,
    patch_index: int,
    control_count: int,
    total_control_points: int,
) -> np.ndarray:
    out = np.zeros((design.shape[0], total_control_points), dtype=np.float64)
    start = patch_index * control_count
    out[:, start : start + control_count] = design
    return out


def _solve_homogeneous_constrained_lstsq(
    design: np.ndarray,
    rhs: np.ndarray,
    constraints: np.ndarray,
) -> tuple[np.ndarray, int]:
    if constraints.shape[0] == 0:
        solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
        return solution, int(rank)

    _, singular_values, vt = np.linalg.svd(constraints, full_matrices=True)
    if singular_values.size:
        tolerance = (
            np.finfo(np.float64).eps
            * max(constraints.shape)
            * float(singular_values[0])
        )
        constraint_rank = int(np.count_nonzero(singular_values > tolerance))
    else:
        constraint_rank = 0
    nullspace = vt[constraint_rank:].T
    if nullspace.shape[1] == 0:
        return np.zeros((design.shape[1], rhs.shape[1]), dtype=np.float64), 0

    reduced_design = design @ nullspace
    reduced_solution, _, rank, _ = np.linalg.lstsq(reduced_design, rhs, rcond=None)
    return nullspace @ reduced_solution, int(rank)


def _c1_continuity_matrix(
    patches: tuple[BezierPatch, ...],
    degree: int,
) -> np.ndarray:
    control_count = (degree + 1) * (degree + 1)
    total_control_points = len(patches) * control_count
    rows = []
    for boundary in _shared_boundaries(patches):
        for sample in np.linspace(boundary.start, boundary.end, degree + 1):
            c0_row = np.zeros(total_control_points, dtype=np.float64)
            c1_row = np.zeros(total_control_points, dtype=np.float64)
            lower = patches[boundary.lower_index]
            upper = patches[boundary.upper_index]
            lower_start = boundary.lower_index * control_count
            upper_start = boundary.upper_index * control_count

            if boundary.axis == "u":
                lower_position = _bezier_position_row(
                    lower.u_max, sample, lower, degree
                )
                upper_position = _bezier_position_row(
                    upper.u_min, sample, upper, degree
                )
                lower_derivative = _bezier_du_row(
                    lower.u_max, sample, lower, degree
                )
                upper_derivative = _bezier_du_row(
                    upper.u_min, sample, upper, degree
                )
            else:
                lower_position = _bezier_position_row(
                    sample, lower.v_max, lower, degree
                )
                upper_position = _bezier_position_row(
                    sample, upper.v_min, upper, degree
                )
                lower_derivative = _bezier_dv_row(
                    sample, lower.v_max, lower, degree
                )
                upper_derivative = _bezier_dv_row(
                    sample, upper.v_min, upper, degree
                )

            c0_row[lower_start : lower_start + control_count] = lower_position
            c0_row[upper_start : upper_start + control_count] = -upper_position
            c1_row[lower_start : lower_start + control_count] = lower_derivative
            c1_row[upper_start : upper_start + control_count] = -upper_derivative
            rows.append(c0_row)
            if degree > 0:
                rows.append(c1_row)

    if not rows:
        return np.empty((0, total_control_points), dtype=np.float64)
    return np.vstack(rows)


def _bezier_position_row(
    u: float,
    v: float,
    patch: BezierPatch,
    degree: int,
) -> np.ndarray:
    side = degree + 1
    u_local = _normalize_parameter(np.asarray([u]), patch.u_min, patch.u_max)[0]
    v_local = _normalize_parameter(np.asarray([v]), patch.v_min, patch.v_max)[0]
    u_local = float(np.clip(u_local, 0.0, 1.0))
    v_local = float(np.clip(v_local, 0.0, 1.0))
    bu = bernstein_basis(np.asarray([u_local]), degree)[0]
    bv = bernstein_basis(np.asarray([v_local]), degree)[0]
    return (bu[:, None] * bv[None, :]).reshape(side * side)


def _bezier_du_row(
    u: float,
    v: float,
    patch: BezierPatch,
    degree: int,
) -> np.ndarray:
    side = degree + 1
    out = np.zeros(side * side, dtype=np.float64)
    if degree == 0:
        return out
    u_local = float(np.clip(_normalize_parameter(np.asarray([u]), patch.u_min, patch.u_max)[0], 0.0, 1.0))
    v_local = float(np.clip(_normalize_parameter(np.asarray([v]), patch.v_min, patch.v_max)[0], 0.0, 1.0))
    bu = bernstein_basis(np.asarray([u_local]), degree - 1)[0]
    bv = bernstein_basis(np.asarray([v_local]), degree)[0]
    scale = degree / (patch.u_max - patch.u_min)
    for i in range(degree):
        for j in range(side):
            value = scale * bu[i] * bv[j]
            out[(i + 1) * side + j] += value
            out[i * side + j] -= value
    return out


def _bezier_dv_row(
    u: float,
    v: float,
    patch: BezierPatch,
    degree: int,
) -> np.ndarray:
    side = degree + 1
    out = np.zeros(side * side, dtype=np.float64)
    if degree == 0:
        return out
    u_local = float(np.clip(_normalize_parameter(np.asarray([u]), patch.u_min, patch.u_max)[0], 0.0, 1.0))
    v_local = float(np.clip(_normalize_parameter(np.asarray([v]), patch.v_min, patch.v_max)[0], 0.0, 1.0))
    bu = bernstein_basis(np.asarray([u_local]), degree)[0]
    bv = bernstein_basis(np.asarray([v_local]), degree - 1)[0]
    scale = degree / (patch.v_max - patch.v_min)
    for i in range(side):
        for j in range(degree):
            value = scale * bu[i] * bv[j]
            out[i * side + j + 1] += value
            out[i * side + j] -= value
    return out


def _shared_boundaries(patches: tuple[BezierPatch, ...]) -> list[_SharedBoundary]:
    tolerance = _patch_coordinate_tolerance(patches)
    boundaries = []
    for first_index, first in enumerate(patches):
        for second_index in range(first_index + 1, len(patches)):
            second = patches[second_index]
            if _same_coordinate(first.u_max, second.u_min, tolerance):
                start = max(first.v_min, second.v_min)
                end = min(first.v_max, second.v_max)
                if end - start > tolerance:
                    boundaries.append(
                        _SharedBoundary(first_index, second_index, "u", start, end)
                    )
            elif _same_coordinate(second.u_max, first.u_min, tolerance):
                start = max(first.v_min, second.v_min)
                end = min(first.v_max, second.v_max)
                if end - start > tolerance:
                    boundaries.append(
                        _SharedBoundary(second_index, first_index, "u", start, end)
                    )

            if _same_coordinate(first.v_max, second.v_min, tolerance):
                start = max(first.u_min, second.u_min)
                end = min(first.u_max, second.u_max)
                if end - start > tolerance:
                    boundaries.append(
                        _SharedBoundary(first_index, second_index, "v", start, end)
                    )
            elif _same_coordinate(second.v_max, first.v_min, tolerance):
                start = max(first.u_min, second.u_min)
                end = min(first.u_max, second.u_max)
                if end - start > tolerance:
                    boundaries.append(
                        _SharedBoundary(second_index, first_index, "v", start, end)
                    )
    return boundaries


def _patch_coordinate_tolerance(patches: tuple[BezierPatch, ...]) -> float:
    if not patches:
        return 1e-12
    values = []
    for patch in patches:
        values.extend([patch.u_min, patch.u_max, patch.v_min, patch.v_max])
    span = max(values) - min(values)
    return max(1e-12, 1e-10 * span)


def _same_coordinate(first: float, second: float, tolerance: float) -> bool:
    return abs(first - second) <= tolerance


def _boundary_diagnostics(
    patches: tuple[BezierPatch, ...],
    degree: int,
    *,
    boundary_constraints: int = 0,
) -> dict[str, float | int]:
    max_c0_gap = 0.0
    max_c1_gap = 0.0
    for boundary in _shared_boundaries(patches):
        lower = patches[boundary.lower_index]
        upper = patches[boundary.upper_index]
        for sample in np.linspace(boundary.start, boundary.end, degree + 1):
            if boundary.axis == "u":
                lower_position = lower.evaluate(
                    np.asarray([lower.u_max]), np.asarray([sample])
                )[0]
                upper_position = upper.evaluate(
                    np.asarray([upper.u_min]), np.asarray([sample])
                )[0]
                lower_derivative = _evaluate_patch_du(
                    lower, np.asarray([lower.u_max]), np.asarray([sample])
                )[0]
                upper_derivative = _evaluate_patch_du(
                    upper, np.asarray([upper.u_min]), np.asarray([sample])
                )[0]
            else:
                lower_position = lower.evaluate(
                    np.asarray([sample]), np.asarray([lower.v_max])
                )[0]
                upper_position = upper.evaluate(
                    np.asarray([sample]), np.asarray([upper.v_min])
                )[0]
                lower_derivative = _evaluate_patch_dv(
                    lower, np.asarray([sample]), np.asarray([lower.v_max])
                )[0]
                upper_derivative = _evaluate_patch_dv(
                    upper, np.asarray([sample]), np.asarray([upper.v_min])
                )[0]
            max_c0_gap = max(
                max_c0_gap, float(np.linalg.norm(lower_position - upper_position))
            )
            max_c1_gap = max(
                max_c1_gap, float(np.linalg.norm(lower_derivative - upper_derivative))
            )
    return {
        "boundary_constraints": int(boundary_constraints),
        "max_c0_gap": max_c0_gap,
        "max_c1_gap": max_c1_gap,
    }


def _evaluate_patch_du(
    patch: BezierPatch,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    if patch.degree == 0:
        return np.zeros(np.asarray(u).shape + (3,), dtype=np.float64)
    u_local = _normalize_parameter(u, patch.u_min, patch.u_max)
    v_local = _normalize_parameter(v, patch.v_min, patch.v_max)
    bu = bernstein_basis(u_local, patch.degree - 1)
    bv = bernstein_basis(v_local, patch.degree)
    controls = patch.control_points.reshape(patch.degree + 1, patch.degree + 1, 3)
    differences = controls[1:, :, :] - controls[:-1, :, :]
    scale = patch.degree / (patch.u_max - patch.u_min)
    return scale * np.einsum("...i,...j,ijc->...c", bu, bv, differences)


def _evaluate_patch_dv(
    patch: BezierPatch,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    if patch.degree == 0:
        return np.zeros(np.asarray(u).shape + (3,), dtype=np.float64)
    u_local = _normalize_parameter(u, patch.u_min, patch.u_max)
    v_local = _normalize_parameter(v, patch.v_min, patch.v_max)
    bu = bernstein_basis(u_local, patch.degree)
    bv = bernstein_basis(v_local, patch.degree - 1)
    controls = patch.control_points.reshape(patch.degree + 1, patch.degree + 1, 3)
    differences = controls[:, 1:, :] - controls[:, :-1, :]
    scale = patch.degree / (patch.v_max - patch.v_min)
    return scale * np.einsum("...i,...j,ijc->...c", bu, bv, differences)


def _initial_regions(
    domain: tuple[float, float, float, float],
    splits: tuple[int, int],
) -> list[_Region]:
    if splits[0] <= 0 or splits[1] <= 0:
        raise ValueError("initial_splits must be positive")
    u_min, u_max, v_min, v_max = domain
    u_edges = np.linspace(u_min, u_max, splits[0] + 1)
    v_edges = np.linspace(v_min, v_max, splits[1] + 1)
    regions = []
    for i in range(splits[0]):
        for j in range(splits[1]):
            regions.append(_Region(u_edges[i], u_edges[i + 1], v_edges[j], v_edges[j + 1]))
    return regions


def _adaptive_fit_regions(
    params: np.ndarray,
    points: np.ndarray,
    initial_regions: list[_Region],
    opts: FastFitOptions,
) -> tuple[list[BezierPatch], int, int, int]:
    pending = list(initial_regions)
    patches: list[BezierPatch] = []
    fitted_regions = 0
    skipped_regions = 0
    stabilized_regions = 0

    while pending:
        pending = sorted(pending, key=lambda r: (r.depth, r.u_min, r.v_min, r.u_max, r.v_max))

        def fit_region(region: _Region) -> tuple[_Region, BezierPatch | None, int]:
            selected_count = int(np.count_nonzero(region.contains(params)))
            if selected_count < opts.effective_min_points() and opts.smoothing <= 0.0:
                return region, None, selected_count
            patch = fit_bezier_patch(
                params,
                points,
                (region.u_min, region.u_max, region.v_min, region.v_max),
                degree=opts.degree,
                smoothing=opts.smoothing,
                depth=region.depth,
                fairing=opts.fairing_weight(),
            )
            return region, patch, selected_count

        if opts.workers is not None and opts.workers > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=opts.workers) as pool:
                results = list(pool.map(fit_region, pending))
        else:
            results = [fit_region(region) for region in pending]

        pending = []
        for region, patch, selected_count in results:
            if patch is None:
                skipped_regions += 1
                continue
            fitted_regions += 1
            sparse_region = (
                opts.min_region_fill_ratio > 0.0
                and _region_fill_ratio(params, region, opts.support_grid)
                < opts.min_region_fill_ratio
            )
            wants_split = patch.max_error > opts.max_error or sparse_region
            can_split = (
                wants_split
                and region.depth < opts.max_depth
                and _can_split(region, params, opts)
            )
            if can_split:
                pending.extend(region.split())
                continue

            if sparse_region and opts.sparse_anchor_weight > 0.0:
                stable_patch = fit_bezier_patch(
                    params,
                    points,
                    (region.u_min, region.u_max, region.v_min, region.v_max),
                    degree=opts.degree,
                    smoothing=opts.smoothing,
                    depth=region.depth,
                    anchor_weight=opts.sparse_anchor_weight,
                    anchor_grid=opts.degree + 1,
                    fairing=opts.fairing_weight(),
                )
                if stable_patch is not None:
                    patch = stable_patch
                    stabilized_regions += 1

            if opts.keep_unsplittable or not wants_split:
                patches.append(patch)
            else:
                skipped_regions += 1
    return patches, fitted_regions, skipped_regions, stabilized_regions


def _can_split(region: _Region, params: np.ndarray, opts: FastFitOptions) -> bool:
    if region.u_max <= region.u_min or region.v_max <= region.v_min:
        return False
    minimum = opts.effective_min_points()
    return all(np.count_nonzero(child.contains(params)) >= minimum for child in region.split())


def _region_fill_ratio(params: np.ndarray, region: _Region, grid: int) -> float:
    if grid <= 0:
        raise ValueError("support_grid must be positive")
    selected = region.contains(params)
    local = params[selected]
    if local.shape[0] == 0:
        return 0.0
    u = np.clip(_normalize_parameter(local[:, 0], region.u_min, region.u_max), 0.0, 1.0)
    v = np.clip(_normalize_parameter(local[:, 1], region.v_min, region.v_max), 0.0, 1.0)
    cols = np.clip((u * grid).astype(np.int64), 0, grid - 1)
    rows = np.clip((v * grid).astype(np.int64), 0, grid - 1)
    occupied = np.zeros((grid, grid), dtype=bool)
    occupied[rows, cols] = True
    return float(np.count_nonzero(occupied) / occupied.size)
