"""Occupancy and boundary detection for parameterized point clouds."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterOccupancy:
    """Rasterized support mask for a 2D parameterized point cloud."""

    resolution: int
    occupied: np.ndarray
    supported: np.ndarray
    exterior: np.ndarray
    holes: np.ndarray
    boundary: np.ndarray

    def cell_indices(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        params = _validate_parameters(parameters)
        cols = np.clip((params[:, 0] * self.resolution).astype(np.int64), 0, self.resolution - 1)
        rows = np.clip((params[:, 1] * self.resolution).astype(np.int64), 0, self.resolution - 1)
        return rows, cols

    def classify_supported(self, parameters: np.ndarray) -> np.ndarray:
        rows, cols = self.cell_indices(parameters)
        return self.supported[rows, cols]

    def classify_boundary(self, parameters: np.ndarray) -> np.ndarray:
        rows, cols = self.cell_indices(parameters)
        return self.boundary[rows, cols]


def build_parameter_occupancy(
    parameters: np.ndarray,
    *,
    resolution: int = 160,
    dilation: int = 2,
) -> ParameterOccupancy:
    """Build a hole-aware occupancy mask in parameter space.

    The mask is deliberately simple and dependency-free. It approximates an
    alpha-shape support region by rasterizing samples, dilating them by a small
    grid radius, and flood-filling empty cells from the outside. Empty cells not
    reached by the flood fill are treated as holes.
    """

    if resolution < 8:
        raise ValueError("resolution must be at least 8")
    if dilation < 0:
        raise ValueError("dilation must be non-negative")

    params = _validate_parameters(parameters)
    occupied = np.zeros((resolution, resolution), dtype=bool)
    cols = np.clip((params[:, 0] * resolution).astype(np.int64), 0, resolution - 1)
    rows = np.clip((params[:, 1] * resolution).astype(np.int64), 0, resolution - 1)
    occupied[rows, cols] = True

    supported = _dilate(occupied, dilation)
    exterior = _flood_exterior(~supported)
    holes = (~supported) & (~exterior)
    boundary_edge = supported & _neighbor_any(~supported)
    boundary = _dilate(boundary_edge, max(1, dilation)) & supported
    return ParameterOccupancy(resolution, occupied, supported, exterior, holes, boundary)


def _validate_parameters(parameters: np.ndarray) -> np.ndarray:
    params = np.asarray(parameters, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("parameters must have shape (N, 2)")
    if not np.all(np.isfinite(params)):
        raise ValueError("parameters must contain only finite values")
    if np.any(params < 0.0) or np.any(params > 1.0):
        raise ValueError("parameters must be normalized to [0, 1]")
    return params


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(mask.shape[0], mask.shape[0] - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(mask.shape[1], mask.shape[1] - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(mask.shape[0], mask.shape[0] + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(mask.shape[1], mask.shape[1] + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _flood_exterior(empty: np.ndarray) -> np.ndarray:
    exterior = np.zeros_like(empty, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    rows, cols = empty.shape

    def add(row: int, col: int) -> None:
        if empty[row, col] and not exterior[row, col]:
            exterior[row, col] = True
            queue.append((row, col))

    for row in range(rows):
        add(row, 0)
        add(row, cols - 1)
    for col in range(cols):
        add(0, col)
        add(rows - 1, col)

    while queue:
        row, col = queue.popleft()
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if 0 <= next_row < rows and 0 <= next_col < cols:
                add(next_row, next_col)
    return exterior


def _neighbor_any(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(rows, rows - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(cols, cols - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(rows, rows + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(cols, cols + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out
