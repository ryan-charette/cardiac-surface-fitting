"""Load the original T-spline sample data into a cleaned package structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cardiac_geometry.geometry.surface import TMeshSurface


SAMPLE_PARAMETER_RANGES = {
    "tube": ((0.1, 0.9), (0.1, 0.9)),
    "bifurcation": ((0.25, 0.75), (0.1, 0.9)),
}


def _load_csv_2d(path: Path, *, dtype=float) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=dtype)
    return np.atleast_2d(arr)


def _parse_knot_distance_blocks(path: Path) -> tuple[list[list[float]], list[list[float]]]:
    blocks: list[list[list[float]]] = [[]]
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            if blocks[-1]:
                blocks.append([])
            continue
        values = [float(item) for item in line.split(",") if item.strip()]
        blocks[-1].append(values)

    blocks = [block for block in blocks if block]
    if len(blocks) != 2:
        raise ValueError(f"{path} must contain u and v knot-distance blocks")
    return blocks[0], blocks[1]


def _cumulative_positions(distance_rows: list[list[float]], *, name: str) -> np.ndarray:
    if not distance_rows:
        raise ValueError(f"{name} distance block is empty")
    width = max(len(row) for row in distance_rows)
    out = np.full((len(distance_rows), width), np.nan, dtype=np.float64)
    for row_index, row in enumerate(distance_rows):
        if len(row) < 2:
            raise ValueError(f"{name} row {row_index} must contain at least two entries")
        distances = np.asarray(row, dtype=np.float64)
        if not np.all(np.isfinite(distances)):
            raise ValueError(f"{name} distances must be finite")
        if np.any(distances < 0.0):
            raise ValueError(f"{name} distances must be non-negative")
        out[row_index, : len(row)] = np.cumsum(distances)

    scale = np.nanmax(out)
    if scale <= 0.0:
        raise ValueError(f"{name} distances do not span a nonzero interval")
    return out / scale


def _point_lookup(points: np.ndarray) -> dict[int, np.ndarray]:
    points = points[np.argsort(points[:, 1])]
    pi = points[:, 1].astype(int)
    expected = np.arange(points.shape[0])
    if not np.array_equal(pi, expected):
        raise ValueError("point ids in points.csv must be contiguous from zero")
    return {int(row[1]): row[2:5].astype(np.float64) for row in points}


def load_tmesh_case(
    case_dir: str | Path,
    *,
    name: str | None = None,
    degree: int = 2,
    parameter_range_u: tuple[float, float] | None = None,
    parameter_range_v: tuple[float, float] | None = None,
) -> TMeshSurface:
    """Load one sample T-mesh case from CSV/knot-distance files."""

    case_path = Path(case_dir)
    points = _load_csv_2d(case_path / "points.csv", dtype=float)
    point_map = _point_lookup(points)
    entries = _load_csv_2d(case_path / "points_indices.csv", dtype=int)
    entries = entries[np.argsort(entries[:, 0])]

    distance_u, distance_v = _parse_knot_distance_blocks(case_path / "knot_distances.txt")
    global_u = _cumulative_positions(distance_u, name="u")
    global_v = _cumulative_positions(distance_v, name="v")

    knot_width = degree + 2
    if global_v.shape[0] != 1:
        raise ValueError("this loader currently expects one shared v knot-distance row")

    knots_u: list[np.ndarray] = []
    knots_v: list[np.ndarray] = []
    control_points: list[np.ndarray] = []
    control_point_ids: list[int] = []

    for _, row, col, point_id in entries:
        row_i = int(row)
        col_i = int(col)
        point_i = int(point_id)
        if point_i not in point_map:
            raise ValueError(f"point id {point_i} is referenced but missing from points.csv")
        if col_i >= global_u.shape[0]:
            raise ValueError(f"u knot row {col_i} is missing")
        local_u = global_u[col_i, row_i : row_i + knot_width]
        local_v = global_v[0, col_i : col_i + knot_width]
        if local_u.shape[0] != knot_width or np.any(~np.isfinite(local_u)):
            raise ValueError(f"not enough u knots for row={row_i}, col={col_i}")
        if local_v.shape[0] != knot_width or np.any(~np.isfinite(local_v)):
            raise ValueError(f"not enough v knots for row={row_i}, col={col_i}")
        knots_u.append(local_u)
        knots_v.append(local_v)
        control_points.append(point_map[point_i])
        control_point_ids.append(point_i)

    case_name = name or case_path.name
    default_ranges = SAMPLE_PARAMETER_RANGES.get(case_name)
    if default_ranges is not None:
        default_u, default_v = default_ranges
    else:
        default_u = (float(np.nanmin(global_u)), float(np.nanmax(global_u)))
        default_v = (float(np.nanmin(global_v)), float(np.nanmax(global_v)))

    return TMeshSurface(
        name=case_name,
        degree=degree,
        knots_u=np.asarray(knots_u, dtype=np.float64),
        knots_v=np.asarray(knots_v, dtype=np.float64),
        control_points=np.asarray(control_points, dtype=np.float64),
        weights=np.ones(len(control_points), dtype=np.float64),
        control_point_indices=np.asarray(control_point_ids, dtype=np.int64),
        parameter_range_u=parameter_range_u or default_u,
        parameter_range_v=parameter_range_v or default_v,
    )


def load_sample_case(
    name: str,
    *,
    data_root: str | Path | None = None,
    degree: int = 2,
) -> TMeshSurface:
    """Load one bundled sample case: ``tube`` or ``bifurcation``."""

    if data_root is None:
        data_root = Path(__file__).resolve().parents[2] / "data" / "sample"
    root = Path(data_root)
    if name not in SAMPLE_PARAMETER_RANGES:
        raise ValueError(f"unknown sample case {name!r}")
    return load_tmesh_case(
        root / name,
        name=name,
        degree=degree,
        parameter_range_u=SAMPLE_PARAMETER_RANGES[name][0],
        parameter_range_v=SAMPLE_PARAMETER_RANGES[name][1],
    )

