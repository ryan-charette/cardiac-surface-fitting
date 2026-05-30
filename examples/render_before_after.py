"""Render a README before/after visualization for LA FasTFit surface fitting.

The script intentionally uses only NumPy and the project package so it can run
in a minimal development environment without Matplotlib or Pillow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cardiac_geometry.fastfit import FastFitOptions, fit_fastfit_surface
from cardiac_geometry.geometry.occupancy import ParameterOccupancy, build_parameter_occupancy
from cardiac_geometry.geometry.parameterization import spherical_parameterize


LA_TEXT_FILES = {
    "ED": "LA-vertices-ED-Centered-Trimmed.txt",
    "ES": "LA-vertices-ES-Centered-Trimmed.txt",
}


def _load_la_points(data_dir: Path, label: str) -> np.ndarray:
    path = data_dir / LA_TEXT_FILES[label]
    points = np.loadtxt(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path} did not contain an Nx3 point cloud")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{path} contains non-finite coordinates")
    return points


def _patch_tiles(model, samples_per_axis: int) -> list[tuple[np.ndarray, np.ndarray]]:
    tiles = []
    for patch in model.patches:
        u = np.linspace(patch.u_min, patch.u_max, samples_per_axis)
        v = np.linspace(patch.v_min, patch.v_max, samples_per_axis)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        uv = np.stack([uu, vv], axis=-1)
        xyz = patch.evaluate(uu.reshape(-1), vv.reshape(-1)).reshape(uu.shape + (3,))
        tiles.append((uv, xyz))
    return tiles


def _smooth_tiles(
    tiles: list[tuple[np.ndarray, np.ndarray]],
    occupancy: ParameterOccupancy,
    *,
    iterations: int,
    strength: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply display-only masked Laplacian smoothing to rendered patch grids."""

    if iterations <= 0 or strength <= 0.0:
        return tiles
    if strength > 1.0:
        raise ValueError("surface smoothing strength must be <= 1")

    smoothed = []
    for uv_tile, xyz_tile in tiles:
        cell_mask = _supported_cells(uv_tile, occupancy)
        vertex_mask = _visible_vertices(cell_mask)
        xyz = xyz_tile.copy()
        for _ in range(iterations):
            xyz = _smooth_grid_once(xyz, vertex_mask, strength)
        smoothed.append((uv_tile, xyz))
    return smoothed


def _smooth_grid_once(xyz: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    out = xyz.copy()
    rows, cols = mask.shape
    for row in range(rows):
        for col in range(cols):
            if not bool(mask[row, col]):
                continue
            neighbors = []
            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if 0 <= next_row < rows and 0 <= next_col < cols and bool(mask[next_row, next_col]):
                    neighbors.append(xyz[next_row, next_col])
            if neighbors:
                average = np.mean(np.asarray(neighbors), axis=0)
                out[row, col] = (1.0 - strength) * xyz[row, col] + strength * average
    return out


def _project(points: np.ndarray, *, azimuth_deg: float = -42.0, elevation_deg: float = 24.0):
    """Orthographically project 3D points to 2D plus a depth coordinate."""

    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    horizontal = np.sin(azimuth) * x + np.cos(azimuth) * y
    screen_x = np.cos(azimuth) * x - np.sin(azimuth) * y
    screen_y = np.sin(elevation) * horizontal + np.cos(elevation) * z
    depth = np.cos(elevation) * horizontal - np.sin(elevation) * z
    return np.column_stack([screen_x, screen_y, depth])


def _panel_mapper(projected: np.ndarray, panel: tuple[float, float, float, float]):
    left, top, width, height = panel
    pad_x = 48.0
    pad_y = 66.0
    xy = projected[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    scale = min((width - 2 * pad_x) / span[0], (height - 2 * pad_y) / span[1])

    def map_points(values: np.ndarray) -> np.ndarray:
        out = np.empty((values.shape[0], 2), dtype=np.float64)
        out[:, 0] = left + 0.5 * width + (values[:, 0] - 0.5 * (mins[0] + maxs[0])) * scale
        out[:, 1] = top + 0.5 * height - (values[:, 1] - 0.5 * (mins[1] + maxs[1])) * scale
        return out

    return map_points


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _shade(base: str, factor: float) -> str:
    r, g, b = _rgb(base)
    factor = float(np.clip(factor, 0.72, 1.16))
    return f"#{min(255, int(r * factor)):02x}{min(255, int(g * factor)):02x}{min(255, int(b * factor)):02x}"


def _surface_polygons(
    projected: np.ndarray,
    screen: np.ndarray,
    rows: int,
    cols: int,
    *,
    cell_mask: np.ndarray | None = None,
) -> list[tuple[float, str]]:
    depth_min = float(projected[:, 2].min())
    depth_span = float(max(projected[:, 2].max() - depth_min, 1e-9))
    polygons = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            if cell_mask is not None and not bool(cell_mask[i, j]):
                continue
            indices = np.array(
                [
                    i * cols + j,
                    (i + 1) * cols + j,
                    (i + 1) * cols + j + 1,
                    i * cols + j + 1,
                ],
                dtype=np.int64,
            )
            pts = screen[indices]
            depth = float(projected[indices, 2].mean())
            normalized_depth = (depth - depth_min) / depth_span
            fill = _shade("#2f8f83", 0.78 + 0.32 * normalized_depth)
            point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            polygons.append(
                (
                    depth,
                    f'<polygon points="{point_string}" fill="{fill}" '
                    'stroke="#155c54" stroke-width="0.55" stroke-opacity="0.34"/>',
                )
            )
    return polygons


def _scatter_points(
    projected: np.ndarray,
    screen: np.ndarray,
    *,
    boundary_mask: np.ndarray | None = None,
) -> list[str]:
    depth_min = float(projected[:, 2].min())
    depth_span = float(max(projected[:, 2].max() - depth_min, 1e-9))
    order = np.argsort(projected[:, 2])
    circles = []
    for index in order:
        x, y = screen[index]
        depth = (projected[index, 2] - depth_min) / depth_span
        radius = 1.45 + 1.35 * depth
        opacity = 0.45 + 0.45 * depth
        fill = "#1d5f99"
        if boundary_mask is not None and bool(boundary_mask[index]):
            fill = "#d36b2a"
            radius += 0.55
            opacity = min(0.95, opacity + 0.1)
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'fill="{fill}" fill-opacity="{opacity:.3f}"/>'
        )
    return circles


def _panel_frame(x: int, y: int, width: int, height: int, title: str, subtitle: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="#f7faf8" stroke="#d7e1db"/>',
            f'<text x="{x + 32}" y="{y + 40}" class="title">{escape(title)}</text>',
            f'<text x="{x + 32}" y="{y + 66}" class="subtitle">{escape(subtitle)}</text>',
        ]
    )


def _supported_cells(uv_tile: np.ndarray, occupancy: ParameterOccupancy) -> np.ndarray:
    centers = (
        uv_tile[:-1, :-1]
        + uv_tile[1:, :-1]
        + uv_tile[1:, 1:]
        + uv_tile[:-1, 1:]
    ) * 0.25
    return occupancy.classify_supported(centers.reshape(-1, 2)).reshape(centers.shape[:2])


def _visible_vertices(cell_mask: np.ndarray) -> np.ndarray:
    rows, cols = cell_mask.shape[0] + 1, cell_mask.shape[1] + 1
    vertices = np.zeros((rows, cols), dtype=bool)
    for i in range(cell_mask.shape[0]):
        for j in range(cell_mask.shape[1]):
            if bool(cell_mask[i, j]):
                vertices[i : i + 2, j : j + 2] = True
    return vertices


def render_svg(
    point_cloud: np.ndarray,
    parameters: np.ndarray,
    fitted_tiles: list[tuple[np.ndarray, np.ndarray]],
    *,
    occupancy: ParameterOccupancy,
    output: Path,
    cloud: str,
    patches: int,
    rmse: float,
) -> None:
    left_panel = (60.0, 78.0, 660.0, 520.0)
    right_panel = (780.0, 78.0, 660.0, 520.0)

    points_flat = point_cloud.reshape(-1, 3)
    visible_parts = []
    tile_cell_masks = []
    tile_vertex_masks = []
    for uv_tile, xyz_tile in fitted_tiles:
        cell_mask = _supported_cells(uv_tile, occupancy)
        vertex_mask = _visible_vertices(cell_mask)
        tile_cell_masks.append(cell_mask)
        tile_vertex_masks.append(vertex_mask)
        if np.any(vertex_mask):
            visible_parts.append(xyz_tile[vertex_mask])
    if not visible_parts:
        visible_parts = [xyz_tile.reshape(-1, 3) for _, xyz_tile in fitted_tiles]

    points_projected = _project(points_flat, azimuth_deg=-36.0, elevation_deg=18.0)
    visible_projected = _project(
        np.vstack(visible_parts), azimuth_deg=-36.0, elevation_deg=18.0
    )

    left_map = _panel_mapper(points_projected, left_panel)
    right_map = _panel_mapper(visible_projected, right_panel)
    points_screen = left_map(points_projected[:, :2])
    boundary_mask = occupancy.classify_boundary(parameters)

    polygons: list[tuple[float, str]] = []
    for tile_index, (uv_tile, xyz_tile) in enumerate(fitted_tiles):
        del uv_tile
        rows, cols, _ = xyz_tile.shape
        projected = _project(
            xyz_tile.reshape(-1, 3), azimuth_deg=-36.0, elevation_deg=18.0
        )
        screen = right_map(projected[:, :2])
        polygons.extend(
            _surface_polygons(
                projected,
                screen,
                rows,
                cols,
                cell_mask=tile_cell_masks[tile_index],
            )
        )
    polygon_svgs = [svg for _, svg in sorted(polygons, key=lambda item: item[0])]

    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="680" viewBox="0 0 1500 680" role="img">',
        "<title>LA point cloud before and fitted T-spline surface after</title>",
        "<desc>Side-by-side visualization of a left-atrium point cloud and the FasTFit fitted surface.</desc>",
        "<style>",
        "text{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#17312c}",
        ".heading{font-size:30px;font-weight:700}",
        ".caption{font-size:16px;fill:#4f625d}",
        ".title{font-size:22px;font-weight:700}",
        ".subtitle{font-size:14px;fill:#5d706b}",
        "</style>",
        '<rect x="0" y="0" width="1500" height="680" fill="#ffffff"/>',
        '<text x="60" y="42" class="heading">LA Point Cloud to T-Spline Surface</text>',
        (
            f'<text x="60" y="66" class="caption">Cloud: LA {escape(cloud)}. '
            f'Input points: {points_flat.shape[0]}. Fitted patches: {patches}. RMSE: {rmse:.3f}.</text>'
        ),
        _panel_frame(60, 78, 660, 520, "Before: point cloud", "Orange marks detected support boundaries"),
        _panel_frame(780, 78, 660, 520, "After: fitted surface", "Unsupported hole/exterior cells are clipped"),
        '<g aria-label="input point cloud">',
        *_scatter_points(points_projected, points_screen, boundary_mask=boundary_mask),
        "</g>",
        '<g aria-label="fitted surface">',
        *polygon_svgs,
        "</g>",
        '<text x="60" y="634" class="caption">Generated by examples/render_before_after.py using spherical parameterization, occupancy masking, and display-only mesh smoothing.</text>',
        "</svg>",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(body), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", choices=["ED", "ES"], default="ED")
    parser.add_argument("--data-dir", type=Path, default=Path("data/la"))
    parser.add_argument("--patch-samples", type=int, default=24)
    parser.add_argument("--occupancy-resolution", type=int, default=180)
    parser.add_argument("--occupancy-dilation", type=int, default=2)
    parser.add_argument("--surface-smoothing-iterations", type=int, default=2)
    parser.add_argument("--surface-smoothing-strength", type=float, default=0.18)
    parser.add_argument("--max-error", type=float, default=12.0)
    parser.add_argument("--smoothing", type=float, default=1e-3)
    parser.add_argument("--min-region-fill-ratio", type=float, default=0.85)
    parser.add_argument("--support-grid", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/before_after_fit.svg"))
    args = parser.parse_args(argv)

    points = _load_la_points(args.data_dir, args.cloud)
    parameters = spherical_parameterize(points)
    occupancy = build_parameter_occupancy(
        parameters,
        resolution=args.occupancy_resolution,
        dilation=args.occupancy_dilation,
    )
    model = fit_fastfit_surface(
        points,
        parameters=parameters,
        options=FastFitOptions(
            degree=3,
            max_error=args.max_error,
            initial_splits=(4, 4),
            max_depth=10,
            smoothing=args.smoothing,
            workers=args.workers,
            min_region_fill_ratio=args.min_region_fill_ratio,
            support_grid=args.support_grid,
        ),
    )

    fitted_points = model.evaluate_points(parameters, workers=args.workers)
    distances = np.linalg.norm(fitted_points - points, axis=1)
    rmse = float(np.sqrt(np.mean(distances * distances)))
    fitted_tiles = _patch_tiles(model, args.patch_samples)
    fitted_tiles = _smooth_tiles(
        fitted_tiles,
        occupancy,
        iterations=args.surface_smoothing_iterations,
        strength=args.surface_smoothing_strength,
    )

    render_svg(
        points,
        parameters,
        fitted_tiles,
        occupancy=occupancy,
        output=args.output,
        cloud=args.cloud,
        patches=len(model.patches),
        rmse=rmse,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
