"""Render a mesh-forward project visualization from the LA FasTFit surface."""

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
from cardiac_geometry.geometry.occupancy import build_parameter_occupancy
from cardiac_geometry.geometry.parameterization import spherical_parameterize
from render_before_after import (
    _load_la_points,
    _patch_tiles,
    _project,
    _smooth_tiles,
    _supported_cells,
    _visible_vertices,
)


def _panel_mapper(projected: np.ndarray, width: int, height: int, *, pad: float):
    xy = projected[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    center = 0.5 * (mins + maxs)

    def map_points(values: np.ndarray) -> np.ndarray:
        out = np.empty((values.shape[0], 2), dtype=np.float64)
        out[:, 0] = 0.5 * width + (values[:, 0] - center[0]) * scale
        out[:, 1] = 0.5 * height - (values[:, 1] - center[1]) * scale
        return out

    return map_points


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))


def _mix(a: str, b: str, t: float) -> str:
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    t = float(np.clip(t, 0.0, 1.0))
    return (
        f"#{round(ar + (br - ar) * t):02x}"
        f"{round(ag + (bg - ag) * t):02x}"
        f"{round(ab + (bb - ab) * t):02x}"
    )


def _shade(depth: float) -> str:
    if depth < 0.5:
        return _mix("#cf5c61", "#eda1a0", depth / 0.5)
    return _mix("#eda1a0", "#ffd7d2", (depth - 0.5) / 0.5)


def _surface_polygons(
    projected: np.ndarray,
    screen: np.ndarray,
    rows: int,
    cols: int,
    cell_mask: np.ndarray,
) -> list[tuple[float, str]]:
    depth_min = float(projected[:, 2].min())
    depth_span = float(max(projected[:, 2].max() - depth_min, 1e-9))
    polygons: list[tuple[float, str]] = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            if not bool(cell_mask[row, col]):
                continue
            indices = np.array(
                [
                    row * cols + col,
                    (row + 1) * cols + col,
                    (row + 1) * cols + col + 1,
                    row * cols + col + 1,
                ],
                dtype=np.int64,
            )
            pts = screen[indices]
            depth = float(projected[indices, 2].mean())
            normalized = (depth - depth_min) / depth_span
            fill = _shade(normalized)
            opacity = 0.76 + 0.18 * normalized
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            polygons.append(
                (
                    depth,
                    f'<polygon points="{points}" fill="{fill}" fill-opacity="{min(opacity, 0.78):.3f}" '
                    'stroke="#9d333b" stroke-width="0.48" stroke-opacity="0.20"/>',
                )
            )
    return polygons


def _mesh_lines(
    projected: np.ndarray,
    screen: np.ndarray,
    rows: int,
    cols: int,
    vertex_mask: np.ndarray,
) -> list[tuple[float, str]]:
    lines: list[tuple[float, str]] = []

    def flat(row: int, col: int) -> int:
        return row * cols + col

    def add_segments(values: list[int]) -> None:
        if len(values) < 2:
            return
        depth = float(projected[values, 2].mean())
        points = " ".join(f"{screen[index, 0]:.2f},{screen[index, 1]:.2f}" for index in values)
        lines.append(
            (
                depth,
                f'<polyline points="{points}" fill="none" stroke="#7f2630" '
                'stroke-width="0.62" stroke-opacity="0.28" stroke-linecap="round" stroke-linejoin="round"/>',
            )
        )

    stride = max(1, rows // 13)
    for row in range(0, rows, stride):
        segment: list[int] = []
        for col in range(cols):
            if vertex_mask[row, col]:
                segment.append(flat(row, col))
            else:
                add_segments(segment)
                segment = []
        add_segments(segment)

    stride = max(1, cols // 13)
    for col in range(0, cols, stride):
        segment = []
        for row in range(rows):
            if vertex_mask[row, col]:
                segment.append(flat(row, col))
            else:
                add_segments(segment)
                segment = []
        add_segments(segment)
    return lines


def _point_cloud_circles(
    points: np.ndarray,
    mapper,
) -> list[str]:
    projected = _project(points, azimuth_deg=-34.0, elevation_deg=21.0)
    screen = mapper(projected[:, :2])
    depth_min = float(projected[:, 2].min())
    depth_span = float(max(projected[:, 2].max() - depth_min, 1e-9))
    circles = []
    for order_index in np.argsort(projected[:, 2]):
        depth = (projected[order_index, 2] - depth_min) / depth_span
        x, y = screen[order_index]
        radius = 0.75 + 0.7 * depth
        opacity = 0.28 + 0.32 * depth
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'fill="#082f73" fill-opacity="{opacity:.3f}"/>'
        )
    return circles


def render_project_visualization(
    *,
    cloud: str,
    data_dir: Path,
    output: Path,
    width: int,
    height: int,
    patch_samples: int,
) -> None:
    points = _load_la_points(data_dir, cloud)
    parameters = spherical_parameterize(points)
    occupancy = build_parameter_occupancy(parameters, resolution=200, dilation=2)
    model = fit_fastfit_surface(
        points,
        parameters=parameters,
        options=FastFitOptions(
            degree=3,
            max_error=12.0,
            initial_splits=(4, 4),
            max_depth=10,
            smoothing=1e-3,
            min_region_fill_ratio=0.85,
            support_grid=8,
            workers=8,
        ),
    )

    fitted_points = model.evaluate_points(parameters, workers=8)
    distances = np.linalg.norm(fitted_points - points, axis=1)
    rmse = float(np.sqrt(np.mean(distances * distances)))
    tiles = _smooth_tiles(
        _patch_tiles(model, patch_samples),
        occupancy,
        iterations=3,
        strength=0.22,
    )

    visible = []
    tile_masks = []
    vertex_masks = []
    for uv_tile, xyz_tile in tiles:
        cell_mask = _supported_cells(uv_tile, occupancy)
        vertex_mask = _visible_vertices(cell_mask)
        tile_masks.append(cell_mask)
        vertex_masks.append(vertex_mask)
        if np.any(vertex_mask):
            visible.append(xyz_tile[vertex_mask])
    visible_points = np.vstack(visible)
    visible_projected = _project(visible_points, azimuth_deg=-34.0, elevation_deg=21.0)
    point_projected = _project(points, azimuth_deg=-34.0, elevation_deg=21.0)
    mapper = _panel_mapper(
        np.vstack([visible_projected, point_projected]),
        width,
        height,
        pad=44.0,
    )

    polygons: list[tuple[float, str]] = []
    lines: list[tuple[float, str]] = []
    for tile_index, (_, xyz_tile) in enumerate(tiles):
        rows, cols, _ = xyz_tile.shape
        projected = _project(xyz_tile.reshape(-1, 3), azimuth_deg=-34.0, elevation_deg=21.0)
        screen = mapper(projected[:, :2])
        polygons.extend(_surface_polygons(projected, screen, rows, cols, tile_masks[tile_index]))
        lines.extend(_mesh_lines(projected, screen, rows, cols, vertex_masks[tile_index]))
    polygons = [svg for _, svg in sorted(polygons, key=lambda item: item[0])]
    lines = [svg for _, svg in sorted(lines, key=lambda item: item[0])]
    point_cloud = _point_cloud_circles(points, mapper)

    output.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g aria-label="fitted surface">',
        *polygons,
        *lines,
        "</g>",
        '<g aria-label="left atrium point cloud">',
        *point_cloud,
        "</g>",
        '<g aria-label="visualization labels" font-family="Inter,Segoe UI,Arial,sans-serif">',
        '<rect x="34" y="24" width="612" height="158" rx="14" fill="#ffffff" fill-opacity="0.84"/>',
        '<text x="46" y="62" fill="#172033" font-size="30" font-weight="700">C1 Fitted Left-Atrium Surface</text>',
        '<text x="48" y="96" fill="#536070" font-size="17">Input point cloud and constrained FasTFit surface share one aligned camera projection.</text>',
        '<circle cx="55" cy="133" r="6" fill="#082f73" fill-opacity="0.86"/>',
        '<text x="72" y="139" fill="#253247" font-size="15">blue: LA point cloud</text>',
        '<rect x="278" y="124" width="22" height="14" rx="2" fill="#eda1a0" stroke="#9d333b" stroke-opacity="0.35"/>',
        '<text x="310" y="139" fill="#253247" font-size="15">pink: C1 fitted surface</text>',
        (
            f'<text x="48" y="{height - 44}" fill="#7c2630" font-size="16">'
            f'LA {escape(cloud)} | {points.shape[0]:,} points | {len(model.patches)} patches | RMSE {rmse:.3f}'
            '</text>'
        ),
        "</g>",
        "</svg>",
    ]
    output.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", choices=["ED", "ES"], default="ED")
    parser.add_argument("--data-dir", type=Path, default=Path("data/la"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/project_visualization.svg"))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=630)
    parser.add_argument("--patch-samples", type=int, default=28)
    args = parser.parse_args(argv)

    render_project_visualization(
        cloud=args.cloud,
        data_dir=args.data_dir,
        output=args.output,
        width=args.width,
        height=args.height,
        patch_samples=args.patch_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
