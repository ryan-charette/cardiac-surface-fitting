"""Benchmark LA point-cloud compression with the FasTFit T-spline baseline."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from cardiac_geometry.fastfit import FastFitOptions, fit_fastfit_surface
from cardiac_geometry.geometry.parameterization import (
    pca_parameterize,
    spherical_parameterize,
)


LA_ZIP_MEMBERS = {
    "ED": "heart-modeling-main/Heart Modeling/LA-vertices-ED-Centered-Trimmed.txt",
    "ES": "heart-modeling-main/Heart Modeling/LA-vertices-ES-Centered-Trimmed.txt",
}

LA_TEXT_FILES = {
    "ED": "LA-vertices-ED-Centered-Trimmed.txt",
    "ES": "LA-vertices-ES-Centered-Trimmed.txt",
}


def validate_points(points: np.ndarray, source: str) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{source} did not contain an Nx3 point cloud")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{source} contains non-finite coordinates")
    return points


def load_points_from_zip(zip_path: Path, member: str) -> tuple[np.ndarray, int]:
    with ZipFile(zip_path) as zf:
        info = zf.getinfo(member)
        with zf.open(member) as fp:
            points = np.loadtxt(fp, dtype=np.float64)
    return validate_points(points, member), int(info.file_size)


def load_points_from_data_dir(data_dir: Path, filename: str) -> tuple[np.ndarray, int]:
    path = data_dir / filename
    points = np.loadtxt(path, dtype=np.float64)
    return validate_points(points, str(path)), int(path.stat().st_size)


def parameterize_points(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.parameterization == "pca":
        return pca_parameterize(points)
    if args.parameterization == "spherical":
        return spherical_parameterize(points, pole_axis=args.pole_axis)
    raise ValueError(f"unknown parameterization {args.parameterization!r}")


def summarize_error(model, parameters: np.ndarray, points: np.ndarray) -> dict[str, float]:
    fitted = model.evaluate_points(parameters).reshape(-1, 3)
    distances = np.linalg.norm(fitted - points, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(distances * distances))),
        "mean_abs_error": float(np.mean(distances)),
        "p95_abs_error": float(np.percentile(distances, 95)),
        "max_abs_error": float(np.max(distances)),
    }


def bytes_to_kib(value: float) -> float:
    return value / 1024.0


def model_memory_bytes(patches: int, degree: int, dtype_bytes: int) -> tuple[int, int]:
    control_points_per_patch = (degree + 1) * (degree + 1)
    compact_scalars_per_patch = control_points_per_patch * 3 + 4
    compact = patches * compact_scalars_per_patch * dtype_bytes

    expanded_control_points = patches * control_points_per_patch
    knot_width = degree + 2
    expanded_scalars_per_control = 2 * knot_width + 3 + 1
    expanded = expanded_control_points * expanded_scalars_per_control * dtype_bytes
    return compact, expanded


def run_one(
    label: str,
    points: np.ndarray,
    text_bytes: int,
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    params = parameterize_points(points, args)
    start = time.perf_counter()
    model = fit_fastfit_surface(
        points,
        parameters=params,
        options=FastFitOptions(
            degree=args.degree,
            max_error=threshold,
            initial_splits=(args.initial_splits[0], args.initial_splits[1]),
            max_depth=args.max_depth,
            smoothing=args.smoothing,
            workers=args.workers,
            min_region_fill_ratio=args.min_region_fill_ratio,
            support_grid=args.support_grid,
        ),
    )
    fit_ms = (time.perf_counter() - start) * 1000.0
    errors = summarize_error(model, params, points)

    raw_f64 = points.size * 8
    raw_f32 = points.size * 4
    compact_f64, expanded_f64 = model_memory_bytes(
        len(model.patches), args.degree, dtype_bytes=8
    )
    compact_f32, expanded_f32 = model_memory_bytes(
        len(model.patches), args.degree, dtype_bytes=4
    )

    return {
        "cloud": label,
        "threshold": threshold,
        "points": int(points.shape[0]),
        "degree": args.degree,
        "workers": args.workers,
        "parameterization": args.parameterization,
        "min_region_fill_ratio": args.min_region_fill_ratio,
        "patches": len(model.patches),
        "control_points": model.num_control_points,
        "fit_ms": fit_ms,
        "rmse": errors["rmse"],
        "mean_abs_error": errors["mean_abs_error"],
        "p95_abs_error": errors["p95_abs_error"],
        "max_abs_error": errors["max_abs_error"],
        "text_file_kib": bytes_to_kib(text_bytes),
        "raw_float64_kib": bytes_to_kib(raw_f64),
        "compact_model_float64_kib": bytes_to_kib(compact_f64),
        "expanded_local_knots_float64_kib": bytes_to_kib(expanded_f64),
        "compact_savings_vs_raw_float64_pct": 100.0 * (1.0 - compact_f64 / raw_f64),
        "expanded_savings_vs_raw_float64_pct": 100.0 * (1.0 - expanded_f64 / raw_f64),
        "raw_float32_kib": bytes_to_kib(raw_f32),
        "compact_model_float32_kib": bytes_to_kib(compact_f32),
        "expanded_local_knots_float32_kib": bytes_to_kib(expanded_f32),
        "compact_savings_vs_raw_float32_pct": 100.0 * (1.0 - compact_f32 / raw_f32),
        "expanded_savings_vs_raw_float32_pct": 100.0 * (1.0 - expanded_f32 / raw_f32),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Optional source zip containing the original heart-modeling files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/la"),
        help="Directory containing the trimmed LA point-cloud text files.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2.0, 3.0, 5.0, 8.0])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--parameterization",
        choices=["spherical", "pca"],
        default="spherical",
        help="Unorganized point-cloud parameterization. Spherical is the LA default because holes remain empty angular regions.",
    )
    parser.add_argument(
        "--pole-axis",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="PCA axis used as the spherical polar axis.",
    )
    parser.add_argument("--initial-splits", nargs=2, type=int, default=[4, 4], metavar=("U", "V"))
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1e-3,
        help="Tikhonov regularization for sparse patch fits.",
    )
    parser.add_argument(
        "--min-region-fill-ratio",
        type=float,
        default=0.85,
        help="Split otherwise-low-error regions that have sparse parameter-space support.",
    )
    parser.add_argument(
        "--support-grid",
        type=int,
        default=8,
        help="Per-region grid resolution used by the fill-ratio split criterion.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/la_memory_benchmark.csv"))
    args = parser.parse_args(argv)

    rows = []
    for label in ["ED", "ES"]:
        if args.zip is not None:
            points, text_bytes = load_points_from_zip(args.zip, LA_ZIP_MEMBERS[label])
        else:
            points, text_bytes = load_points_from_data_dir(
                args.data_dir, LA_TEXT_FILES[label]
            )
        for threshold in args.thresholds:
            rows.append(run_one(label, points, text_bytes, threshold, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[name]) for name in fieldnames))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
