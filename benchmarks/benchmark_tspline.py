"""Benchmark the NumPy reference and fused CPU mapping."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from cardiac_geometry.io.load_tmesh import load_sample_case
from cardiac_geometry.reference.tspline_numpy import (
    evaluate_tspline_numpy,
    evaluate_tspline_numpy_fused,
)


def _time(func, args, repeats: int) -> tuple[float, float, np.ndarray]:
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - start)
    assert result is not None
    return float(np.mean(times)), float(np.median(times)), result


def run_case(case: str, grid: tuple[int, int], repeats: int) -> list[dict[str, float | str | int]]:
    surface = load_sample_case(case)
    u = np.linspace(surface.parameter_range_u[0], surface.parameter_range_u[1], grid[0])
    v = np.linspace(surface.parameter_range_v[0], surface.parameter_range_v[1], grid[1])
    call_args = (
        u,
        v,
        surface.knots_u,
        surface.knots_v,
        surface.control_points,
        surface.weights,
    )
    vector_mean, vector_median, vector_result = _time(
        evaluate_tspline_numpy, call_args, repeats
    )
    fused_mean, fused_median, fused_result = _time(
        evaluate_tspline_numpy_fused, call_args, 1
    )
    samples = grid[0] * grid[1]
    return [
        {
            "case": case,
            "method": "numpy_vectorized",
            "num_u": grid[0],
            "num_v": grid[1],
            "control_points": surface.num_control_points,
            "mean_ms": vector_mean * 1000.0,
            "median_ms": vector_median * 1000.0,
            "samples_per_second": samples / vector_mean,
            "max_abs_error": 0.0,
        },
        {
            "case": case,
            "method": "numpy_fused_mapping",
            "num_u": grid[0],
            "num_v": grid[1],
            "control_points": surface.num_control_points,
            "mean_ms": fused_mean * 1000.0,
            "median_ms": fused_median * 1000.0,
            "samples_per_second": samples / fused_mean,
            "max_abs_error": float(np.max(np.abs(vector_result - fused_result))),
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["tube", "bifurcation"], default="bifurcation")
    parser.add_argument("--grid", nargs=2, type=int, default=[80, 40], metavar=("U", "V"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="outputs/benchmark_tspline.csv")
    args = parser.parse_args(argv)

    rows = run_case(args.case, (args.grid[0], args.grid[1]), args.repeats)
    fieldnames = list(rows[0].keys())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[key]) for key in fieldnames))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
