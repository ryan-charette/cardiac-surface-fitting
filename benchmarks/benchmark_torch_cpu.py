"""CPU benchmarks for PyTorch framework-native baselines."""

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
import torch

from cardiac_geometry.geometry.point_cloud import chamfer_distance_numpy
from cardiac_geometry.io.load_tmesh import load_sample_case
from cardiac_geometry.reference.tspline_numpy import evaluate_tspline_numpy
from cardiac_geometry.torch_ops.chamfer_op import chamfer_distance_torch
from cardiac_geometry.torch_ops.tspline_op import evaluate_tspline_torch


def _time_cpu(func, repeats: int):
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)
    assert result is not None
    return float(np.mean(times)), float(np.median(times)), result


def benchmark_tspline(case: str, grid: tuple[int, int], repeats: int) -> list[dict[str, object]]:
    surface = load_sample_case(case)
    u = np.linspace(surface.parameter_range_u[0], surface.parameter_range_u[1], grid[0])
    v = np.linspace(surface.parameter_range_v[0], surface.parameter_range_v[1], grid[1])
    numpy_mean, numpy_median, numpy_result = _time_cpu(
        lambda: evaluate_tspline_numpy(
            u,
            v,
            surface.knots_u,
            surface.knots_v,
            surface.control_points,
            surface.weights,
            degree=surface.degree,
        ),
        repeats,
    )

    u_t = torch.tensor(u, dtype=torch.float32)
    v_t = torch.tensor(v, dtype=torch.float32)
    ku_t = torch.tensor(surface.knots_u, dtype=torch.float32)
    kv_t = torch.tensor(surface.knots_v, dtype=torch.float32)
    points_t = torch.tensor(surface.control_points, dtype=torch.float32)
    weights_t = torch.tensor(surface.weights, dtype=torch.float32)

    torch_mean, torch_median, torch_result = _time_cpu(
        lambda: evaluate_tspline_torch(
            u_t,
            v_t,
            ku_t,
            kv_t,
            points_t,
            weights_t,
            degree=surface.degree,
        ),
        repeats,
    )
    torch_np = torch_result.detach().cpu().numpy()
    samples = grid[0] * grid[1]
    return [
        {
            "benchmark": "tspline_eval",
            "method": "numpy_vectorized_cpu",
            "case": case,
            "size": f"{grid[0]}x{grid[1]}",
            "points": "",
            "mean_ms": numpy_mean * 1000.0,
            "median_ms": numpy_median * 1000.0,
            "throughput_per_s": samples / numpy_mean,
            "max_abs_error": 0.0,
        },
        {
            "benchmark": "tspline_eval",
            "method": "torch_framework_cpu",
            "case": case,
            "size": f"{grid[0]}x{grid[1]}",
            "points": "",
            "mean_ms": torch_mean * 1000.0,
            "median_ms": torch_median * 1000.0,
            "throughput_per_s": samples / torch_mean,
            "max_abs_error": float(np.max(np.abs(torch_np - numpy_result))),
        },
    ]


def benchmark_chamfer(num_points: int, repeats: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(2026)
    source = rng.normal(size=(num_points, 3)).astype(np.float32)
    target = rng.normal(size=(num_points, 3)).astype(np.float32)
    numpy_mean, numpy_median, numpy_result = _time_cpu(
        lambda: chamfer_distance_numpy(source, target, chunk_size=512),
        repeats,
    )

    source_t = torch.tensor(source)
    target_t = torch.tensor(target)
    torch_mean, torch_median, torch_result = _time_cpu(
        lambda: chamfer_distance_torch(source_t, target_t),
        repeats,
    )
    return [
        {
            "benchmark": "chamfer",
            "method": "numpy_chunked_cpu",
            "case": "",
            "size": "",
            "points": num_points,
            "mean_ms": numpy_mean * 1000.0,
            "median_ms": numpy_median * 1000.0,
            "throughput_per_s": num_points / numpy_mean,
            "max_abs_error": 0.0,
        },
        {
            "benchmark": "chamfer",
            "method": "torch_cdist_cpu",
            "case": "",
            "size": "",
            "points": num_points,
            "mean_ms": torch_mean * 1000.0,
            "median_ms": torch_median * 1000.0,
            "throughput_per_s": num_points / torch_mean,
            "max_abs_error": abs(float(torch_result.detach().cpu()) - float(numpy_result)),
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["tube", "bifurcation"], default="tube")
    parser.add_argument("--grid", nargs=2, type=int, default=[48, 32], metavar=("U", "V"))
    parser.add_argument("--chamfer-points", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_torch_cpu.csv"))
    args = parser.parse_args(argv)

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    rows = []
    rows.extend(benchmark_tspline(args.case, (args.grid[0], args.grid[1]), args.repeats))
    rows.extend(benchmark_chamfer(args.chamfer_points, args.repeats))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("environment")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_threads={torch.get_num_threads()}")
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[name]) for name in fieldnames))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

