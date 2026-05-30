"""Command-line interface for examples and smoke benchmarks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cardiac_geometry.fastfit import FastFitOptions, fit_fastfit_surface
from cardiac_geometry.geometry.surface import sample_surface
from cardiac_geometry.io.load_tmesh import load_sample_case
from cardiac_geometry.reference.tspline_numpy import (
    evaluate_tspline_numpy,
    evaluate_tspline_numpy_fused,
)


def _save_surface_csv(path: Path, u: np.ndarray, v: np.ndarray, xyz: np.ndarray) -> None:
    uu, vv = np.meshgrid(u, v, indexing="ij")
    rows = np.column_stack([uu.reshape(-1), vv.reshape(-1), xyz.reshape(-1, 3)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", header="u,v,x,y,z", comments="")


def _cmd_eval(args: argparse.Namespace) -> int:
    surface = load_sample_case(args.case)
    u, v, xyz = sample_surface(
        surface,
        args.grid[0],
        args.grid[1],
        normalize_basis=args.normalize_basis,
    )
    _save_surface_csv(Path(args.output), u, v, xyz)
    print(
        f"case={surface.name} grid={args.grid[0]}x{args.grid[1]} "
        f"control_points={surface.num_control_points} output={args.output}"
    )
    return 0


def _time_call(func, *call_args, repeats: int) -> tuple[float, np.ndarray]:
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*call_args)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    assert result is not None
    return best, result


def _cmd_bench(args: argparse.Namespace) -> int:
    surface = load_sample_case(args.case)
    u = np.linspace(surface.parameter_range_u[0], surface.parameter_range_u[1], args.grid[0])
    v = np.linspace(surface.parameter_range_v[0], surface.parameter_range_v[1], args.grid[1])
    inputs = (
        u,
        v,
        surface.knots_u,
        surface.knots_v,
        surface.control_points,
        surface.weights,
    )
    vectorized_s, vectorized = _time_call(
        evaluate_tspline_numpy, *inputs, repeats=args.repeats
    )
    fused_s, fused = _time_call(evaluate_tspline_numpy_fused, *inputs, repeats=1)
    max_abs_error = float(np.max(np.abs(vectorized - fused)))
    samples = args.grid[0] * args.grid[1]
    print("method,best_ms,samples_per_second,max_abs_error")
    print(f"numpy_vectorized,{vectorized_s * 1000.0:.3f},{samples / vectorized_s:.1f},0.0")
    print(f"numpy_fused,{fused_s * 1000.0:.3f},{samples / fused_s:.1f},{max_abs_error:.3e}")
    return 0


def _cmd_fastfit(args: argparse.Namespace) -> int:
    source = load_sample_case(args.case)
    _, _, xyz = sample_surface(source, args.input_grid[0], args.input_grid[1])
    model = fit_fastfit_surface(
        xyz,
        options=FastFitOptions(
            degree=args.degree,
            max_error=args.max_error,
            initial_splits=(args.initial_splits[0], args.initial_splits[1]),
            max_depth=args.max_depth,
            smoothing=args.smoothing,
            workers=args.workers,
        ),
    )
    u = np.linspace(0.0, 1.0, args.output_grid[0])
    v = np.linspace(0.0, 1.0, args.output_grid[1])
    fitted = model.evaluate(u, v, workers=args.workers)
    _save_surface_csv(Path(args.output), u, v, fitted)
    print(
        f"case={args.case} patches={len(model.patches)} "
        f"control_points={model.num_control_points} "
        f"mean_patch_rmse={model.diagnostics['mean_patch_rmse']:.6g} "
        f"max_patch_error={model.diagnostics['max_patch_error']:.6g} "
        f"output={args.output}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cardiac-kernels")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval", help="evaluate a sample T-spline surface")
    eval_parser.add_argument("--case", choices=["tube", "bifurcation"], default="bifurcation")
    eval_parser.add_argument("--grid", nargs=2, type=int, metavar=("U", "V"), default=[120, 80])
    eval_parser.add_argument("--normalize-basis", action="store_true")
    eval_parser.add_argument("--output", default="outputs/surface.csv")
    eval_parser.set_defaults(func=_cmd_eval)

    bench_parser = sub.add_parser("bench", help="run a NumPy correctness benchmark")
    bench_parser.add_argument("--case", choices=["tube", "bifurcation"], default="bifurcation")
    bench_parser.add_argument("--grid", nargs=2, type=int, metavar=("U", "V"), default=[80, 40])
    bench_parser.add_argument("--repeats", type=int, default=3)
    bench_parser.set_defaults(func=_cmd_bench)

    fastfit_parser = sub.add_parser("fastfit", help="fit a sample surface with FasTFit")
    fastfit_parser.add_argument("--case", choices=["tube", "bifurcation"], default="tube")
    fastfit_parser.add_argument("--input-grid", nargs=2, type=int, metavar=("U", "V"), default=[64, 48])
    fastfit_parser.add_argument("--output-grid", nargs=2, type=int, metavar=("U", "V"), default=[96, 72])
    fastfit_parser.add_argument("--degree", type=int, default=3)
    fastfit_parser.add_argument("--max-error", type=float, default=0.025)
    fastfit_parser.add_argument("--initial-splits", nargs=2, type=int, metavar=("U", "V"), default=[4, 4])
    fastfit_parser.add_argument("--max-depth", type=int, default=6)
    fastfit_parser.add_argument("--smoothing", type=float, default=0.0)
    fastfit_parser.add_argument("--workers", type=int, default=None)
    fastfit_parser.add_argument("--output", default="outputs/fastfit_surface.csv")
    fastfit_parser.set_defaults(func=_cmd_fastfit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
