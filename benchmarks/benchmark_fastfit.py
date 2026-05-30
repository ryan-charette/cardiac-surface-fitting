"""Benchmark the parallel FasTFit implementation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from cardiac_geometry.fastfit import FastFitOptions, fit_fastfit_surface
from cardiac_geometry.geometry.surface import sample_surface
from cardiac_geometry.io.load_tmesh import load_sample_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["tube", "bifurcation"], default="tube")
    parser.add_argument("--grid", nargs=2, type=int, default=[64, 48], metavar=("U", "V"))
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--max-error", type=float, default=0.025)
    parser.add_argument("--initial-splits", nargs=2, type=int, default=[4, 4], metavar=("U", "V"))
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args(argv)

    surface = load_sample_case(args.case)
    _, _, xyz = sample_surface(surface, args.grid[0], args.grid[1])
    print("method,case,input_u,input_v,workers,patches,control_points,fit_ms,mean_patch_rmse,max_patch_error")
    for workers in args.workers:
        start = time.perf_counter()
        model = fit_fastfit_surface(
            xyz,
            options=FastFitOptions(
                degree=3,
                max_error=args.max_error,
                initial_splits=(args.initial_splits[0], args.initial_splits[1]),
                max_depth=args.max_depth,
                workers=workers,
            ),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"fastfit,{args.case},{args.grid[0]},{args.grid[1]},{workers},"
            f"{len(model.patches)},{model.num_control_points},{elapsed_ms:.3f},"
            f"{model.diagnostics['mean_patch_rmse']:.8g},"
            f"{model.diagnostics['max_patch_error']:.8g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

