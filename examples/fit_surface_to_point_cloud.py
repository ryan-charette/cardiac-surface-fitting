"""Fit a point cloud with the parallel FasTFit-style implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    parser.add_argument("--grid", nargs=2, type=int, default=[32, 24], metavar=("U", "V"))
    parser.add_argument("--max-error", type=float, default=0.025)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    surface = load_sample_case(args.case)
    _, _, target = sample_surface(surface, args.grid[0], args.grid[1])
    model = fit_fastfit_surface(
        target,
        options=FastFitOptions(
            degree=3,
            max_error=args.max_error,
            initial_splits=(4, 4),
            max_depth=6,
            workers=args.workers,
        ),
    )
    u = np.linspace(0.0, 1.0, args.grid[0])
    v = np.linspace(0.0, 1.0, args.grid[1])
    fitted = model.evaluate(u, v, workers=args.workers)
    rmse = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=-1))))
    print(
        f"case={args.case} patches={len(model.patches)} "
        f"control_points={model.num_control_points} rmse={rmse:.6g}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
