"""Benchmark the NumPy Chamfer-distance reference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from cardiac_geometry.geometry.point_cloud import chamfer_distance_numpy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-points", type=int, default=2048)
    parser.add_argument("--target-points", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(2026)
    source = rng.normal(size=(args.source_points, 3))
    target = rng.normal(size=(args.target_points, 3))
    times = []
    value = 0.0
    for _ in range(args.repeats):
        start = time.perf_counter()
        value = chamfer_distance_numpy(source, target, chunk_size=args.chunk_size)
        times.append(time.perf_counter() - start)
    print("method,source_points,target_points,chunk_size,mean_ms,value")
    print(
        "numpy_chunked,"
        f"{args.source_points},{args.target_points},{args.chunk_size},"
        f"{np.mean(times) * 1000.0:.3f},{value:.8f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
