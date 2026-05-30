from __future__ import annotations

import argparse, csv, statistics, time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from cardiac_geometry.io.load_tmesh import load_sample_case
from cardiac_geometry.reference.tspline_numpy import evaluate_tspline_numpy
from cardiac_geometry.torch_ops.tspline_op import evaluate_tspline_torch
from cardiac_geometry.torch_ops.chamfer_op import chamfer_distance_torch

def time_cuda(fn, repeats=20, warmup=5):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times = []
    result = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(statistics.mean(times)), float(statistics.median(times)), result

def bench_tspline(case, grid, repeats):
    surface = load_sample_case(case)
    u = np.linspace(*surface.parameter_range_u, grid[0])
    v = np.linspace(*surface.parameter_range_v, grid[1])
    ref = evaluate_tspline_numpy(
        u, v, surface.knots_u, surface.knots_v,
        surface.control_points, surface.weights, degree=surface.degree
    )

    device = "cuda"
    u_t = torch.tensor(u, device=device, dtype=torch.float32)
    v_t = torch.tensor(v, device=device, dtype=torch.float32)
    ku = torch.tensor(surface.knots_u, device=device, dtype=torch.float32)
    kv = torch.tensor(surface.knots_v, device=device, dtype=torch.float32)
    cp = torch.tensor(surface.control_points, device=device, dtype=torch.float32)
    w = torch.tensor(surface.weights, device=device, dtype=torch.float32)

    rows = []
    samples = grid[0] * grid[1]

    mean_ms, median_ms, out = time_cuda(
        lambda: evaluate_tspline_torch(u_t, v_t, ku, kv, cp, w, degree=surface.degree),
        repeats=repeats,
    )
    err = np.max(np.abs(out.detach().cpu().numpy() - ref))
    rows.append(["tspline_eval", "torch_framework_cuda", case, f"{grid[0]}x{grid[1]}", samples, mean_ms, median_ms, samples / (mean_ms / 1000), err])

    try:
        from triton_kernels.tspline_eval import evaluate_tspline_triton
        mean_ms, median_ms, out = time_cuda(
            lambda: evaluate_tspline_triton(u_t, v_t, ku, kv, cp, w),
            repeats=repeats,
        )
        err = np.max(np.abs(out.detach().cpu().numpy() - ref))
        rows.append(["tspline_eval", "triton_degree2_cuda", case, f"{grid[0]}x{grid[1]}", samples, mean_ms, median_ms, samples / (mean_ms / 1000), err])
    except Exception as exc:
        rows.append(["tspline_eval", f"triton_failed:{type(exc).__name__}", case, f"{grid[0]}x{grid[1]}", samples, "", "", "", str(exc)[:160]])

    return rows

def bench_chamfer(points, repeats):
    torch.manual_seed(2026)
    src = torch.randn(points, 3, device="cuda")
    tgt = torch.randn(points, 3, device="cuda")
    rows = []

    mean_ms, median_ms, out = time_cuda(
        lambda: chamfer_distance_torch(src, tgt),
        repeats=repeats,
    )
    rows.append(["chamfer", "torch_cdist_cuda", "", f"{points}x{points}", points, mean_ms, median_ms, points / (mean_ms / 1000), 0.0])

    try:
        from triton_kernels.chamfer_distance import chamfer_distance_triton
        mean_ms, median_ms, out2 = time_cuda(
            lambda: chamfer_distance_triton(src, tgt),
            repeats=repeats,
        )
        err = abs(float(out2.detach().cpu()) - float(out.detach().cpu()))
        rows.append(["chamfer", "triton_tiled_cuda", "", f"{points}x{points}", points, mean_ms, median_ms, points / (mean_ms / 1000), err])
    except Exception as exc:
        rows.append(["chamfer", f"triton_failed:{type(exc).__name__}", "", f"{points}x{points}", points, "", "", "", str(exc)[:160]])

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="bifurcation", choices=["tube", "bifurcation"])
    ap.add_argument("--grid", nargs=2, type=int, default=[400, 200])
    ap.add_argument("--chamfer-points", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--output", type=Path, default=Path("outputs/gpu_benchmark_4090.csv"))
    args = ap.parse_args()

    print("env")
    print("torch", torch.__version__)
    print("cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))

    rows = []
    rows += bench_tspline(args.case, tuple(args.grid), args.repeats)
    rows += bench_chamfer(args.chamfer_points, args.repeats)

    args.output.parent.mkdir(exist_ok=True)
    header = ["benchmark", "method", "case", "size", "samples_or_points", "mean_ms", "median_ms", "throughput_per_s", "max_abs_error"]
    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(",".join(header))
    for r in rows:
        print(",".join(map(str, r)))
    print("wrote", args.output)

if __name__ == "__main__":
    main()
