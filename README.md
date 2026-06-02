# Diffeomorphic Modeling of Cardiac Surfaces

A GPU-accelerated toolkit for fitting T-spline surfaces to cardiac point-cloud data. The project includes NumPy reference evaluators, parallel FasTFit-style surface fitting, CUDA/Triton kernels, correctness tests, benchmark artifacts, and Python demos for visualization and point-cloud fitting.

## Highlights

- Clean NumPy reference evaluator for local-knot-vector rational T-splines.
- Parallel FasTFit implementation based on the split-connect-fit paper:
  adaptive Bezier patch generation over fixed 2D-parameterized point clouds.
- Default C1-constrained global refit over accepted FasTFit patches, with
  `continuity="none"` retaining the legacy independent-patch baseline.
- Correctness checks for repeated knots, endpoint sampling, zero denominators,
  Bernstein bases, C1 patch boundaries, deterministic parallel fitting,
  vector-vs-mesh evaluation, and fused-vs-vectorized evaluation.
- Fused CUDA forward kernel that computes basis values and rational accumulation
  in one pass without materializing a `U x V x C` basis tensor.
- Triton degree-2 surface-evaluation prototype for AI-kernel workflow
  comparison.
- CUDA, Triton, PyTorch, and NumPy Chamfer-distance paths for point-cloud
  fitting experiments.
- PyTorch-native fitted-surface evaluation for autograd through parameters and
  fitted control points.
- Measured RTX 4090 benchmark artifacts showing the Triton T-spline prototype
  at 0.117 ms for an 80,000-sample grid and the tiled Triton Chamfer prototype
  at 0.321 ms for an 8,192 x 8,192 point workload.
- Small bundled tube and bifurcation samples extracted from the original heart
  modeling repository.

## Visualization

![Aligned LA point cloud and C1 fitted FasTFit surface visualization](docs/assets/project_visualization.svg)

The main visualization overlays the bundled LA ED point cloud and the fitted
C1-constrained FasTFit surface in one aligned camera projection. Deep blue marks
the input point cloud; soft red marks the fitted surface. The image is generated
using spherical parameterization, parameter-space occupancy masking around
holes, and display-only mesh smoothing:

```bash
python examples/render_project_visualization.py
```

The fitted surface clips hole and exterior cells instead of rendering
unsupported patch interiors. The rendered surface uses a higher-resolution patch
grid than the fitted control mesh, so this improves presentation without
changing the benchmark fit metrics.

Use `--cloud ES` to render the end-systolic LA point cloud instead.

## Quickstart

```bash
pip install -e .
python -m unittest discover -s tests
python examples/run_bifurcation_reference.py
python benchmarks/benchmark_tspline.py --case bifurcation --grid 80 40
python benchmarks/benchmark_fastfit.py --case tube --grid 64 48 --workers 1 2 4
python benchmarks/benchmark_la_memory.py --thresholds 2 3 5 8 12 --workers 8
```

The CLI provides the same smoke workflows:

```bash
cardiac-kernels eval --case bifurcation --grid 160 96 --output outputs/bifurcation.csv
cardiac-kernels bench --case tube --grid 80 40
cardiac-kernels fastfit --case tube --input-grid 64 48 --workers 4
```

CUDA/Triton extension paths require a CUDA-enabled PyTorch install:

```bash
pip install -e .[torch,triton]
python examples/fit_surface_to_point_cloud.py --case tube
python benchmarks/benchmark_gpu_vast.py --case bifurcation --grid 400 200 --chamfer-points 8192 --repeats 20
```

## Repository Structure

```text
cardiac_geometry/
  fastfit.py         Parallel FasTFit adaptive Bezier/T-spline fitting
  reference/        NumPy basis and T-spline evaluators
  io/               T-mesh sample-data loaders
  geometry/         Surface and point-cloud helpers
  torch_ops/        Optional PyTorch differentiable wrappers
cuda/               CUDA extension sources
triton_kernels/     Triton kernel prototypes
benchmarks/         Runtime and accuracy benchmark scripts
tests/              Unit tests for reference math and loaders
examples/           Surface generation and fitting demos
docs/               Design and portfolio writeups
docs/assets/        README visualization assets
data/sample/        Small tube and bifurcation cases
data/la/            LA ED/ES point clouds used for memory benchmarks
```

## Current Validation

The local Windows workspace validates the NumPy, PyTorch CPU, FasTFit C1
continuity, optional PyTorch fitted-surface autograd, and reference-math paths.
The GPU paths were benchmarked on an RTX 4090 instance with PyTorch
2.11.0+cu128 and Triton 3.6.0.

Headline RTX 4090 results:

- T-spline eval, 400 x 200 grid: PyTorch CUDA 139.253 ms, Triton 0.117 ms,
  about 1,190x faster with max absolute error 1.87e-6.
- Chamfer distance, 8,192 x 8,192 points: `torch.cdist` 2.738 ms, Triton tiled
  0.321 ms, about 8.5x faster with max absolute error 1.49e-8.
- LA compact fitted models save 89.6% memory for ED and 87.6% for ES versus raw
  float64 point arrays at threshold 12 after spherical parameterization and
  occupancy-aware splitting.

See `docs/performance_report.md` for the full tables. Downloaded GPU artifacts,
including CSVs and Nsight Compute reports, are archived in `outputs/final_4090/`.
