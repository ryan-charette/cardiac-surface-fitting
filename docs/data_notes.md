# Data Notes

The repository includes small synthetic tube and bifurcation T-spline examples
under `data/sample`. These are enough for tests, correctness comparisons, and
smoke benchmarks.

The repository also includes the trimmed left-atrium point clouds used for the
memory benchmark:

```text
data/la/LA-vertices-ED-Centered-Trimmed.txt
data/la/LA-vertices-ES-Centered-Trimmed.txt
```

Large 4D CTA frame files from the source archive are intentionally excluded from
the cleaned project. They should be handled through Git LFS, a download script,
or a local data path outside the public repository.

This project is for computational geometry and GPU performance engineering
demonstration only. It is not intended for diagnosis, treatment planning, or
clinical use.
