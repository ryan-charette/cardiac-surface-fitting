"""Evaluate the bifurcation sample with the NumPy reference implementation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from cardiac_geometry.geometry.surface import sample_surface
from cardiac_geometry.io.load_tmesh import load_sample_case


def main() -> int:
    surface = load_sample_case("bifurcation")
    u, v, xyz = sample_surface(surface, 160, 96)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    rows = np.column_stack([uu.reshape(-1), vv.reshape(-1), xyz.reshape(-1, 3)])
    output = Path("outputs/bifurcation_reference_surface.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, rows, delimiter=",", header="u,v,x,y,z", comments="")
    print(f"wrote {rows.shape[0]} samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
