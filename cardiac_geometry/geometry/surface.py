"""Surface data structures and sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cardiac_geometry.reference.tspline_numpy import evaluate_tspline_numpy


@dataclass(frozen=True)
class TMeshSurface:
    """A local-knot-vector T-spline surface description."""

    name: str
    degree: int
    knots_u: np.ndarray
    knots_v: np.ndarray
    control_points: np.ndarray
    weights: np.ndarray
    control_point_indices: np.ndarray
    parameter_range_u: tuple[float, float]
    parameter_range_v: tuple[float, float]

    @property
    def num_control_points(self) -> int:
        return int(self.control_points.shape[0])

    def evaluate(
        self,
        u: np.ndarray,
        v: np.ndarray,
        *,
        normalize_basis: bool = False,
    ) -> np.ndarray:
        return evaluate_tspline_numpy(
            u,
            v,
            self.knots_u,
            self.knots_v,
            self.control_points,
            self.weights,
            degree=self.degree,
            normalize_basis=normalize_basis,
        )


def make_parameter_grid(
    surface: TMeshSurface,
    num_u: int,
    num_v: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the recommended structured parameter grid for a sample surface."""

    u = np.linspace(surface.parameter_range_u[0], surface.parameter_range_u[1], num_u)
    v = np.linspace(surface.parameter_range_v[0], surface.parameter_range_v[1], num_v)
    return u, v


def sample_surface(
    surface: TMeshSurface,
    num_u: int,
    num_v: int,
    *,
    normalize_basis: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a T-mesh surface on its recommended parameter range."""

    u, v = make_parameter_grid(surface, num_u, num_v)
    xyz = surface.evaluate(u, v, normalize_basis=normalize_basis)
    return u, v, xyz

