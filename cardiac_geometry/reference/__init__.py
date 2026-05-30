"""Numerically trusted CPU reference implementations."""

from cardiac_geometry.reference.bspline_basis import (
    basis_matrix,
    bspline_basis_one,
    tensor_product_basis,
)
from cardiac_geometry.reference.tspline_numpy import (
    evaluate_tspline_numpy,
    evaluate_tspline_numpy_fused,
)

__all__ = [
    "basis_matrix",
    "bspline_basis_one",
    "tensor_product_basis",
    "evaluate_tspline_numpy",
    "evaluate_tspline_numpy_fused",
]

