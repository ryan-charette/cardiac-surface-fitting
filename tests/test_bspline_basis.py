import unittest

import numpy as np

from cardiac_geometry.reference.bspline_basis import (
    basis_matrix,
    bspline_basis_one,
    tensor_product_basis,
)


class BSplineBasisTests(unittest.TestCase):
    def test_degree_zero_includes_right_endpoint_left_limit(self):
        self.assertAlmostEqual(bspline_basis_one(0.5, [0.0, 1.0], degree=0), 1.0)
        self.assertAlmostEqual(bspline_basis_one(1.0, [0.0, 1.0], degree=0), 1.0)

    def test_degree_one_triangular_basis(self):
        knots = [0.0, 0.5, 1.0]
        self.assertAlmostEqual(bspline_basis_one(0.0, knots, degree=1), 0.0)
        self.assertAlmostEqual(bspline_basis_one(0.25, knots, degree=1), 0.5)
        self.assertAlmostEqual(bspline_basis_one(0.5, knots, degree=1), 1.0)
        self.assertAlmostEqual(bspline_basis_one(0.75, knots, degree=1), 0.5)

    def test_repeated_knots_do_not_create_nan(self):
        values = [
            bspline_basis_one(x, [0.0, 0.0, 1.0], degree=1)
            for x in np.linspace(0.0, 1.0, 9)
        ]
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreaterEqual(min(values), 0.0)

    def test_basis_matrix_shape(self):
        samples = np.linspace(0.0, 1.0, 5)
        knots = np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 1.0]])
        out = basis_matrix(samples, knots, degree=1)
        self.assertEqual(out.shape, (5, 2))

    def test_normalized_tensor_product_basis_sums_to_one(self):
        u = np.array([0.25, 0.5, 0.75])
        v = np.array([0.25, 0.5])
        knots = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        basis = tensor_product_basis(u, v, knots, knots, degree=1, normalize=True)
        np.testing.assert_allclose(basis.sum(axis=-1), 1.0)

    def test_normalization_rejects_zero_support(self):
        knots = np.array([[0.0, 0.5, 1.0]])
        with self.assertRaises(FloatingPointError):
            tensor_product_basis([2.0], [2.0], knots, knots, degree=1, normalize=True)


if __name__ == "__main__":
    unittest.main()

