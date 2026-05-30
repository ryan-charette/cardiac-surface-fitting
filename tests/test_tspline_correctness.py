import unittest

import numpy as np

from cardiac_geometry.io.load_tmesh import load_sample_case
from cardiac_geometry.reference.tspline_numpy import (
    evaluate_tspline_numpy,
    evaluate_tspline_numpy_fused,
)


class TSplineCorrectnessTests(unittest.TestCase):
    def test_sample_case_control_point_counts(self):
        self.assertEqual(load_sample_case("tube").num_control_points, 42)
        self.assertEqual(load_sample_case("bifurcation").num_control_points, 84)

    def test_vectorized_and_fused_agree_on_tube(self):
        surface = load_sample_case("tube")
        u = np.linspace(*surface.parameter_range_u, 8)
        v = np.linspace(*surface.parameter_range_v, 5)
        vectorized = evaluate_tspline_numpy(
            u,
            v,
            surface.knots_u,
            surface.knots_v,
            surface.control_points,
            surface.weights,
            degree=surface.degree,
        )
        fused = evaluate_tspline_numpy_fused(
            u,
            v,
            surface.knots_u,
            surface.knots_v,
            surface.control_points,
            surface.weights,
            degree=surface.degree,
        )
        np.testing.assert_allclose(vectorized, fused, rtol=0.0, atol=1e-12)

    def test_mesh_and_vector_inputs_agree(self):
        surface = load_sample_case("bifurcation")
        u = np.linspace(*surface.parameter_range_u, 6)
        v = np.linspace(*surface.parameter_range_v, 4)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        vector = surface.evaluate(u, v)
        mesh = surface.evaluate(uu, vv)
        np.testing.assert_allclose(vector, mesh, rtol=0.0, atol=1e-12)

    def test_normalized_basis_does_not_change_rational_surface(self):
        surface = load_sample_case("tube")
        u = np.linspace(*surface.parameter_range_u, 7)
        v = np.linspace(*surface.parameter_range_v, 5)
        raw = surface.evaluate(u, v, normalize_basis=False)
        normalized = surface.evaluate(u, v, normalize_basis=True)
        np.testing.assert_allclose(raw, normalized, rtol=0.0, atol=1e-12)

    def test_constant_control_points_produce_constant_surface(self):
        surface = load_sample_case("tube")
        constant = np.tile(np.array([[3.0, -2.0, 0.5]]), (surface.num_control_points, 1))
        u = np.linspace(*surface.parameter_range_u, 5)
        v = np.linspace(*surface.parameter_range_v, 5)
        xyz = evaluate_tspline_numpy(
            u,
            v,
            surface.knots_u,
            surface.knots_v,
            constant,
            surface.weights,
            degree=surface.degree,
        )
        expected = np.broadcast_to(constant[0], xyz.shape)
        np.testing.assert_allclose(xyz, expected, rtol=0.0, atol=1e-12)

    def test_translation_equivariance(self):
        surface = load_sample_case("bifurcation")
        shift = np.array([1.5, -0.25, 2.0])
        u = np.linspace(*surface.parameter_range_u, 5)
        v = np.linspace(*surface.parameter_range_v, 3)
        baseline = surface.evaluate(u, v)
        shifted = evaluate_tspline_numpy(
            u,
            v,
            surface.knots_u,
            surface.knots_v,
            surface.control_points + shift,
            surface.weights,
            degree=surface.degree,
        )
        np.testing.assert_allclose(shifted, baseline + shift, rtol=0.0, atol=1e-12)

    def test_zero_denominator_is_reported(self):
        surface = load_sample_case("tube")
        with self.assertRaises(FloatingPointError):
            surface.evaluate(np.array([2.0]), np.array([2.0]))


if __name__ == "__main__":
    unittest.main()
