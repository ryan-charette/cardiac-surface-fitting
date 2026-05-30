import unittest

import numpy as np

from cardiac_geometry.fastfit import (
    FastFitOptions,
    bernstein_basis,
    fit_bezier_patch,
    fit_fastfit_surface,
)
from cardiac_geometry.reference.tspline_numpy import evaluate_tspline_numpy


def _smooth_test_surface(num_u=25, num_v=21):
    u = np.linspace(0.0, 1.0, num_u)
    v = np.linspace(0.0, 1.0, num_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    xyz = np.empty(uu.shape + (3,), dtype=np.float64)
    xyz[..., 0] = uu
    xyz[..., 1] = vv
    xyz[..., 2] = 0.25 * uu * uu - 0.5 * uu * vv + 0.1 * vv
    return xyz


class FastFitTests(unittest.TestCase):
    def test_bernstein_partition_of_unity(self):
        samples = np.linspace(0.0, 1.0, 11)
        basis = bernstein_basis(samples, degree=3)
        np.testing.assert_allclose(basis.sum(axis=-1), 1.0, atol=1e-14)

    def test_single_patch_recovers_polynomial_surface(self):
        xyz = _smooth_test_surface()
        model = fit_fastfit_surface(
            xyz,
            options=FastFitOptions(
                degree=3,
                max_error=1e-10,
                initial_splits=(1, 1),
                workers=1,
            ),
        )
        self.assertEqual(len(model.patches), 1)
        u = np.linspace(0.0, 1.0, xyz.shape[0])
        v = np.linspace(0.0, 1.0, xyz.shape[1])
        fitted = model.evaluate(u, v)
        np.testing.assert_allclose(fitted, xyz, atol=1e-12, rtol=0.0)

    def test_adaptive_split_creates_multiple_patches_for_discontinuity(self):
        xyz = _smooth_test_surface(33, 29)
        xyz[17:, :, 2] += 0.75
        model = fit_fastfit_surface(
            xyz,
            options=FastFitOptions(
                degree=3,
                max_error=0.05,
                initial_splits=(1, 1),
                max_depth=4,
                workers=2,
                continuity="none",
            ),
        )
        self.assertGreater(len(model.patches), 1)
        self.assertLess(model.diagnostics["max_patch_error"], 0.75)
        self.assertEqual(model.diagnostics["continuity"], "none")

    def test_default_c1_refit_smooths_forced_two_patch_boundary(self):
        xyz = _smooth_test_surface(29, 23)
        model = fit_fastfit_surface(
            xyz,
            options=FastFitOptions(
                degree=3,
                max_error=1.0,
                initial_splits=(2, 1),
                max_depth=0,
                workers=1,
            ),
        )
        self.assertEqual(len(model.patches), 2)
        self.assertEqual(model.diagnostics["continuity"], "c1")
        self.assertEqual(model.diagnostics["boundary_constraints"], 8)
        self.assertLess(model.diagnostics["max_c0_gap"], 1e-10)
        self.assertLess(model.diagnostics["max_c1_gap"], 1e-10)

    def test_default_c1_refit_smooths_two_by_two_grid(self):
        xyz = _smooth_test_surface(29, 23)
        model = fit_fastfit_surface(
            xyz,
            options=FastFitOptions(
                degree=3,
                max_error=1.0,
                initial_splits=(2, 2),
                max_depth=0,
                workers=1,
            ),
        )

        self.assertEqual(len(model.patches), 4)
        self.assertEqual(model.diagnostics["boundary_constraints"], 32)
        self.assertLess(model.diagnostics["max_c0_gap"], 1e-10)
        self.assertLess(model.diagnostics["max_c1_gap"], 1e-10)

    def test_default_c1_refit_smooths_t_junction_boundaries(self):
        u = np.linspace(0.0, 1.0, 41)
        v = np.linspace(0.0, 1.0, 41)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        xyz = np.empty(uu.shape + (3,), dtype=np.float64)
        xyz[..., 0] = uu
        xyz[..., 1] = vv
        xyz[..., 2] = (
            0.2 * uu * uu
            + 0.1 * vv
            + 0.08 * (1.0 - uu) ** 2 * np.sin(5.0 * np.pi * vv)
        )

        model = fit_fastfit_surface(
            xyz,
            options=FastFitOptions(
                degree=3,
                max_error=0.03,
                initial_splits=(1, 1),
                max_depth=2,
                workers=1,
            ),
        )

        self.assertEqual(len(model.patches), 3)
        self.assertEqual(model.diagnostics["boundary_constraints"], 24)
        self.assertLess(model.diagnostics["max_c0_gap"], 1e-10)
        self.assertLess(model.diagnostics["max_c1_gap"], 1e-10)

    def test_parallel_and_serial_are_deterministic(self):
        xyz = _smooth_test_surface(31, 27)
        options = FastFitOptions(
            degree=3,
            max_error=1e-8,
            initial_splits=(2, 2),
            max_depth=2,
            workers=1,
        )
        serial = fit_fastfit_surface(xyz, options=options)
        parallel = fit_fastfit_surface(
            xyz,
            options=FastFitOptions(
                degree=3,
                max_error=1e-8,
                initial_splits=(2, 2),
                max_depth=2,
                workers=4,
            ),
        )
        self.assertEqual(len(serial.patches), len(parallel.patches))
        u = np.linspace(0.0, 1.0, 9)
        v = np.linspace(0.0, 1.0, 7)
        np.testing.assert_allclose(serial.evaluate(u, v), parallel.evaluate(u, v), atol=1e-12)

    def test_evaluate_points_uses_paired_parameters(self):
        xyz = _smooth_test_surface()
        model = fit_fastfit_surface(
            xyz,
            options=FastFitOptions(degree=3, max_error=1e-10, initial_splits=(1, 1)),
        )
        params = np.array([[0.1, 0.2], [0.4, 0.6], [0.8, 0.3]])
        values = model.evaluate_points(params)
        expected = np.column_stack(
            [
                params[:, 0],
                params[:, 1],
                0.25 * params[:, 0] ** 2
                - 0.5 * params[:, 0] * params[:, 1]
                + 0.1 * params[:, 1],
            ]
        )
        np.testing.assert_allclose(values, expected, atol=1e-12, rtol=0.0)

    def test_empty_regularized_patch_is_skipped(self):
        parameters = np.empty((0, 2), dtype=np.float64)
        points = np.empty((0, 3), dtype=np.float64)
        patch = fit_bezier_patch(
            parameters,
            points,
            (0.0, 1.0, 0.0, 1.0),
            degree=3,
            smoothing=1e-3,
        )
        self.assertIsNone(patch)

    def test_sparse_unsplittable_patch_is_stabilized(self):
        rng = np.random.default_rng(8)
        u = 0.65 + 0.12 * rng.random(30)
        v = 0.2 + 0.6 * rng.random(30)
        z = 15.0 + 3.0 * np.sin(8.0 * v) + 0.5 * rng.normal(size=30)
        points = np.column_stack([10.0 * u, 10.0 * v, z])
        parameters = np.column_stack([u, v])

        model = fit_fastfit_surface(
            points,
            parameters=parameters,
            options=FastFitOptions(
                degree=3,
                initial_splits=(1, 1),
                max_depth=0,
                max_error=100.0,
                smoothing=1e-3,
                min_region_fill_ratio=0.85,
                support_grid=8,
                workers=1,
            ),
        )

        self.assertEqual(model.diagnostics["stabilized_sparse_regions"], 1)
        patch = model.patches[0]
        uu, vv = np.meshgrid(
            np.linspace(patch.u_min, patch.u_max, 30),
            np.linspace(patch.v_min, patch.v_max, 30),
            indexing="ij",
        )
        sampled = patch.evaluate(uu.reshape(-1), vv.reshape(-1))
        self.assertLessEqual(float(np.max(sampled[:, 2])), float(np.max(z) + 3.0))
        self.assertGreaterEqual(float(np.min(sampled[:, 2])), float(np.min(z) - 3.0))

    def test_local_knot_export_shapes(self):
        model = fit_fastfit_surface(
            _smooth_test_surface(),
            options=FastFitOptions(degree=3, max_error=1e-10, initial_splits=(1, 1)),
        )
        knots_u, knots_v, points, anchors = model.to_local_knot_surface()
        self.assertEqual(knots_u.shape, (16, 5))
        self.assertEqual(knots_v.shape, (16, 5))
        self.assertEqual(points.shape, (16, 3))
        self.assertEqual(anchors.shape, (16, 2))

    def test_full_multiplicity_export_matches_local_knot_evaluator_interior(self):
        model = fit_fastfit_surface(
            _smooth_test_surface(),
            options=FastFitOptions(degree=3, max_error=1e-10, initial_splits=(1, 1)),
        )
        knots_u, knots_v, points, _ = model.to_local_knot_surface()
        u = np.array([0.2, 0.4, 0.8])
        v = np.array([0.15, 0.55, 0.85])
        piecewise = model.evaluate(u, v)
        local_knot = evaluate_tspline_numpy(
            u,
            v,
            knots_u,
            knots_v,
            points,
            np.ones(points.shape[0]),
            degree=model.degree,
        )
        np.testing.assert_allclose(piecewise, local_knot, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
