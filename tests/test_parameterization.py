import unittest

import numpy as np

from cardiac_geometry.geometry.parameterization import (
    _seam_angle,
    pca_parameterize,
    spherical_parameterize,
)


class ParameterizationTests(unittest.TestCase):
    def test_pca_parameterization_is_unit_square(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.5],
                [1.0, 1.0, 1.0],
            ]
        )
        uv = pca_parameterize(points)
        self.assertEqual(uv.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(uv)))
        self.assertGreaterEqual(float(uv.min()), 0.0)
        self.assertLessEqual(float(uv.max()), 1.0)

    def test_largest_gap_seam_avoids_dense_arc(self):
        theta = np.deg2rad(np.array([30.0, 60.0, 90.0, 120.0, 150.0]))
        seam = _seam_angle(theta, mode="largest_gap")
        shifted = np.mod(theta - seam, 2.0 * np.pi) / (2.0 * np.pi)
        self.assertAlmostEqual(float(np.rad2deg(seam)), 270.0)
        self.assertLess(float(np.ptp(shifted)), 0.5)

    def test_spherical_parameterization_is_unit_square(self):
        theta = np.linspace(0.0, 1.5 * np.pi, 12)
        points = np.column_stack(
            [3.0 * np.cos(theta), 2.0 * np.sin(theta), np.linspace(-1.0, 1.0, 12)]
        )
        uv = spherical_parameterize(points, pole_axis=2)
        self.assertEqual(uv.shape, (12, 2))
        self.assertTrue(np.all(np.isfinite(uv)))
        self.assertGreaterEqual(float(uv.min()), 0.0)
        self.assertLessEqual(float(uv.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
