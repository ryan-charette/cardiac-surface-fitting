import unittest

import numpy as np

from cardiac_geometry.geometry.point_cloud import chamfer_distance_numpy


class ChamferCorrectnessTests(unittest.TestCase):
    def test_identical_clouds_have_zero_distance(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertAlmostEqual(chamfer_distance_numpy(points, points), 0.0)

    def test_known_shifted_single_point_distance(self):
        source = np.array([[0.0, 0.0, 0.0]])
        target = np.array([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(chamfer_distance_numpy(source, target), 2.0)

    def test_chunked_matches_unchunked(self):
        rng = np.random.default_rng(123)
        source = rng.normal(size=(17, 3))
        target = rng.normal(size=(11, 3))
        chunked = chamfer_distance_numpy(source, target, chunk_size=4)
        full = chamfer_distance_numpy(source, target, chunk_size=1000)
        self.assertAlmostEqual(chunked, full)


if __name__ == "__main__":
    unittest.main()

