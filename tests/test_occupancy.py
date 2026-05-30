import unittest

import numpy as np

from cardiac_geometry.geometry.occupancy import build_parameter_occupancy


class OccupancyTests(unittest.TestCase):
    def test_frame_detects_internal_hole(self):
        values = np.linspace(0.25, 0.75, 60)
        bottom = np.column_stack([values, np.full_like(values, 0.25)])
        top = np.column_stack([values, np.full_like(values, 0.75)])
        left = np.column_stack([np.full_like(values, 0.25), values])
        right = np.column_stack([np.full_like(values, 0.75), values])
        params = np.vstack([bottom, top, left, right])

        occupancy = build_parameter_occupancy(params, resolution=64, dilation=2)

        self.assertFalse(bool(occupancy.classify_supported(np.array([[0.5, 0.5]]))[0]))
        self.assertTrue(bool(occupancy.holes[32, 32]))
        self.assertTrue(bool(occupancy.classify_supported(np.array([[0.25, 0.5]]))[0]))
        self.assertGreater(int(np.count_nonzero(occupancy.boundary)), 0)
        self.assertTrue(bool(occupancy.exterior[0, 0]))


if __name__ == "__main__":
    unittest.main()
