import unittest

import numpy as np

from cardiac_geometry.fastfit import FastFitOptions, fit_fastfit_surface

try:
    import torch
except ImportError:
    torch = None


def _smooth_test_surface(num_u=17, num_v=15):
    u = np.linspace(0.0, 1.0, num_u)
    v = np.linspace(0.0, 1.0, num_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    xyz = np.empty(uu.shape + (3,), dtype=np.float64)
    xyz[..., 0] = uu
    xyz[..., 1] = vv
    xyz[..., 2] = 0.25 * uu * uu - 0.5 * uu * vv + 0.1 * vv
    return xyz


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchFastFitSurfaceTests(unittest.TestCase):
    def test_torch_evaluation_matches_numpy_surface(self):
        model = fit_fastfit_surface(
            _smooth_test_surface(),
            options=FastFitOptions(
                degree=3,
                max_error=1.0,
                initial_splits=(2, 1),
                max_depth=0,
            ),
        )
        torch_surface = model.to_torch(dtype=torch.float64)
        u = np.linspace(0.0, 1.0, 9)
        v = np.linspace(0.0, 1.0, 7)

        expected = model.evaluate(u, v)
        actual = torch_surface.evaluate(
            torch.tensor(u, dtype=torch.float64),
            torch.tensor(v, dtype=torch.float64),
        )

        np.testing.assert_allclose(
            actual.detach().cpu().numpy(), expected, atol=1e-12, rtol=0.0
        )

    def test_torch_evaluation_has_parameter_and_control_gradients(self):
        model = fit_fastfit_surface(
            _smooth_test_surface(),
            options=FastFitOptions(
                degree=3,
                max_error=1.0,
                initial_splits=(2, 1),
                max_depth=0,
            ),
        )
        torch_surface = model.to_torch(dtype=torch.float64)
        torch_surface.control_points.requires_grad_(True)
        params = torch.tensor(
            [[0.25, 0.4], [0.75, 0.6]],
            dtype=torch.float64,
            requires_grad=True,
        )

        loss = torch_surface.evaluate_points(params).square().sum()
        loss.backward()

        self.assertIsNotNone(params.grad)
        self.assertIsNotNone(torch_surface.control_points.grad)
        self.assertTrue(bool(torch.all(torch.isfinite(params.grad)).detach().cpu()))
        self.assertTrue(
            bool(torch.all(torch.isfinite(torch_surface.control_points.grad)).detach().cpu())
        )


if __name__ == "__main__":
    unittest.main()

