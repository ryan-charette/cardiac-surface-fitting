"""Triton forward kernel for degree-2 rational T-spline evaluation."""

from __future__ import annotations


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def _require_triton():
    if triton is None or tl is None:
        raise ImportError("Triton is required: pip install -e .[triton]")


if triton is not None:

    @triton.jit
    def _basis_degree2(x, k0, k1, k2, k3):
        n0 = ((k0 <= x) & (x < k1)).to(tl.float32)
        n1 = ((k1 <= x) & (x < k2)).to(tl.float32)
        n2 = ((k2 <= x) & (x < k3)).to(tl.float32)

        d10 = k1 - k0
        d11 = k2 - k1
        d12 = k3 - k2
        a0 = tl.where(d10 != 0.0, ((x - k0) / d10) * n0, 0.0)
        b0 = tl.where(d11 != 0.0, ((k2 - x) / d11) * n1, 0.0)
        n01 = a0 + b0

        a1 = tl.where(d11 != 0.0, ((x - k1) / d11) * n1, 0.0)
        b1 = tl.where(d12 != 0.0, ((k3 - x) / d12) * n2, 0.0)
        n11 = a1 + b1

        d20 = k2 - k0
        d21 = k3 - k1
        a = tl.where(d20 != 0.0, ((x - k0) / d20) * n01, 0.0)
        b = tl.where(d21 != 0.0, ((k3 - x) / d21) * n11, 0.0)
        return a + b

    @triton.jit
    def _tspline_eval_degree2_kernel(
        u,
        v,
        knots_u,
        knots_v,
        control_points,
        weights,
        output,
        num_u: tl.constexpr,
        num_v: tl.constexpr,
        num_control_points: tl.constexpr,
        knot_width: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < (num_u * num_v)
        iu = offsets // num_v
        iv = offsets - iu * num_v
        u_value = tl.load(u + iu, mask=mask, other=0.0)
        v_value = tl.load(v + iv, mask=mask, other=0.0)

        nx = tl.zeros((block_size,), dtype=tl.float32)
        ny = tl.zeros((block_size,), dtype=tl.float32)
        nz = tl.zeros((block_size,), dtype=tl.float32)
        denominator = tl.zeros((block_size,), dtype=tl.float32)

        for cp in tl.static_range(0, num_control_points):
            ku = knots_u + cp * knot_width
            kv = knots_v + cp * knot_width
            bu = _basis_degree2(
                u_value,
                tl.load(ku + 0),
                tl.load(ku + 1),
                tl.load(ku + 2),
                tl.load(ku + 3),
            )
            bv = _basis_degree2(
                v_value,
                tl.load(kv + 0),
                tl.load(kv + 1),
                tl.load(kv + 2),
                tl.load(kv + 3),
            )
            wb = tl.load(weights + cp) * bu * bv
            denominator += wb
            nx += wb * tl.load(control_points + cp * 3 + 0)
            ny += wb * tl.load(control_points + cp * 3 + 1)
            nz += wb * tl.load(control_points + cp * 3 + 2)

        base = offsets * 3
        tl.store(output + base + 0, nx / denominator, mask=mask)
        tl.store(output + base + 1, ny / denominator, mask=mask)
        tl.store(output + base + 2, nz / denominator, mask=mask)


def evaluate_tspline_triton(
    u,
    v,
    knots_u,
    knots_v,
    control_points,
    weights,
    *,
    block_size: int = 128,
):
    """Evaluate a degree-2 T-spline using Triton.

    Inputs must be CUDA torch tensors with dtype ``torch.float32``.
    """

    _require_triton()
    import torch

    if knots_u.shape[1] != 4 or knots_v.shape[1] != 4:
        raise ValueError("the Triton prototype currently specializes degree=2")
    tensors = [u, v, knots_u, knots_v, control_points, weights]
    if not all(t.is_cuda for t in tensors):
        raise ValueError("all inputs must be CUDA tensors")
    if not all(t.dtype == torch.float32 for t in tensors):
        raise ValueError("all inputs must be torch.float32")

    u = u.contiguous()
    v = v.contiguous()
    knots_u = knots_u.contiguous()
    knots_v = knots_v.contiguous()
    control_points = control_points.contiguous()
    weights = weights.contiguous()
    output = torch.empty((u.numel(), v.numel(), 3), device=u.device, dtype=u.dtype)
    total = u.numel() * v.numel()
    grid = (triton.cdiv(total, block_size),)
    _tspline_eval_degree2_kernel[grid](
        u,
        v,
        knots_u,
        knots_v,
        control_points,
        weights,
        output,
        u.numel(),
        v.numel(),
        control_points.shape[0],
        knots_u.shape[1],
        block_size,
        num_warps=4,
    )
    return output.reshape(u.numel(), v.numel(), 3)

