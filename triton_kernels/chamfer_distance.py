"""Triton Chamfer-distance prototype."""

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
    def _one_sided_kernel(
        source,
        target,
        out,
        num_source: tl.constexpr,
        num_target: tl.constexpr,
        block_source: tl.constexpr,
        block_target: tl.constexpr,
    ):
        src_offsets = tl.program_id(0) * block_source + tl.arange(0, block_source)
        src_mask = src_offsets < num_source
        sx = tl.load(source + src_offsets * 3 + 0, mask=src_mask, other=0.0)
        sy = tl.load(source + src_offsets * 3 + 1, mask=src_mask, other=0.0)
        sz = tl.load(source + src_offsets * 3 + 2, mask=src_mask, other=0.0)
        best = tl.full((block_source,), float("inf"), dtype=tl.float32)

        for start in range(0, num_target, block_target):
            tgt_offsets = start + tl.arange(0, block_target)
            tgt_mask = tgt_offsets < num_target
            tx = tl.load(target + tgt_offsets * 3 + 0, mask=tgt_mask, other=0.0)
            ty = tl.load(target + tgt_offsets * 3 + 1, mask=tgt_mask, other=0.0)
            tz = tl.load(target + tgt_offsets * 3 + 2, mask=tgt_mask, other=0.0)
            dx = sx[:, None] - tx[None, :]
            dy = sy[:, None] - ty[None, :]
            dz = sz[:, None] - tz[None, :]
            dist = dx * dx + dy * dy + dz * dz
            dist = tl.where(tgt_mask[None, :], dist, float("inf"))
            best = tl.minimum(best, tl.min(dist, axis=1))

        tl.store(out + src_offsets, best, mask=src_mask)


def chamfer_distance_triton(
    source,
    target,
    *,
    block_source: int = 64,
    block_target: int = 64,
):
    """Symmetric squared Chamfer distance using Triton and torch reductions."""

    _require_triton()
    import torch

    if not source.is_cuda or not target.is_cuda:
        raise ValueError("source and target must be CUDA tensors")
    if source.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError("source and target must be float32 tensors")
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("source and target must have shape (N, 3) and (M, 3)")

    source = source.contiguous()
    target = target.contiguous()
    src_out = torch.empty((source.shape[0],), device=source.device, dtype=source.dtype)
    tgt_out = torch.empty((target.shape[0],), device=target.device, dtype=target.dtype)

    _one_sided_kernel[(triton.cdiv(source.shape[0], block_source),)](
        source,
        target,
        src_out,
        source.shape[0],
        target.shape[0],
        block_source,
        block_target,
        num_warps=4,
    )
    _one_sided_kernel[(triton.cdiv(target.shape[0], block_source),)](
        target,
        source,
        tgt_out,
        target.shape[0],
        source.shape[0],
        block_source,
        block_target,
        num_warps=4,
    )
    return src_out.mean() + tgt_out.mean()

