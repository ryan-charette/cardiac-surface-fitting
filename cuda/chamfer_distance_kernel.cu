#include "chamfer_distance_kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

__global__ void one_sided_chamfer_kernel(
    const float* __restrict__ source,
    const float* __restrict__ target,
    float* __restrict__ min_distances,
    int num_source,
    int num_target) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_source) {
    return;
  }

  const float sx = source[idx * 3 + 0];
  const float sy = source[idx * 3 + 1];
  const float sz = source[idx * 3 + 2];
  float best = CUDART_INF_F;

  for (int j = 0; j < num_target; ++j) {
    const float dx = sx - target[j * 3 + 0];
    const float dy = sy - target[j * 3 + 1];
    const float dz = sz - target[j * 3 + 2];
    const float dist = dx * dx + dy * dy + dz * dz;
    best = fminf(best, dist);
  }

  min_distances[idx] = best;
}

}  // namespace

torch::Tensor chamfer_distance_cuda(torch::Tensor source, torch::Tensor target) {
  TORCH_CHECK(source.is_cuda(), "source must be CUDA");
  TORCH_CHECK(target.is_cuda(), "target must be CUDA");
  TORCH_CHECK(source.scalar_type() == torch::kFloat32, "source must be float32");
  TORCH_CHECK(target.scalar_type() == torch::kFloat32, "target must be float32");
  TORCH_CHECK(source.dim() == 2 && source.size(1) == 3, "source must have shape (N, 3)");
  TORCH_CHECK(target.dim() == 2 && target.size(1) == 3, "target must have shape (M, 3)");

  source = source.contiguous();
  target = target.contiguous();
  auto src_min = torch::empty({source.size(0)}, source.options());
  auto tgt_min = torch::empty({target.size(0)}, target.options());

  const int threads = 256;
  const int src_blocks = (static_cast<int>(source.size(0)) + threads - 1) / threads;
  const int tgt_blocks = (static_cast<int>(target.size(0)) + threads - 1) / threads;
  one_sided_chamfer_kernel<<<src_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      source.data_ptr<float>(),
      target.data_ptr<float>(),
      src_min.data_ptr<float>(),
      static_cast<int>(source.size(0)),
      static_cast<int>(target.size(0)));
  one_sided_chamfer_kernel<<<tgt_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      target.data_ptr<float>(),
      source.data_ptr<float>(),
      tgt_min.data_ptr<float>(),
      static_cast<int>(target.size(0)),
      static_cast<int>(source.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return src_min.mean() + tgt_min.mean();
}

