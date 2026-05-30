#include "tspline_eval_kernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

constexpr int kMaxDegree = 8;

__device__ float bspline_basis_local(float x, const float* knots, int degree) {
  if (degree < 0 || degree > kMaxDegree) {
    return NAN;
  }

  if (x == knots[degree + 1]) {
    x = nextafterf(x, -CUDART_INF_F);
  }

  float values[kMaxDegree + 1];
  #pragma unroll
  for (int i = 0; i <= kMaxDegree; ++i) {
    values[i] = 0.0f;
  }

  for (int i = 0; i <= degree; ++i) {
    values[i] = (knots[i] <= x && x < knots[i + 1]) ? 1.0f : 0.0f;
  }

  for (int p = 1; p <= degree; ++p) {
    float next_values[kMaxDegree + 1];
    #pragma unroll
    for (int i = 0; i <= kMaxDegree; ++i) {
      next_values[i] = 0.0f;
    }

    for (int i = 0; i <= degree - p; ++i) {
      const float left_den = knots[i + p] - knots[i];
      const float right_den = knots[i + p + 1] - knots[i + 1];

      float left = 0.0f;
      if (left_den != 0.0f) {
        left = ((x - knots[i]) / left_den) * values[i];
      }

      float right = 0.0f;
      if (right_den != 0.0f) {
        right = ((knots[i + p + 1] - x) / right_den) * values[i + 1];
      }

      next_values[i] = left + right;
    }

    for (int i = 0; i <= degree - p; ++i) {
      values[i] = next_values[i];
    }
  }

  return values[0];
}

__global__ void tspline_eval_kernel(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ knots_u,
    const float* __restrict__ knots_v,
    const float* __restrict__ control_points,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int num_u,
    int num_v,
    int num_control_points,
    int knot_width) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_u * num_v;
  if (sample >= total) {
    return;
  }

  const int iu = sample / num_v;
  const int iv = sample - iu * num_v;
  const float u_value = u[iu];
  const float v_value = v[iv];
  const int degree = knot_width - 2;

  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  float denominator = 0.0f;

  for (int cp = 0; cp < num_control_points; ++cp) {
    const float bu = bspline_basis_local(u_value, knots_u + cp * knot_width, degree);
    const float bv = bspline_basis_local(v_value, knots_v + cp * knot_width, degree);
    const float wb = weights[cp] * bu * bv;
    denominator += wb;
    nx += wb * control_points[cp * 3 + 0];
    ny += wb * control_points[cp * 3 + 1];
    nz += wb * control_points[cp * 3 + 2];
  }

  const int out = sample * 3;
  if (denominator == 0.0f) {
    output[out + 0] = NAN;
    output[out + 1] = NAN;
    output[out + 2] = NAN;
    return;
  }

  output[out + 0] = nx / denominator;
  output[out + 1] = ny / denominator;
  output[out + 2] = nz / denominator;
}

}  // namespace

torch::Tensor tspline_eval_cuda(
    torch::Tensor u,
    torch::Tensor v,
    torch::Tensor knots_u,
    torch::Tensor knots_v,
    torch::Tensor control_points,
    torch::Tensor weights) {
  TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(knots_u.is_cuda(), "knots_u must be a CUDA tensor");
  TORCH_CHECK(knots_v.is_cuda(), "knots_v must be a CUDA tensor");
  TORCH_CHECK(control_points.is_cuda(), "control_points must be a CUDA tensor");
  TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
  TORCH_CHECK(u.scalar_type() == torch::kFloat32, "u must be float32");
  TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
  TORCH_CHECK(knots_u.scalar_type() == torch::kFloat32, "knots_u must be float32");
  TORCH_CHECK(knots_v.scalar_type() == torch::kFloat32, "knots_v must be float32");
  TORCH_CHECK(control_points.scalar_type() == torch::kFloat32, "control_points must be float32");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
  TORCH_CHECK(u.dim() == 1 && v.dim() == 1, "u and v must be 1D tensors");
  TORCH_CHECK(knots_u.dim() == 2 && knots_v.dim() == 2, "knots must be 2D tensors");
  TORCH_CHECK(control_points.dim() == 2 && control_points.size(1) == 3,
              "control_points must have shape (C, 3)");
  TORCH_CHECK(weights.dim() == 1, "weights must be 1D");
  TORCH_CHECK(knots_u.size(0) == knots_v.size(0), "knot row counts must match");
  TORCH_CHECK(knots_u.size(0) == control_points.size(0), "control point count must match knots");
  TORCH_CHECK(weights.size(0) == control_points.size(0), "weight count must match control points");
  TORCH_CHECK(knots_u.size(1) == knots_v.size(1), "knot widths must match");
  TORCH_CHECK(knots_u.size(1) - 2 <= kMaxDegree, "degree exceeds CUDA kernel maximum");

  u = u.contiguous();
  v = v.contiguous();
  knots_u = knots_u.contiguous();
  knots_v = knots_v.contiguous();
  control_points = control_points.contiguous();
  weights = weights.contiguous();

  auto output = torch::empty({u.size(0), v.size(0), 3}, control_points.options());
  const int total = static_cast<int>(u.size(0) * v.size(0));
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  tspline_eval_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      u.data_ptr<float>(),
      v.data_ptr<float>(),
      knots_u.data_ptr<float>(),
      knots_v.data_ptr<float>(),
      control_points.data_ptr<float>(),
      weights.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(u.size(0)),
      static_cast<int>(v.size(0)),
      static_cast<int>(control_points.size(0)),
      static_cast<int>(knots_u.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

