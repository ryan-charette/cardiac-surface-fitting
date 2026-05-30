#include <torch/extension.h>

#include "chamfer_distance_kernel.h"
#include "tspline_eval_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tspline_eval", &tspline_eval_cuda, "Fused rational T-spline evaluation (CUDA)");
  m.def("chamfer_distance", &chamfer_distance_cuda, "Symmetric Chamfer distance (CUDA)");
}

