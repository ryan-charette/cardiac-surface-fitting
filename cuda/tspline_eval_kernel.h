#pragma once

#include <torch/extension.h>

torch::Tensor tspline_eval_cuda(
    torch::Tensor u,
    torch::Tensor v,
    torch::Tensor knots_u,
    torch::Tensor knots_v,
    torch::Tensor control_points,
    torch::Tensor weights);

