#pragma once

#include <torch/extension.h>

torch::Tensor chamfer_distance_cuda(torch::Tensor source, torch::Tensor target);

