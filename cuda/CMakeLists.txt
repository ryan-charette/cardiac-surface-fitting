cmake_minimum_required(VERSION 3.20)
project(cardiac_geometry_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Torch REQUIRED)

add_library(
  cardiac_geometry_cuda SHARED
  bindings.cpp
  tspline_eval_kernel.cu
  chamfer_distance_kernel.cu
)
target_link_libraries(cardiac_geometry_cuda PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(cardiac_geometry_cuda PRIVATE "${TORCH_INCLUDE_DIRS}")

