
#include <vector>
#include "torch/all.h"

auto options_fp16 = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_fp16_cpu = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCPU, 0)
    .requires_grad(false);
  auto options_fp32 = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_int64 = torch::TensorOptions()
    .dtype(torch::kInt64)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_fp32_cpu = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCPU, 0)
    .requires_grad(false);


void my_compare(torch::Tensor& a, torch::Tensor& b, float rotl, float aotl, int print_detail=0);