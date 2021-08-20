#pragma once

#include <torch/extension.h>

at::Tensor affine_transform_dets_cuda_forward(const at::Tensor dets,
                    const at::Tensor trans,
                    int num_classes);


