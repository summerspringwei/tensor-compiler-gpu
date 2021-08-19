#pragma once

#include <torch/extension.h>

at::Tensor transform_preds_cuda_forward(const at::Tensor dets,
                    const at::Tensor &center,
                    const at::Tensor &scale,
                    const at::Tensor &output_size,
                    int num_classes);