#pragma once

// #include <torch/extension.h>
#include "transform_preds_wrapper_cuda.hpp"

at::Tensor transform_preds_forward(const at::Tensor &dets,
                    const at::Tensor &center,
                    const at::Tensor &scale,
                    const at::Tensor &output_size,
                    int num_classes){
    return transform_preds_cuda_forward(dets,
                    center,
                    scale,
                    output_size,
                    num_classes);
}