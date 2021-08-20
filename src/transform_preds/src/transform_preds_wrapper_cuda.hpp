#pragma once

#include <torch/extension.h>

at::Tensor transform_preds_cuda_forward(const at::Tensor dets,
                    const at::Tensor &center,
                    const at::Tensor &scale,
                    const at::Tensor &output_size,
                    int num_classes);

at::Tensor affine_transform_dets_cuda_forward(const at::Tensor dets,
                    const at::Tensor trans,
                    int num_classes);

at::Tensor get_affine_transform_cpu_forward(const at::Tensor center, 
                    const at::Tensor scale, float rot, const at::Tensor output_size, 
                    const at::Tensor shift, int inv);
