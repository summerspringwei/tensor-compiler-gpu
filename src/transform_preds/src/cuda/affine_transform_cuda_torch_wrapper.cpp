
#include "affine_transform_cuda_torch_wrapper.hpp"

#include <cuda_runtime.h>

#include "affine_transform_cuda.hpp"
#include "print_cuda.hpp"


at::Tensor affine_transform_dets_cuda_forward(const at::Tensor dets,
                    const at::Tensor trans,
                    int num_classes){
    int batch = dets.size(0);
    int n = dets.size(1);
    int last_dim = dets.size(2);
    auto target_dets = torch::ones({batch, n, last_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    affine_transform_dets_cuda(target_dets.data_ptr<float>(), dets.data_ptr<float>(), trans.data_ptr<float>(), batch, n);
    #ifdef DEBUG
        print_cuda(target_dets.data_ptr<float>(), long(batch * n * last_dim));
    #endif
    return target_dets;
}

