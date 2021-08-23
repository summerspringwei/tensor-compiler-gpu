#pragma once

#include "cuda/affine_transform_cuda_torch_wrapper.hpp"
#include "cpu/get_affine_transform_cpu_torch_wrapper.hpp"

at::Tensor affine_transform_dets_forward(const at::Tensor dets,
                    const at::Tensor trans,
                    int num_classes, float scale){
    if(dets.is_cuda() && trans.is_cuda()){
        return affine_transform_dets_cuda_forward(dets,
                        trans,
                        num_classes, scale);
    }else{
        throw ("affine_transform_dets_forward does not support CPU\n");
    }
}


at::Tensor get_affine_transform_forward(at::Tensor center, at::Tensor scale, 
    float rot, at::Tensor output_size, at::Tensor shift, int inv){
    if(center.is_cpu() && scale.is_cpu() &&
        output_size.is_cpu() && shift.is_cpu()){
        return get_affine_transform_cpu_forward(center, scale, rot, output_size, shift, inv);
    }else{
        throw ("get_affine_transform_forward does not support CUDA\n");
    }
}
