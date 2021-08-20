
#include "affine_transform_cuda.hpp"

#include <assert.h>

#include <iostream>

#include <cuda_runtime.h>
#include "print_cuda.hpp"


__global__ void affine_transform_dets_kernel(float* d_target_dets, float* dets, float* trans, int batch, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch * n){
        return;
    }
    int stride = 6;
    // Do slice from 0 to 2
    float x = dets[stride * idx];
    float y = dets[stride * idx + 1];
    d_target_dets[stride * idx] = trans[0] * x + trans[1] * y + trans[2] * 1;
    d_target_dets[stride * idx + 1] = trans[3] * x + trans[4] * y + trans[5] * 1;
    // Do slice from 2 to 4
    x = dets[stride * idx + 2];
    y = dets[stride * idx + 3];
    d_target_dets[stride * idx + 2] = trans[0] * x + trans[1] * y + trans[2] * 1;
    d_target_dets[stride * idx + 3] = trans[3] * x + trans[4] * y + trans[5] * 1;
    // Copy rest
    d_target_dets[stride * idx + 4] = dets[stride * idx + 4];
    d_target_dets[stride * idx + 5] = dets[stride * idx + 5];
    #ifdef DEBUG
        printf("dets: %f %f %f %f %f %f\n", 
            dets[stride * idx], dets[stride * idx + 1], dets[stride * idx + 2],
            dets[stride * idx + 3], dets[stride * idx + 4], dets[stride * idx + 5]);
        printf("target_dets: %f %f %f %f %f %f\n", 
            d_target_dets[stride * idx], d_target_dets[stride * idx + 1], d_target_dets[stride * idx + 2],
            d_target_dets[stride * idx + 3], d_target_dets[stride * idx + 4], d_target_dets[stride * idx + 5]);
    #endif
}

/**
 * @brief 
 * 
 * @param target_dets Transformed detections with the same shape as det
 * @param dets Original detections with shape (batch, n, 6)
 * @param trans transformation matrix with shape (2, 3)
 * @param batch 
 * @param n Number of boxes in one image
 */
void affine_transform_dets_cuda(float* target_dets, float* dets, float* trans, int batch, int n){
    const int block_size = 128;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(batch * n / threadsPerBlock.x + 1);
    affine_transform_dets_kernel<<<numBlocks, threadsPerBlock>>>(target_dets, dets, trans, batch, n);
    auto err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
}
