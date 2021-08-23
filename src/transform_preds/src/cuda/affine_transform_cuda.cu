
#include "affine_transform_cuda.hpp"

#include <assert.h>

#include <iostream>

#include <cuda_runtime.h>
#include "print_cuda.hpp"


// Each block process n boxes
__global__ void affine_transform_dets_kernel(float* d_target_dets, float* dets, float* trans, 
    float scale, int batch, int n){
    int batch_idx = blockIdx.x;
    int n_offset = threadIdx.x;
    if(batch_idx >= batch || n_offset >= n){
        return;
    }
    int stride = 6;
    int offset = (batch_idx * n + n_offset) * stride;
    
    // Do slice from 0 to 2
    float x = dets[offset];
    float y = dets[offset + 1];
    d_target_dets[offset] = (trans[0] * x + trans[1] * y + trans[2] * 1) / scale;
    d_target_dets[offset + 1] = (trans[3] * x + trans[4] * y + trans[5] * 1) / scale;
    // Do slice from 2 to 4
    x = dets[offset + 2];
    y = dets[offset + 3];
    d_target_dets[offset + 2] = (trans[0] * x + trans[1] * y + trans[2] * 1) / scale;
    d_target_dets[offset + 3] = (trans[3] * x + trans[4] * y + trans[5] * 1) / scale;
    // Copy rest
    d_target_dets[offset + 4] = dets[offset + 4];
    d_target_dets[offset + 5] = dets[offset + 5];
    #ifdef DEBUG
        printf("dets: %f %f %f %f %f %f\n", 
            dets[offset], dets[offset + 1], dets[offset + 2],
            dets[offset + 3], dets[offset + 4], dets[offset + 5]);
        printf("target_dets: %f %f %f %f %f %f\n", 
            d_target_dets[offset], d_target_dets[offset + 1], d_target_dets[offset + 2],
            d_target_dets[offset + 3], d_target_dets[offset + 4], d_target_dets[offset + 5]);
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
void affine_transform_dets_cuda(float* target_dets, float* dets, float* trans, 
    float scale, int batch, int n){
    const int block_size = ((n / 32) + 1) * 32;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(batch);
    affine_transform_dets_kernel<<<numBlocks, threadsPerBlock>>>(target_dets, dets, trans, scale, batch, n);
    auto err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
}
