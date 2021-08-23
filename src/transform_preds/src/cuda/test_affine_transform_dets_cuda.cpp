
#include <stdio.h>
#include <assert.h>

#include <chrono>
#include <vector>
#include <memory>

#include <cuda_runtime.h>

#include "affine_transform_cuda.hpp"


void test_affine_transform_dets_cuda(){
    float target_dets[6], dets[6] = {78.93826f, 163.65175f, 78.93826f, 163.65175f, 78.93826f, 163.65175f};
    float trans[6] = {4.000000, -0.000000, -16.000000, -0.000000, 4.000000, -16.000000};
    int batch=1, n=1;
    float* d_target_dets = nullptr;
    float* d_dets = nullptr;
    float* d_trans = nullptr;
    cudaMalloc((void**)&d_target_dets, sizeof(float) * 6);
    cudaMalloc((void**)&d_dets, sizeof(float) * 6);
    cudaMalloc((void**)&d_trans, sizeof(float) * 6);
    cudaMemcpy(d_dets, dets, sizeof(float) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_trans, trans, sizeof(float) * 6, cudaMemcpyHostToDevice);
    affine_transform_dets_cuda(d_target_dets, d_dets, d_trans, 1.0, batch, n);
    cudaDeviceSynchronize();
}


int main(int argc, char** argv){
    test_affine_transform_dets_cuda();
    return 0;
}
