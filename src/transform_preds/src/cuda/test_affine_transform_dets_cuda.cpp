
#include <stdio.h>
#include <assert.h>

#include <chrono>
#include <vector>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

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

    cudaFree(d_target_dets);
    cudaFree(d_dets);
    cudaFree(d_trans);
}

void benchmark_affine_transform_dets_cuda(){
    const size_t n = 100, batch=1;
    std::vector<float> dets(n * batch), target_dets(n * batch);
    std::vector<float> trans = {4.000000, -0.000000, -16.000000, -0.000000, 4.000000, -16.000000};
    
    float* d_target_dets = nullptr;
    float* d_dets = nullptr;
    float* d_trans = nullptr;
    cudaMalloc((void**)&d_target_dets, sizeof(float) * 6);
    cudaMalloc((void**)&d_dets, sizeof(float) * 6);
    cudaMalloc((void**)&d_trans, sizeof(float) * 6);
    // Warm up
    for (int i_=0; i_<5; i_++)
    {
        cudaMemcpy(d_dets, dets.data(), sizeof(float) * 6 * n * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_trans, trans.data(), sizeof(float) * 6 * n * batch, cudaMemcpyHostToDevice);
        affine_transform_dets_cuda(d_target_dets, d_dets, d_trans, 1.0, batch, n);
    }
    //GPU time measurement
    float ms_max = std::numeric_limits<float>::min();
    float ms_min = std::numeric_limits<float>::max();
    float ms_total, ms_i;
    cudaEvent_t start_i, stop_i;
    cudaEventCreate(&start_i);
    cudaEventCreate(&stop_i);
    
    //time measurement
    ms_total = 0;
    int steps = 10000;
    cudaProfilerStart();
    for (int i_=0; i_<steps; i_++)
    {
        
        cudaMemcpy(d_dets, dets.data(), sizeof(float) * 6 * n * batch, cudaMemcpyHostToDevice);
        cudaMemcpy(d_trans, trans.data(), sizeof(float) * 6 * n * batch, cudaMemcpyHostToDevice);
        cudaEventRecord(start_i, 0);
        affine_transform_dets_cuda(d_target_dets, d_dets, d_trans, 1.0, batch, n);
        cudaEventRecord(stop_i, 0);
        cudaEventSynchronize(stop_i);
        cudaEventElapsedTime(&ms_i, start_i, stop_i);
        printf("Iteration time %f ms\n", ms_i);
        ms_total += ms_i;
        if (ms_i > ms_max)  ms_max = ms_i;
        if (ms_i < ms_min) ms_min = ms_i;
    }
    cudaProfilerStop();
    cudaDeviceSynchronize();
    printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total / steps);
    cudaFree(d_target_dets);
    cudaFree(d_dets);
    cudaFree(d_trans);
}



int main(int argc, char** argv){
    test_affine_transform_dets_cuda();
    benchmark_affine_transform_dets_cuda();
    return 0;
}
