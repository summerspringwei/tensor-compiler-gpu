#include <iostream>
#include <list>
#include <vector>
#include <chrono>
#include <cstring>
#include <map>
#include <assert.h>

#include <cuda_runtime.h>

#include "affine_transform_cuda.hpp"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


const int block_size = 256;
__global__ void affine_transform_kernel(float* target_coords, float* coords, float* trans, int batch, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch * n){
        return;
    }
    float x = coords[2 * idx];
    float y = coords[2 * idx + 1];
    target_coords[2 * idx] = trans[0] * x + trans[1] * y + trans[2] * 1;
    target_coords[2 * idx + 1] = trans[3] * x + trans[4] * y + trans[5] * 1;
}


/**
 * @brief 
 * 
 * @param coords shape (n, 2)
 * @param center shape (2)
 * @param scale shape(2)
 * @param output_size shape(2)
 */
void affine_transform(float* target_coords, float* coords, float* trans, int batch, int n, int loop_count){
    const int k = 2;
    // coords shape (b, n, 2)
    float* d_coords = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&d_coords, sizeof(float)*n*k*batch);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_coords (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // trans shape (2, 3), save in opencv
    float *d_trans = NULL;
    err = cudaMalloc((void **)&d_trans, sizeof(float) * 2 * 3);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_trans (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // target_cords shape (b, 100, 3), save in opencv
    float *d_target_coords = NULL;
    err = cudaMalloc((void **)&d_target_coords, sizeof(float)*n*k*batch);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector d_target_coords (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_coords, coords, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_trans, trans, sizeof(float)*2*3, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    float ms = 0.0;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    
    auto t1 = std::chrono::steady_clock::now();
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(batch * n / threadsPerBlock.x + 1);
    // Launch the arnold CUDA Kernel
    checkCuda( cudaEventRecord(startEvent,0));
    for(int i=0; i<loop_count; ++i){
        affine_transform_kernel<<<numBlocks, threadsPerBlock>>>(d_target_coords, d_coords, d_trans, batch, n);
        checkCuda( cudaEventRecord(stopEvent,0));
    }
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    cudaDeviceSynchronize();

    auto t2 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::micro>(t2-t1).count();
    printf("cudaEvent latency avg %f\n", ms / loop_count);
    // printf("[%f, %f] bandwidth %f\n", latency, ms,  n*sizeof(float) * 1e3 /1024/1024/ms);
    err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(target_coords, d_target_coords, sizeof(float) *n*k*batch, cudaMemcpyDeviceToHost);

    cudaFree(d_coords);
    cudaFree(d_trans);
    cudaFree(d_target_coords);
}

