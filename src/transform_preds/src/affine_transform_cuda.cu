#include <iostream>
#include <list>
#include <vector>
#include <chrono>
#include <cstring>
#include <map>
#include <assert.h>

#include <cuda_runtime.h>


#include "affine_transform_cuda.hpp"
#include "print_cuda.hpp"

__global__ void copy_dets_with_slice_kernel(float* coords, float* dets, int batch, int n, int slice_from, int slice_to){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch * n){
        return;
    }
    const int stride = slice_to - slice_from;
    for(int i=slice_from; i<slice_to; ++i){
        coords[stride * idx + i] = dets[6 * idx + i];
    }
}

void copy_dets_with_slice_cuda(float* d_coords, float* d_dets, int batch, int n, int slice_from, int slice_to){
    const int block_size = 128;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(batch * n / threadsPerBlock.x + 1);
    copy_dets_with_slice_kernel<<<numBlocks, threadsPerBlock>>>(d_coords, d_dets, batch, n, slice_from, slice_to);
}


__global__ void affine_transform_dets_kernel(float* d_target_dets, float* dets, float* trans, int batch, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch * n){
        return;
    }
    int stride = 6;
    // Do slice from 0 to 2
    float x = dets[stride * idx];
    float y = dets[stride * idx + 1];
    printf("idx %d\n", stride * idx);
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
    printf("dets: %f %f %f %f %f %f\n", 
        dets[stride * idx], dets[stride * idx + 1], dets[stride * idx + 2],
        dets[stride * idx + 3], dets[stride * idx + 4], dets[stride * idx + 5]);
    printf("target_dets: %f %f %f %f %f %f\n", 
    d_target_dets[stride * idx], d_target_dets[stride * idx + 1], d_target_dets[stride * idx + 2],
    d_target_dets[stride * idx + 3], d_target_dets[stride * idx + 4], d_target_dets[stride * idx + 5]);
}


void affine_transform_dets_cuda(float* target_dets, float* dets, float* trans, int batch, int n){
    const int block_size = 128;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(batch * n / threadsPerBlock.x + 1);
    affine_transform_dets_kernel<<<numBlocks, threadsPerBlock>>>(target_dets, dets, trans, batch, n);
    auto err = cudaDeviceSynchronize();

}



__global__ void affine_transform_dets_kernel(torch::PackedTensorAccessor64<float, 1> d_target_dets, 
    torch::PackedTensorAccessor64<float, 1> dets, float* trans, int batch, int n){
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
    printf("dets: %f %f %f %f %f %f\n", 
        dets[stride * idx], dets[stride * idx + 1], dets[stride * idx + 2],
        dets[stride * idx + 3], dets[stride * idx + 4], dets[stride * idx + 5]);
    printf("target_dets: %f %f %f %f %f %f\n", 
    d_target_dets[stride * idx], d_target_dets[stride * idx + 1], d_target_dets[stride * idx + 2],
    d_target_dets[stride * idx + 3], d_target_dets[stride * idx + 4], d_target_dets[stride * idx + 5]);
}


void affine_transform_dets_cuda(torch::PackedTensorAccessor64<float, 1> target_dets,
    torch::PackedTensorAccessor64<float, 1> dets, float* trans, int batch, int n){
    const int block_size = 128;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(batch * n / threadsPerBlock.x + 1);
    affine_transform_dets_kernel<<<numBlocks, threadsPerBlock>>>(target_dets, dets, trans, batch, n);
    checkCuda(cudaDeviceSynchronize(), "affine_transform_dets_cuda");
}


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


void affine_transform_cuda(float* d_target_coords, float* d_coords, float* d_trans, int batch, int n){
    const int block_size = 256;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(batch * n / threadsPerBlock.x + 1);
    affine_transform_kernel<<<numBlocks, threadsPerBlock>>>(d_target_coords, d_coords, d_trans, batch, n);
    cudaDeviceSynchronize();
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
    
    // Launch the arnold CUDA Kernel
    checkCuda( cudaEventRecord(startEvent,0));
    for(int i=0; i<loop_count; ++i){
        affine_transform_cuda(d_target_coords, d_coords, d_trans, batch, n);
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

