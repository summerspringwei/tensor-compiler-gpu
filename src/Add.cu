#include <iostream>
#include <list>
#include <vector>
#include <chrono>
#include <cstring>
#include <map>
#include <assert.h>

#include <cuda_runtime.h>

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


__global__ void add(float* d_a, float* d_b, float* d_c, const size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n ){
        // __shared__ s_a[64];
        d_c[idx] =  d_a[idx] + d_b[idx];
  }
}
#define tile 16

__global__ void add_v2(float* d_a, float* d_b, float* d_c, const size_t n){
//    const int tile = 64;
    const int block_base = blockIdx.x * blockDim.x * tile;
    for(int i=0; i<tile; ++i){
        int idx = block_base + threadIdx.x + i * blockDim.x;
        d_c[idx] =  d_a[idx] + d_b[idx];
    }
}


void add_cuda(float* a, float* b, float* c, const size_t n){
    // Error code to check return values for CUDA calls
    float* d_a = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&d_a, sizeof(float)*n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *d_b = NULL;
    err = cudaMalloc((void **)&d_b, sizeof(float)*n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_a, a, sizeof(float)*n, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_b, b, sizeof(float)*n, cudaMemcpyHostToDevice);
    float *d_c = NULL;
    err = cudaMalloc((void **)&d_c, sizeof(float)*n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_c (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    auto t1 = std::chrono::steady_clock::now();
    cudaEvent_t startEvent, stopEvent;
    float ms = 0.0;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    // Launch the arnold CUDA Kernel
    dim3 threadsPerBlock(128);
    dim3 numBlocks(n / threadsPerBlock.x / tile);
    //dim3 numBlocks(n / threadsPerBlock.x + 1);
    checkCuda( cudaEventRecord(startEvent,0) );
    add_v2<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

    cudaDeviceSynchronize();

    auto t2 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::micro>(t2-t1).count();
    printf("[%f, %f] bandwidth %f\n", latency, ms, n*3*sizeof(float) *1e3 /1024/1024/ms );
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(c, d_c, sizeof(float)*n, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


int main(int argc, char** argv)
{
    if(argc < 2){
        printf("Usage: EncryptionTime img_size loop_count\n");
        return 0;
    }
    
    int n = atoi(argv[1]);
    int loop_count = atoi(argv[2]);
    assert((loop_count>0) && (n>0));
    float* a = (float*)malloc(sizeof(float) * n*n);
    float* b = (float*)malloc(sizeof(float) * n*n);
    float* c = (float*)malloc(sizeof(float) * n*n);
    for(int i=0; i<n*n;++i){
        a[i] = (float)i;
        b[i] = (float)-1;
    }
    auto t1 = std::chrono::steady_clock::now();
    for(int i=0; i<loop_count; ++i){
        add_cuda(a, b, c, n*n);
    }
    
    for(int i=0; i<128;++i){
        printf("%f ", c[i]);
    }printf("\n");
    auto t2 = std::chrono::steady_clock::now();
    printf("%d %d %f\n", n, loop_count, std::chrono::duration<double, std::micro>(t2-t1).count() / loop_count);
   
    return 0;
}

