#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

cudaError_t checkCuda(cudaError_t result, char* msg)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "In %s CUDA Runtime Error: %s\n", msg, cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "In CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void print_cuda_kernel(float* input, long n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx != 0){
        return;
    }
    for(long i=0; i<n; ++i){
        printf("%f ", input[i]);
    }printf("\n");
}

void print_cuda(float* input, long n){
    int block_size = 32;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(1);
    print_cuda_kernel<<<numBlocks, threadsPerBlock>>>(input, n);
    checkCuda(cudaDeviceSynchronize(), "print_cuda");
}
