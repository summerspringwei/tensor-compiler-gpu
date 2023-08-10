
#include "cuda_fp16.h"
#include "../cuda_kernel_utils.h"
#include <stdio.h>

#define kBlockSize 256
#define kGridSize 84  // The number of SM on RTX3090 is 84
// #include "kernels/vector_matrix_mul.cu"

struct __align__(16) MyAlignedStruct {
    half a[8];
};

__global__ void __launch_bounds__(kBlockSize)
    vector_matrix_mul_kernel(half *__restrict__ input,
                             half *__restrict__ weight,
                             half *__restrict__ output) {
  const int warpIdx = threadIdx.x / 32;
  const int laneIdx = threadIdx.x % 32;
  const int numWarp = kBlockSize / 32;
  const int vectorLength = sizeof(float4) / sizeof(half);
//   half local_input[8];
//   half local_weight.a[8];
    MyAlignedStruct local_input;
    MyAlignedStruct local_weight;
  const int64_t batch_size = 1;
  const int64_t reduce_dim = 1280;
  const int64_t out_dim = 5120;
  // Iterate over batch_size
  for (int64_t b = 0; b < batch_size; ++b) {
    // Iterate over out_dim
    for (int64_t idx = 0; UPDIV(out_dim, kGridSize * numWarp); ++idx) {
      // Each warp reduce one reduce_dim
      float local_sum = 0;
      const int64_t weight_row_idx =
          (idx * kGridSize * numWarp + blockIdx.x * numWarp + warpIdx);
      // Guard against over indexing
      if (weight_row_idx >= out_dim) break;
#pragma unroll
      for (int64_t k = 0; k < reduce_dim; k += (warpSize * vectorLength)) {
        const int64_t col_idx = k + laneIdx * vectorLength;
        // Guard against over indexing
        if (col_idx >= reduce_dim) break;
        *(float4 *)&local_input =
            *((float4 *)&(input[(b * reduce_dim + col_idx)]));
        *(float4 *)&local_weight =
            *((float4 *)&(weight[(weight_row_idx * reduce_dim + col_idx)]));
        float2 tmp;
        tmp = __half22float2(__hmul2(half2(local_input.a[0], local_input.a[1]), half2(local_weight.a[0], local_weight.a[1])));
        local_sum += (tmp.x + tmp.y);
        tmp = __half22float2(__hmul2(half2(local_input.a[2], local_input.a[3]), half2(local_weight.a[2], local_weight.a[3])));
        local_sum += (tmp.x + tmp.y);
        tmp = __half22float2(__hmul2(half2(local_input.a[4], local_input.a[5]), half2(local_weight.a[4], local_weight.a[5])));
        local_sum += (tmp.x + tmp.y);
        tmp = __half22float2(__hmul2(half2(local_input.a[6], local_input.a[7]), half2(local_weight.a[6], local_weight.a[7])));
        local_sum += (tmp.x + tmp.y);
      }
      // Reduce within warp
      local_sum = warpReduceSum(local_sum);
      // Write to output
      if (laneIdx == 0) {
        output[b * out_dim + weight_row_idx] = __float2half(local_sum);
      }
    }
  }
}


int main(int argc, char* argv[]) {
    half* input = (half*)malloc(1280 * sizeof(half));
    half* weight = (half*)malloc(1280 * 5120 * sizeof(half));
    half* output = (half*)malloc(5120 * sizeof(half));

    half* d_input;
    half* d_weight;
    half* d_output;
    cudaMalloc(&d_input, 1280 * sizeof(half));
    cudaMalloc(&d_weight, 1280 * 5120 * sizeof(half));
    cudaMalloc(&d_output, 5120 * sizeof(half));

    for(int i = 0; i < 1280; i++){
        input[i] = __float2half(1.0);
    }
    for(int i = 0; i < 1280 * 5120; i++){
        weight[i] = __float2half(0.125);
    }
    for(int i = 0; i < 5120; i++){
        output[i] = __float2half(0.0);
    }

    cudaMemcpy(d_input, input, 1280 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, 1280 * 5120 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, 5120 * sizeof(half), cudaMemcpyHostToDevice);

    void* args[] = {
    (void**)&d_input, (void**)&d_weight, (void**)&d_output
  };
  cudaLaunchKernel((void*)vector_matrix_mul_kernel, dim3(kGridSize, 1, 1), 
    dim3(kBlockSize, 1, 1), args);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, 5120 * sizeof(half), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5120; i++){
        printf("%f ", __half2float(output[i]));
    }
}