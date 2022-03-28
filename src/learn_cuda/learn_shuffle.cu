
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../utils.h"

extern "C" __global__ void __launch_bounds__(128) test_condition(float* input, float* output){
  int new_i =((threadIdx.x % 2) == 0? 1: -2);
  // input[threadIdx.x] = new_i;
  output[threadIdx.x] = new_i;
}

extern "C" __global__ void __launch_bounds__(128) test_shuffle(float* input, float* output){
  // input[threadIdx.x] = __shfl_xor_sync(0xffffffff, input[threadIdx.x], 2);
  float val = input[threadIdx.x];
  for(int i=warpSize/2; i>=1; i=i/2){
    val += __shfl_down_sync(0xffffffff, val, i);
  }
  output[threadIdx.x] = val;
}

extern "C" __global__ void __launch_bounds__(128) test_vector(float* input, float* output){
  float4 tmp = reinterpret_cast<float4*>(input)[threadIdx.x * 4];
  

  output[threadIdx.x] = input[threadIdx.x];
}

int main(){
  int input_size = 128;
  int output_size = 128;
  float* input = new float[input_size];
  float* output = new float[output_size];

  for(int i=0; i<input_size; ++i){
    input[i] = 1;
  }
  float* d_input=nullptr, *d_output=nullptr;
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&d_input, sizeof(float)*input_size);
  err = cudaMalloc((void **)&d_output, sizeof(float)*output_size);
  cudaMemcpy(d_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
  // test_condition<<<dim3(1, 1, 1), dim3(128, 1, 1)>>>(d_input, d_output);
  test_shuffle<<<dim3(1, 1, 1), dim3(128, 1, 1)>>>(d_input, d_output);
  cudaMemcpy(output, d_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
  for(int i=0; i<output_size; ++i){
    printf("%.2f ", output[i]);
  }
  printf("\n");
  test_init_values();
  cudaFree(d_input);
  cudaFree(d_output);
  delete[] input;
  delete[] output;
  return 0;
}