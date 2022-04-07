

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "fused_pointwise_pointwise.h"
#include "auto_scheduler_codegen/pointwise_112_112_16_32.h"
#include "auto_scheduler_codegen/pointwise_112_112_32_96.h"
#include "auto_scheduler_codegen/pointwise_conv_56_56_144_24.h"
#include "auto_scheduler_codegen/pointwise_conv_56_56_24_144.h"
#include "auto_scheduler_codegen/pointwise_conv_28_28_240_40.h"
#include "auto_scheduler_codegen/pointwise_conv_28_28_40_240.h"

#include "../../utils.h"
#include "../../cuda_utils.h"

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

void init_inputs_and_weights(float* input, float* weight1, float* bias1, float* weight2, float* bias2,
  int in_channels, int height, int width, int out_channels_1, int out_channels_2){
  // init_values(input, {height, width, in_channels}, 1, 1);
  // init_values(weight1, {in_channels, out_channels_1}, 1, 1);
  // init_values(bias1, {out_channels_1}, 1);
  // init_values(weight2, {out_channels_1, out_channels_2}, 1, 1);
  // init_values(bias2, {out_channels_2}, 1);
  init_values(input, {height, width, in_channels}, 1);
  init_values(weight1, {in_channels, out_channels_1}, 1);
  init_values(bias1, {out_channels_1}, 1);
  init_values(weight2, {out_channels_1, out_channels_2}, 1);
  init_values(bias2, {out_channels_2}, 1);
}

// For 112x112x32x16
// const int height = 112, width = 112, in_channels=32, out_channels_1=16, out_channels_2 = 96, block_size = 256, num_blocks = height * width / 32;
// #define FUNC1 pointwise_112_112_16_32<<<dim3(784,1,1),  dim3(32,1,1)>>>(d_input, d_pw_weight1, d_tmp_output);
// #define FUNC2 pointwise_112_112_32_96<<<dim3(784,1,1),  dim3(32,1,1)>>>(d_tmp_output, d_pw_weight2, d_ori_output);
// #define FUSED_FUNC1_FUNC2 fused_pointwise_pointwise<32, out_channels_2, num_blocks, block_size, 16, 16, 8, 32,\
//     height, width, in_channels, out_channels_1, out_channels_2><<<dim3(num_blocks,1,1), dim3(block_size,1,1)>>>(d_input, d_pw_weight1, d_pw_weight2, d_output);

// For 56x56x144x24
// const int height = 56, width = 56, in_channels=144, out_channels_1=24, out_channels_2 = 144, block_size = 16*24, tile_size_x = 32, num_blocks = height * width / tile_size_x;
// #define FUNC1 pointwise_56_56_144_24<<<dim3(196,1,1), dim3(96,1,1)>>>(d_input, d_pw_weight1, d_tmp_output);
// #define FUNC2 pointwise_56_56_24_144<<<dim3(196,1,1), dim3(288,1,1)>>>(d_tmp_output, d_pw_weight2, d_ori_output);
// #define FUSED_FUNC1_FUNC2 fused_pointwise_pointwise<tile_size_x, out_channels_2, num_blocks, block_size, 16, 24, 16, 24,\
//     height, width, in_channels, out_channels_1, out_channels_2><<<dim3(num_blocks,1,1), dim3(block_size,1,1)>>>(d_input, d_pw_weight1, d_pw_weight2, d_output);



const int height = 28, width = 28, in_channels=240, out_channels_1=40, out_channels_2 = 240, block_size = 4*64, tile_size_x = 8, num_blocks = height * width / tile_size_x;
#define FUNC1 pointwise_28_28_240_40<<<dim3(196,1,1), dim3(80,1,1)>>>(d_input, d_pw_weight1, d_tmp_output);
#define FUNC2 pointwise_28_28_40_240<<<dim3(196,1,1), dim3(60,1,1)>>>(d_tmp_output, d_pw_weight2, d_ori_output);
#define FUSED_FUNC1_FUNC2 fused_pointwise_pointwise<tile_size_x, out_channels_2, num_blocks, block_size, 8, 32, 4, 64,\
    height, width, in_channels, out_channels_1, out_channels_2><<<dim3(num_blocks,1,1), dim3(block_size,1,1)>>>(d_input, d_pw_weight1, d_pw_weight2, d_output);
// #define FUSED_FUNC1_FUNC2 {cudaLaunchCooperativeKernel((void*)fused_pointwise_pointwise<tile_size_x, out_channels_2, num_blocks, block_size, 4, 40, 4, 40,\
//     height, width, in_channels, out_channels_1, out_channels_2>, dim3(49,1,1), dim3(block_size,1,1), fused_kernelArgs, 64*1024);};

int main() {

  // const int kernel_height = 1, kernel_width = 1;
  const int input_size = in_channels * height * width;
  const int weight1_size = in_channels * out_channels_1;
  const int weight2_size = out_channels_1 * out_channels_2;
  const int tmp_output_size = height * width * out_channels_1;
  const int output_size = height * width * out_channels_2;

  // Declare arrays
  float *input = new float[input_size];
  float *pw_weight1 = new float[weight1_size];
  float *bias1 = new float[out_channels_1];
  float *pw_weight2 = new float[weight2_size];
  float* bias2 = new float[out_channels_2];
  float *tmp_output = new float[tmp_output_size];
  float *output = new float[output_size];
  float *cpu_output = new float[output_size];
  float* d_input = NULL, *d_pw_weight1 = NULL, *d_pw_weight2=NULL, *d_tmp_output=NULL, *d_output = NULL, *d_ori_output=NULL;
  

  // Allocate space on device
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&d_input, sizeof(float)*input_size);
  err = cudaMalloc((void **)&d_pw_weight1, sizeof(float)*weight1_size);
  err = cudaMalloc((void **)&d_pw_weight2, sizeof(float)*weight2_size);
  err = cudaMalloc((void **)&d_tmp_output, sizeof(float)*tmp_output_size);
  err = cudaMalloc((void **)&d_output, sizeof(float)*output_size);
  err = cudaMalloc((void **)&d_ori_output, sizeof(float)*output_size);

  // Check error
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  // Copy data and convert data type
  init_inputs_and_weights(input, pw_weight1, bias1, pw_weight2, bias2, in_channels, height, width, out_channels_1, out_channels_2);

  // CPU implementation to check result
  pointwise_conv(input, pw_weight1, tmp_output, height, width, in_channels, out_channels_1);
  pointwise_conv(tmp_output, pw_weight2, cpu_output, height, width, out_channels_1, out_channels_2);

  cudaMemcpy(d_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pw_weight1, pw_weight1, sizeof(float)*weight1_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pw_weight2, pw_weight2, sizeof(float)*weight2_size, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  
  void *fused_kernelArgs[] = { (void *)&(d_input), (void *)&(d_pw_weight1), (void *)&(d_pw_weight2), (void *)&(d_output) };
  int dev = 0;
  int numThreads = 240, numBlocksPerSm=0; \
  cudaDeviceProp deviceProp; \
  cudaGetDeviceProperties(&deviceProp, dev); \
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, fused_pointwise_pointwise<tile_size_x, out_channels_2, num_blocks, block_size, 4, 40, 4, 40,\
    height, width, in_channels, out_channels_1, out_channels_2>, numThreads, 0); \
  printf("OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);\
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  // FUNC1
  // FUNC2
  FUSED_FUNC1_FUNC2
  err = cudaMemcpy(output, d_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  bool equal = check_equal(output, cpu_output, height, width, out_channels_2);

  // Benchmark
  const int round_cout = 2, loop = 10000;
  float ms = 0, sum = 0;
  // 1. For original pointwise conv
  for(int round =0; round<round_cout; ++round){
    ms = 0, sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      FUNC1
      FUNC2
      checkCuda( cudaEventRecord(stopEvent,0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      sum += ms;
    }printf("Before fuse avg time %f\n", sum / loop);
  }
  
  for(int round =0; round<round_cout; ++round){
    ms = 0, sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      FUSED_FUNC1_FUNC2
      checkCuda( cudaEventRecord(stopEvent,0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      sum += ms;
    }printf("After fuse avg time %f\n", sum / loop);
  }

  // Free
  cudaFree(d_input);
  cudaFree(d_pw_weight1);
  cudaFree(d_pw_weight2);
  cudaFree(d_output);
  cudaFree(d_ori_output);
  cudaFree(d_tmp_output);
  delete[] input;
  delete[] pw_weight1;
  delete[] pw_weight2;
  delete[] output;
  delete[] cpu_output;
  return 0;
}
