

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

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


extern "C" __global__ void __launch_bounds__(128) fused_pointwise_112_112_144_6_6_144(float* __restrict__ input, float* __restrict__ weight, 
  float* __restrict__ weight2, float* __restrict__ final_output);
// grid=(196,1,1),  block=(128,1,1)
extern "C" __global__ void __launch_bounds__(128) pointwise_112_112_144_6(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output);
//dim3(196,1,1),  dim3(576,1,1)
extern "C" __global__ void __launch_bounds__(576) pointwise_112_112_6_144_v2(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output);

int main() {
  // Declare size
  const int height = 112;
  const int width = 112;
  const int in_channel=144;
  const int out_channel=6;
  const int kernel_height = 3, kernel_width = 3;
  const int input_size = in_channel * height * width;
  const int pw_weight_size = in_channel * out_channel;
  const int tmp_output_size = height * width * out_channel;
  const int output_size = in_channel * height * width;

  // Declare arrays
  float *input = new float[input_size];
  float *pw_weight1 = new float[pw_weight_size];
  float *pw_weight2 = new float[pw_weight_size];
  float *tmp_output = new float[tmp_output_size];
  float *output = new float[output_size];
  float *cpu_output = new float[output_size];
  float* d_input = NULL, *d_pw_weight1 = NULL, *d_pw_weight2=NULL, *d_tmp_output=NULL, *d_output = NULL;
  

  // Allocate space on device
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&d_input, sizeof(float)*input_size);
  err = cudaMalloc((void **)&d_pw_weight1, sizeof(float)*pw_weight_size);
  err = cudaMalloc((void **)&d_pw_weight2, sizeof(float)*pw_weight_size);
  err = cudaMalloc((void **)&d_tmp_output, sizeof(float)*tmp_output_size);
  err = cudaMalloc((void **)&d_output, sizeof(float)*output_size);

  // Check error
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  // Copy data and convert data type
  init_conv_conv_fusion_data(input, pw_weight1, pw_weight2, output, height, width, 1, 1, in_channel, out_channel, 1, 1, out_channel, in_channel);
  cudaMemcpy(d_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pw_weight1, pw_weight1, sizeof(float)*pw_weight_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pw_weight2, pw_weight2, sizeof(float)*pw_weight_size, cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();

  // Warm up
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  dim3 threadsPerBlock(128, 1, 1);
  dim3 numBlocks(196, 1, 1);  
  fused_pointwise_112_112_144_6_6_144<<<numBlocks, threadsPerBlock>>>(d_input, d_pw_weight1, d_pw_weight2, d_output);
  // pointwise_112_112_144_6<<<dim3(196,1,1),  dim3(128,1,1)>>>(d_input, d_pw_weight1, d_tmp_output);
  // pointwise_112_112_6_144_v2<<<dim3(196,1,1),  dim3(576,1,1)>>>(d_tmp_output, d_pw_weight2, d_output);
  err = cudaMemcpy(output, d_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Benchmark
  const int loop = 10000;
  // 1. For original pointwise conv
  float ms = 0, sum = 0;
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    pointwise_112_112_144_6<<<dim3(196,1,1),  dim3(128,1,1)>>>(d_input, d_pw_weight1, d_tmp_output);
    pointwise_112_112_6_144_v2<<<dim3(196,1,1),  dim3(576,1,1)>>>(d_tmp_output, d_pw_weight2, d_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
  }printf("Before fuse avg time %f\n", sum / loop);
  sum = 0;
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    pointwise_112_112_144_6<<<dim3(196,1,1),  dim3(128,1,1)>>>(d_input, d_pw_weight1, d_tmp_output);
    pointwise_112_112_6_144_v2<<<dim3(196,1,1),  dim3(576,1,1)>>>(d_tmp_output, d_pw_weight2, d_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
  }printf("Before fuse avg time %f\n", sum / loop);
  
  ms = 0, sum = 0;
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    fused_pointwise_112_112_144_6_6_144<<<numBlocks, threadsPerBlock>>>(d_input, d_pw_weight1, d_pw_weight2, d_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
  }printf("After fuse avg time %f\n", sum / loop);
  sum = 0;
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    fused_pointwise_112_112_144_6_6_144<<<numBlocks, threadsPerBlock>>>(d_input, d_pw_weight1, d_pw_weight2, d_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
  }printf("After fuse avg time %f\n", sum / loop);
  
  
  // Print result
  printf("outputs:->\n");
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int oc = 0; oc < out_channel; ++oc) {
        int idx = h*width*out_channel + w*out_channel+oc;
        printf("%.2f ", output[idx]);
        // if(std::abs(output[idx] - cpu_output[idx]) > 0.1 ){
        //   printf("<%d, %d, %d> %.2f, %.2f\n",h, w, oc, output[idx], cpu_output[idx]);
        // }
      };
    }printf("\n");
  }


  // Free
  cudaFree(d_input);
  cudaFree(d_pw_weight1);
  cudaFree(d_pw_weight2);
  cudaFree(d_output);
  delete[] input;
  delete[] pw_weight1;
  delete[] pw_weight2;
  delete[] output;
  delete[] cpu_output;
  return 0;
}
