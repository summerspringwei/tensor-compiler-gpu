
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

// dim3(196,1,1), dim3(288,1,1)
extern "C" __global__ void __launch_bounds__(288) pointwise_56_56_24_144(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output);

// dim3(504,1,1), dim3(128,1,1)
extern "C" __global__ void __launch_bounds__(128) depthwise_56_56_144_s11(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ DepthwiseConv2d);

extern "C" __global__ void __launch_bounds__(128) fused_pointwise_56_56_24_144_depthwise_56_56_144_s11(
  float* __restrict__ input,  float* __restrict__ pointwise_weight, float* __restrict__ depthwise_weight, float* __restrict__ DepthwiseConv2d);

// dim3(112, 0, 0), dim3(256, 0, 0)
extern "C" __global__ void __launch_bounds__(256) fused_pointwise_depthwise(
  float* __restrict__ input,  float* __restrict__ pointwise_weight, float* __restrict__ depthwise_weight, float* __restrict__ DepthwiseConv2d);

int main() {
  // Declare size
  const int height = 56;
  const int width = 56;
  const int in_channel=24;
  const int out_channel=144;
  const int kernel_height = 3, kernel_width = 3;
  const int input_size = in_channel * height * width;
  const int pw_weight_size = in_channel * out_channel;
  const int dw_weight_size = kernel_height * kernel_width * out_channel;
  const int output_size = out_channel * height * width;
  const int intermedia_size = height*width*out_channel;

  // Declare arrays
  float *input = new float[input_size];
  float *pw_weight = new float[pw_weight_size];
  float *dw_weight = new float[dw_weight_size];
  float *output = new float[output_size];
  float *ori_output = new float[output_size];
  float *intermedia_output = new float[intermedia_size];
  float *cpu_output = new float[output_size];
  float* d_input = NULL, *d_pw_weight = NULL, *d_dw_weight=NULL, *d_intermedia_output=NULL, *d_output = NULL, *d_ori_output = NULL;
  

  // Allocate space on device
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&d_input, sizeof(float)*input_size);
  err = cudaMalloc((void **)&d_pw_weight, sizeof(float)*pw_weight_size);
  err = cudaMalloc((void **)&d_dw_weight, sizeof(float)*dw_weight_size);
  err = cudaMalloc((void **)&d_output, sizeof(float)*output_size);
  err = cudaMalloc((void **)&d_ori_output, sizeof(float)*output_size);
  err = cudaMalloc((void **)&d_intermedia_output, sizeof(float)*intermedia_size);
  // Check error
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  // Copy data and convert data type
  init_conv_conv_fusion_data(input, pw_weight, dw_weight, output, \
    height, width, 1, 1, in_channel, out_channel, 3, 3, out_channel, 1);
  cudaMemcpy(d_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pw_weight, pw_weight, sizeof(float)*pw_weight_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dw_weight, dw_weight, sizeof(float)*dw_weight_size, cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();

  // Warm up
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  pointwise_56_56_24_144<<<dim3(196,1,1), dim3(288,1,1)>>>(d_input, d_pw_weight, d_intermedia_output);
  depthwise_56_56_144_s11<<<dim3(504,1,1), dim3(128,1,1)>>>(d_intermedia_output, d_dw_weight, d_ori_output);
  // fused_pointwise_depthwise<<<dim3(112,1,1), dim3(256,1,1)>>>(d_input, d_pw_weight, d_dw_weight, d_output);
  // fused_pointwise_56_56_24_144_depthwise_56_56_144_s11<<<dim3(504,1,1), dim3(128,1,1)>>>(d_input, d_pw_weight, d_dw_weight, d_output);
  fused_pointwise_depthwise<<<dim3(392,1,1), dim3(256,1,1)>>>(d_input, d_pw_weight, d_dw_weight, d_output);
  cudaDeviceSynchronize();
  err = cudaMemcpy(ori_output, d_ori_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
  err = cudaMemcpy(output, d_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Benchmark
  const int loop = 10000;
  float ms = 0, sum = 0, min = 10000, max=0;
  // 1. For original pointwise conv
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    pointwise_56_56_24_144<<<dim3(196,1,1), dim3(288,1,1)>>>(d_input, d_pw_weight, d_intermedia_output);
    depthwise_56_56_144_s11<<<dim3(504,1,1), dim3(128,1,1)>>>(d_intermedia_output, d_dw_weight, d_ori_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
    ms > max? max=ms: 0;
    ms < min? min=ms: 0;
  }printf("Before fuse avg time %f, min %f, max %f\n", sum / loop, min, max);
  sum = 0, min = 10000, max=0;;
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    pointwise_56_56_24_144<<<dim3(196,1,1), dim3(288,1,1)>>>(d_input, d_pw_weight, d_intermedia_output);
    depthwise_56_56_144_s11<<<dim3(504,1,1), dim3(128,1,1)>>>(d_intermedia_output, d_dw_weight, d_ori_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
    ms > max? max=ms: 0;
    ms < min? min=ms: 0;
  }printf("Before fuse avg time %f, min %f, max %f\n", sum / loop, min, max);
  
  ms = 0, sum = 0, min = 10000, max=0;
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    // fused_pointwise_56_56_24_144_depthwise_56_56_144_s11<<<dim3(504,1,1), dim3(128,1,1)>>>(d_input, d_pw_weight, d_dw_weight, d_output);
    fused_pointwise_depthwise<<<dim3(112,1,1), dim3(256,1,1)>>>(d_input, d_pw_weight, d_dw_weight, d_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
    ms > max? max=ms: 0;
    ms < min? min=ms: 0;
  }printf("After fuse avg time %f, min %f, max %f\n", sum / loop, min, max);
  sum = 0;
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    // fused_pointwise_56_56_24_144_depthwise_56_56_144_s11<<<dim3(504,1,1), dim3(128,1,1)>>>(d_input, d_pw_weight, d_dw_weight, d_output);
    fused_pointwise_depthwise<<<dim3(112,1,1), dim3(256,1,1)>>>(d_input, d_pw_weight, d_dw_weight, d_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;  
    ms > max? max=ms: 0;
    ms < min? min=ms: 0;
  }printf("After fuse avg time %f, min %f, max %f\n", sum / loop, min, max);
  
  
  // Print result
  printf("outputs:->\n");
  bool equal = true;
  for (int oc = 0; oc < out_channel; ++oc) {
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
        int idx = h*width*out_channel + w*out_channel+oc;
        // printf("%.2f ", output[idx]);
        if(std::abs(output[idx] - ori_output[idx]) > 0.1 ){
          printf("<%d, %d, %d> %.2f, %.2f\n",h, w, oc, output[idx], ori_output[idx]);
          equal = false;
        }
      }printf("\n");
    } printf("\n");
  }printf("\n");
  if(equal){
    printf("Check passed\n");
  }else{
    printf("Check failed\n");
  }
  
  // Free
  cudaFree(d_input);
  cudaFree(d_pw_weight);
  cudaFree(d_dw_weight);
  cudaFree(d_output);
  cudaFree(d_ori_output);
  cudaFree(d_intermedia_output);
  delete[] input;
  delete[] pw_weight;
  delete[] dw_weight;
  delete[] output;
  delete[] cpu_output;
  return 0;
}
