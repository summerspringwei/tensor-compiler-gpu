
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <time.h> 

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
  init_values(input, {height, width, in_channels}, 1);
  init_values(weight1, {out_channels_1, in_channels}, 0.01);
  init_values(bias1, {out_channels_1}, 1);
  init_values(weight2, {out_channels_1, out_channels_2}, 0.01);
  init_values(bias2, {out_channels_2}, 1);
}

extern "C" __global__ void __launch_bounds__(256) fused_micro_operators(
  float* input, float* weight1, float* bias1, float* weight2, float* bias2, float* output);

int main() {
  // Declare size
  const int batch = 1;
  const int in_channels = 480, height = 14, width=14;
  const int out_channels_1 = 20, out_channels_2 = 480;
  const int input_size = batch*height*width*in_channels;
  const int weight1_size = in_channels * out_channels_1;
  const int weight2_size = out_channels_1 * out_channels_2;
  const int output_size = out_channels_2;


  // Declare arrays
  float *input = new float[input_size];
  float *intermedia_reduce_mean = new float[in_channels];
  float *weight1 = new float[weight1_size];
  float *bias1 = new float[out_channels_1];
  float *intermedia_output1 = new float[out_channels_1];
  float *weight2 = new float[weight2_size];
  float *bias2 = new float[out_channels_2];
  float *output = new float[output_size];
  float *ori_output = new float[output_size];

  float* d_input = NULL, *d_intermedia_reduce_mean=NULL,
    *d_weight1 = NULL, *d_bias1=NULL, *d_intermedia_output1=NULL,
    *d_weight2=NULL, *d_bias2 = NULL, *d_output = NULL, *d_ori_output = NULL;
  
  // Allocate space on device
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&d_input, sizeof(float)*input_size);
  err = cudaMalloc((void **)&d_intermedia_reduce_mean, sizeof(float)*in_channels);
  err = cudaMalloc((void **)&d_weight1, sizeof(float)*weight1_size);
  err = cudaMalloc((void **)&d_bias1, sizeof(float)*out_channels_1);
  err = cudaMalloc((void **)&d_intermedia_output1, sizeof(float)*out_channels_1);
  err = cudaMalloc((void **)&d_weight2, sizeof(float)*weight2_size);
  err = cudaMalloc((void **)&d_bias2, sizeof(float)*out_channels_2);
  err = cudaMalloc((void **)&d_output, sizeof(float)*output_size);
  err = cudaMalloc((void **)&d_ori_output, sizeof(float)*output_size);

  // Check error
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  // Copy data and convert data type
  init_inputs_and_weights(input, weight1, bias1, weight2, bias2,
    in_channels, height, width, out_channels_1, out_channels_2);
  cudaMemcpy(d_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight1, weight1, sizeof(float)*weight1_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias1, bias1, sizeof(float)*out_channels_1, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight2, weight2, sizeof(float)*weight2_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias2, bias2, sizeof(float)*out_channels_2, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Warm up
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  fused_micro_operators<<<dim3(1, 1, 1), dim3(256, 1, 1)>>>(d_input, d_weight1, d_bias1, d_weight2, d_bias2, d_output);
  cudaDeviceSynchronize();
  // err = cudaMemcpy(ori_output, d_ori_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
  err = cudaMemcpy(output, d_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Benchmark
  const int loop = 100;
  float ms = 0, sum = 0, min = 10000, max=0;
  // 1. For original pointwise conv
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    
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
    fused_micro_operators<<<dim3(1, 1, 1), dim3(256, 1, 1)>>>(d_input, d_weight1, d_bias1, d_weight2, d_bias2, d_output);
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
    fused_micro_operators<<<dim3(1, 1, 1), dim3(256, 1, 1)>>>(d_input, d_weight1, d_bias1, d_weight2, d_bias2, d_output);
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
  for (int oc = 0; oc < out_channels_2; ++oc) {
    printf("%.6f ", output[oc]);
  }printf("\n");
  if(equal){
    printf("Check passed\n");
  }else{
    printf("Check failed\n");
  }
  
  // Free
  cudaFree(d_input);
  cudaFree(d_intermedia_reduce_mean);
  cudaFree(d_weight1);
  cudaFree(d_bias1);
  cudaFree(d_intermedia_output1);
  cudaFree(d_weight2);
  cudaFree(d_bias2);
  cudaFree(d_output);
  cudaFree(d_ori_output);
  delete []input;
  delete []intermedia_reduce_mean;
  delete []weight1;
  delete []bias1;
  delete []intermedia_output1;
  delete []weight2;
  delete []bias2;
  delete []output;
  delete []ori_output;
  return 0;
}
