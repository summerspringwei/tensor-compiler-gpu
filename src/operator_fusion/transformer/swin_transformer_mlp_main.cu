
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../utils.h"
#include "../../cuda_utils.h"

#include "swin_transformer_mlp.h"
#include "swin_transformer_mlp_tvm.h"

void init_values(half* input, half* weight, half* short_cut){
  hf_init_values<half>(input, {524288}, 1, 0);
  hf_init_values<half>(weight, {1048576}, 1, 0);
  hf_init_values<half>(short_cut, {131072}, 1, 0);
}

#define FUNC_TVM default_function_kernel0<<<dim3(8, 8, 1), dim3(32, 1, 4)>>>(d_input, d_weight, d_output, d_short_cut);
#define FUNC_TVM_V2 fc2_16_16_2048_512_tvm_v2<<<dim3(8, 8, 1), dim3(32, 1, 4)>>>(d_input, d_weight, d_output, d_short_cut);
#define FUNC_TVM_V3 cudaLaunchCooperativeKernel((void*)fc2_16_16_2048_512_tvm_v3, dim3(8, 8, 1), dim3(32, 1, 4), encoder_kernelArgs, 64*1024);
// #define FUNC fc2_16_16_2048_512_v2<<<dim3(16, 16, 1), dim3(32, 4, 1)>>>(d_input, d_weight, d_output, (half*)nullptr);
// #define FUNC fc2_16_16_2048_512_v3<<<dim3(8, 16, 1), dim3(32, 4, 1)>>>(d_input, d_weight, d_output, (half*)nullptr);

int main(){
	const int input_size=524288;	half *input = new half[input_size];
	const int weight_size=1048576;	half *weight = new half[weight_size];
	const int output_size=131072;	half *output = new half[output_size];
  const int short_cut_size=131072;	half *short_cut = new half[short_cut_size];

  init_values(input, weight, short_cut);

	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight=NULL;
	half *d_output=NULL;
  half *d_short_cut=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
	err=cudaMalloc((void **)&d_weight, sizeof(half)*weight_size);
	err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);
  err=cudaMalloc((void **)&d_short_cut, sizeof(half)*short_cut_size);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight, sizeof(half)*weight_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_short_cut, short_cut, sizeof(half)*short_cut_size, cudaMemcpyHostToDevice);
	
  FUNC_TVM
	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
  
  for(int i=0; i<10; ++i){
    printf("(%d, %d): %f\n", i/512, i%512, __half2float(output[i]));
  }
  for(int i=0; i<output_size; ++i){
		if(__half2float(output[i]) != (float)(2048.0)){
			printf("(%d, %d): %f\n", i/512, i%512, __half2float(output[i]));
		}
  }printf("\n");

  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  if(supportsCoopLaunch){
      printf("Device support CoopLaunch\n");
  }
  int numThreads = 64*4, numBlocksPerSm=0; \
  cudaDeviceProp deviceProp; \
  cudaGetDeviceProperties(&deviceProp, dev); \
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void*)fc2_16_16_2048_512_tvm_v3, numThreads, 0); \
  printf("OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);\
  void *encoder_kernelArgs[] = {d_input, d_weight, d_output, d_short_cut};
  
  FUNC_TVM_V3
	cudaDeviceSynchronize();
  cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
  for(int i=0; i<output_size; ++i){
		if( __half2float(output[i]) != 2048){
			printf("(%d, %d): %f\n", i/512, i%512, __half2float(output[i]));
		}
  }printf("\n");

	cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  const int round_cout = 10, loop = 10000;
  float ms = 0, sum = 0;
  // 1. For original pointwise conv
  for(int round =0; round<round_cout; ++round){
    ms = 0, sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      FUNC_TVM
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
      FUNC_TVM_V3
      checkCuda( cudaEventRecord(stopEvent,0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      sum += ms;
    }printf("After fuse avg time %f\n", sum / loop);
  }
  
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  delete[] input;
	delete[] weight;
	delete[] output;
  delete[] short_cut;
	cudaFree(d_input);
	cudaFree(d_weight);
	cudaFree(d_output);
  cudaFree(d_short_cut);
	return 0;
}