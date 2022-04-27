#include <stdio.h>
#include <cuda_runtime.h>

#include "../../utils.h"
#include "../../cuda_utils.h"

#include "swin_transformer_query_key_softmax.h"

void init_input_weight(half* input, half* weight, half* output){
  hf_init_values<half>(input, {64*4, 64, 32}, __float2half(.1));
  hf_init_values<half>(weight, {64*4, 64, 32}, __float2half(.1));
  hf_init_values<half>(output, {64*4, 64, 32}, __float2half(0.0));
}

#define FUNC1 swin_transformer_query_key<<<dim3(1, 1, 256), dim3(32, 1, 1)>>>(d_input, d_weight, d_intermedia_output);
#define FUNC2 fused_mul_softmax<64, 4, 64, 32><<<dim3(64*4, 1, 1), dim3(64, 1, 1)>>>(d_intermedia_output, d_ori_output);

#define FUSED_FUNC fused_swin_transformer_query_key_mul_softmax<<<dim3(1, 1, 256), dim3(32, 1, 1)>>>(d_input, d_weight, d_output);

int main(){
  // batch, num_heads, seq_length, dim = 64, 4, 64, 32
	half *input = new half[524288];
	half *weight = new half[524288];
	half *output = new half[1048576];
	half *intermedia_output = new half[1048576];
	half *ori_output = new half[1048576];

  init_input_weight(input, weight, output);

	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight=NULL;
	half *d_output=NULL;
	half *d_intermedia_output=NULL;
	half *d_ori_output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*524288);
	err=cudaMalloc((void **)&d_weight, sizeof(half)*524288);
	err=cudaMalloc((void **)&d_output, sizeof(half)*1048576);
	err=cudaMalloc((void **)&d_intermedia_output, sizeof(half)*1048576);
	err=cudaMalloc((void **)&d_ori_output, sizeof(half)*1048576);

	cudaMemcpy(d_input, input, sizeof(half)*524288, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight, sizeof(half)*524288, cudaMemcpyHostToDevice);
  FUNC1
  FUNC2
  FUSED_FUNC
	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*1048576, cudaMemcpyDeviceToHost);
	cudaMemcpy(intermedia_output, d_intermedia_output, sizeof(half)*1048576, cudaMemcpyDeviceToHost);
	cudaMemcpy(ori_output, d_ori_output, sizeof(half)*1048576, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  // Benchmark
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
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
      FUSED_FUNC
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
  for(int i=0;i<10;++i){
    printf("%.6f ", __half2float(output[i]));
  }printf("\n");
  for(int i=0;i<10;++i){
    printf("%.6f ", __half2float(ori_output[i]));
  }printf("\n");

  delete[] input;
	delete[] weight;
	delete[] output;
	delete[] intermedia_output;
	delete[] ori_output;
	cudaFree(d_input);
	cudaFree(d_weight);
	cudaFree(d_output);
	cudaFree(d_intermedia_output);
	cudaFree(d_ori_output);
	return 0;
}
