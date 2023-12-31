
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include "../../utils.h"
#include "../../cuda_utils.h"

#include "fused_fc_fc.h"
#include "fused_fc_fc_opt.h"

void init_values(half* input, half* weight1, half* weight2, half* weight3, half* weight4, half* output){
  hf_init_values<half>(input, {16777216}, 1, 0);
  hf_init_values<half>(weight1, {65536}, 1, 0);
	hf_init_values<half>(weight2, {65536}, 1, 0);
	hf_init_values<half>(weight3, {65536}, 1, 0);
	hf_init_values<half>(weight4, {65536}, 1, 0);
  hf_init_values<half>(output, {16777216}, 1, 0);
}

// #define FUNC fused_fc_fc<<<dim3(108*4, 1, 1), dim3(32, 4, 1)>>>(d_input, d_weight1, d_weight2, d_weight3, d_weight4, d_output);
// #define FUNC fused_fc_fc_v2<<<dim3(108*4, 1, 1), dim3(32, 4, 1)>>>(d_input, d_weight1, d_weight2, d_weight3, d_weight4, d_output);
// #define FUNC fused_fc_fc_v3<<<dim3(108*4, 1, 1), dim3(32, 8, 1)>>>(d_input, d_weight1, d_weight2, d_weight3, d_weight4, d_output);
// #define FUNC fused_fc_fc_v4<<<dim3(108*4, 1, 1), dim3(32, 8, 1)>>>(d_input, d_weight1, d_weight2, d_weight3, d_weight4, d_output);
// #define FUNC fused_fc_fc_v5<<<dim3(108*4, 1, 1), dim3(32, 8, 1)>>>(d_input, d_weight1, d_weight2, d_weight3, d_weight4, d_output);
#define FUNC fused_fc_fc_v4_reuse_input_shared<<<dim3(108*4, 1, 1), dim3(32, 8, 1)>>>(d_input, d_weight1, d_weight2, d_weight3, d_weight4, d_output);
// #define FUNC tvm_matmul<<<dim3(54, 2, 1), dim3(32, 4, 2)>>>(d_input, d_weight1, d_output);


int main(){
	int m = 108 * 4 * 16, n = 256, k = 256;
	const int input_size=16777216*2;	half *input = new half[input_size];
	const int weight1_size=65536;	half *weight1 = new half[weight1_size];
	const int weight2_size=65536;	half *weight2 = new half[weight2_size];
	const int weight3_size=65536;	half *weight3 = new half[weight3_size];
	const int weight4_size=65536;	half *weight4 = new half[weight4_size];
	const int output_size=input_size;	half *output = new half[output_size];

	init_values(input, weight1, weight2, weight3, weight4, output);

	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight1=NULL;
	half *d_weight2=NULL;
	half *d_weight3=NULL;
	half *d_weight4=NULL;
	half *d_output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
	err=cudaMalloc((void **)&d_weight1, sizeof(half)*weight1_size);
	err=cudaMalloc((void **)&d_weight2, sizeof(half)*weight2_size);
	err=cudaMalloc((void **)&d_weight3, sizeof(half)*weight3_size);
	err=cudaMalloc((void **)&d_weight4, sizeof(half)*weight4_size);
	err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight1, weight1, sizeof(half)*weight1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight2, weight2, sizeof(half)*weight2_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight3, weight3, sizeof(half)*weight3_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight4, weight4, sizeof(half)*weight4_size, cudaMemcpyHostToDevice);

	FUNC

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int error_cnt = 0;
	for(int i=0; i<m*n; ++i){
		if( __half2float(output[i]) != 256){
			printf("(%d, %d): %f\n", i/256, i%256, __half2float(output[i]));
			error_cnt++;
			if(error_cnt>1000){
				break;
			}
		}
  }printf("\n");
	printf("error_cnd: %d\n", error_cnt);
	

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

	cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  const int round_cout = 0, loop = 1;
  float ms = 0, sum = 0;
  // 1. For original pointwise conv
  for(int round =0; round<round_cout; ++round){
    ms = 0, sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      FUNC
      checkCuda( cudaEventRecord(stopEvent,0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      sum += ms;
    }printf("Before fuse avg time %f\n", sum / loop);
  }

  delete[] input;
	delete[] weight1;
	delete[] weight2;
	delete[] weight3;
	delete[] weight4;
	delete[] output;
	cudaFree(d_input);
	cudaFree(d_weight1);
	cudaFree(d_weight2);
	cudaFree(d_weight3);
	cudaFree(d_weight4);
	cudaFree(d_output);
	return 0;
}

