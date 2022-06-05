
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

void init_values(half* input, half* weight1, half* weight2, half* weight3, half* weight4, half* output){
  hf_init_values<half>(input, {16777216}, 1, 0);
  hf_init_values<half>(weight1, {65536}, 1, 0);
	hf_init_values<half>(weight2, {65536}, 1, 0);
	hf_init_values<half>(weight3, {65536}, 1, 0);
	hf_init_values<half>(weight4, {65536}, 1, 0);
  hf_init_values<half>(output, {16777216}, 1, 0);
}


int main(){
	const int input_size=16777216;	half *input = new half[input_size];
	const int weight1_size=65536;	half *weight1 = new half[weight1_size];
	const int weight2_size=65536;	half *weight2 = new half[weight2_size];
	const int weight3_size=65536;	half *weight3 = new half[weight3_size];
	const int weight4_size=65536;	half *weight4 = new half[weight4_size];
	const int output_size=16777216;	half *output = new half[output_size];

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

	fused_fc_fc<<<dim3(128*4, 1, 1), dim3(32, 4, 1)>>>(d_input, d_weight1, d_weight2, d_weight3, d_weight4, d_output);

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int error_cnt = 0;
	for(int i=0; i<128*4*16*256; ++i){
		if( __half2float(output[i]) != 256){
			// printf("(%d, %d): %f\n", i/256, i%256, __half2float(output[i]));
			error_cnt++;
			// if(error_cnt>1000){
			// 	break;
			// }
		}
  }printf("\n");

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
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