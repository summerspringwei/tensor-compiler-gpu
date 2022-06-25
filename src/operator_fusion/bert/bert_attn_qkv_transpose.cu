
#include <assert.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

#include "../../cuda_utils.h"
#include "../../utils.h"
#include "../transformer/half.hpp"


#include "kernels/bert_qkv.h"
#include "kernels/bert_qkv_matmul_transpose.h"

int m1 = 128, n1 = 768*3, k1 = 768;

void init_values(half *input, half *weight1, half *output) {
  hf_init_values<half>(input, {m1 * k1}, half(1.0/16), 0);
  hf_init_values<half>(weight1, {n1 * k1}, half(1.0/16), 0);
  hf_init_values<half>(output, {m1 * n1}, 0, 0);
}


int main(){
	const int input_size=98304;	half *input = new half[input_size];
	const int weight_qkv_size=1769472;	half *weight_qkv = new half[weight_qkv_size];
	const int query_size=98304;	half *query = new half[query_size];
	const int key_size=98304;	half *key = new half[key_size];
	const int value_size=98304;	half *value = new half[value_size];
	const int output_size=98304;	half *output = new half[output_size];

	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight_qkv=NULL;
	half *d_query=NULL;
	half *d_key=NULL;
	half *d_value=NULL;
	half *d_output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
	err=cudaMalloc((void **)&d_weight_qkv, sizeof(half)*weight_qkv_size);
	err=cudaMalloc((void **)&d_query, sizeof(half)*query_size);
	err=cudaMalloc((void **)&d_key, sizeof(half)*key_size);
	err=cudaMalloc((void **)&d_value, sizeof(half)*value_size);
	err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);
	
	init_values(input, weight_qkv, output);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight_qkv, weight_qkv, sizeof(half)*weight_qkv_size, cudaMemcpyHostToDevice);

	void *fused_kernel_args[] = { (void *)&(d_input), (void *)&(d_weight_qkv), 
    (void *)&(d_output), (void *)&(d_query), (void *)&(d_key), (void *)&(d_value)};
	checkCuda(cudaLaunchCooperativeKernel((void*)fused_attn_qkv_matmul_transpose_kernel, dim3(4, 36,1), dim3(32,2,1), fused_kernel_args, 13056 * sizeof(half)));

	cudaDeviceSynchronize();
	cudaMemcpy(query, d_query, sizeof(half)*query_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(key, d_key, sizeof(half)*key_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(value, d_value, sizeof(half)*value_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  	delete[] input;
	delete[] weight_qkv;
	delete[] query;
	delete[] key;
	delete[] value;
	delete[] output;
	cudaFree(d_input);
	cudaFree(d_weight_qkv);
	cudaFree(d_query);
	cudaFree(d_key);
	cudaFree(d_value);
	cudaFree(d_output);
	return 0;
}