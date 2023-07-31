
#include <iostream>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>


#include "../../cuda_utils.h"
#include "../../utils.h"

#include "kernels/bert_fused_fc_fc.h"
#include "kernels/bert_query_key_matmul_softmax.h"
#include "kernels/bert_qkv.h"
#include "kernels/bert_qkv_matmul_transpose.h"
#include "kernels/bert_main_kernel.h"

int m1 = 128, n1 = 768*3, k1 = 768;

void init_values(half *input, half *weight1, half *output) {
  hf_init_values<half>(input, {m1 * k1}, half(1.0/16), 0);
  hf_init_values<half>(weight1, {n1 * k1}, half(1.0/16), 0);
  hf_init_values<half>(output, {m1 * n1}, 0, 0);
}


int main(){
	const int input_size=98304;	half *input = new half[input_size];
	const int weight_qkv_size=768*768*3;	half *weight_qkv = new half[weight_qkv_size];
	const int qkv_output_size=128*768*3;	half *qkv_output = new half[qkv_output_size];
	const int query_size=98304;	half *query = new half[query_size];
	const int key_size=98304;	half *key = new half[key_size];
	const int value_size=98304;	half *value = new half[value_size];
	const int output_size=98304;	half *output = new half[output_size];
	const int query_key_output_size=196608;	half *query_key_output = new half[query_key_output_size];
	const int sum_size=1536;	half *sum = new half[sum_size];
	const int qv_value_output_size=98304;	half *qv_value_output = new half[qv_value_output_size];
	const int attn_fc_weight_size=589824;	half *attn_fc_weight = new half[attn_fc_weight_size];
	const int attn_fc_output_size=98304;	half *attn_fc_output = new half[attn_fc_output_size];
	const int attn_output_layer_norm_sum_size=128;	half *attn_output_layer_norm_sum = new half[attn_output_layer_norm_sum_size];
	const int attn_output_layer_norm_variance_size=128;	half *attn_output_layer_norm_variance = new half[attn_output_layer_norm_variance_size];

	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight_qkv=NULL;
	half *d_qkv_output=NULL;
	half *d_query=NULL;
	half *d_key=NULL;
	half *d_value=NULL;
	half *d_output=NULL;
	half *d_query_key_output=NULL;
	half *d_sum=NULL;
	// half *d_qv_value_output=NULL;
	// half *d_attn_fc_weight=NULL;
	// half *d_attn_fc_output=NULL;
	// half *d_attn_output_layer_norm_sum=NULL;
	// half *d_attn_output_layer_norm_variance=NULL;
	checkCuda(cudaMalloc((void **)&d_input, sizeof(half)*input_size));
	checkCuda(cudaMalloc((void **)&d_weight_qkv, sizeof(half)*weight_qkv_size));
	checkCuda(cudaMalloc((void **)&d_qkv_output, sizeof(half)*qkv_output_size));
	checkCuda(cudaMalloc((void **)&d_query, sizeof(half)*query_size));
	checkCuda(cudaMalloc((void **)&d_key, sizeof(half)*key_size));
	checkCuda(cudaMalloc((void **)&d_value, sizeof(half)*value_size));
	checkCuda(cudaMalloc((void **)&d_output, sizeof(half)*output_size));
	checkCuda(cudaMalloc((void **)&d_query_key_output, sizeof(half)*query_key_output_size));
	checkCuda(cudaMalloc((void **)&d_sum, sizeof(half)*sum_size));
	// checkCuda(cudaMalloc((void **)&d_qv_value_output, sizeof(half)*qv_value_output_size));
	// checkCuda(cudaMalloc((void **)&d_attn_fc_weight, sizeof(half)*attn_fc_weight_size));
	// checkCuda(cudaMalloc((void **)&d_attn_fc_output, sizeof(half)*attn_fc_output_size));
	// checkCuda(cudaMalloc((void **)&d_attn_output_layer_norm_sum, sizeof(half)*attn_output_layer_norm_sum_size));
	// checkCuda(cudaMalloc((void **)&d_attn_output_layer_norm_variance, sizeof(half)*attn_output_layer_norm_variance_size));

	init_values(input, weight_qkv, qkv_output);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_input, input, 1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight_qkv, weight_qkv, sizeof(half)*weight_qkv_size, cudaMemcpyHostToDevice);
	// // cudaMemcpy(d_qkv_output, qkv_output, sizeof(half)*qkv_output_size, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_attn_fc_weight, attn_fc_weight, sizeof(half)*attn_fc_weight_size, cudaMemcpyHostToDevice);

	void *fused_kernel_args[] = { (void *)&(d_input), (void *)&(d_weight_qkv), 
    (void *)&(d_qkv_output), (void *)&(d_query), (void *)&(d_key), 
    (void *)&(d_value), (void*)&(d_query_key_output), (void*)&(d_sum)};
	checkCuda(cudaLaunchCooperativeKernel((void*)bert_attn_kernel, dim3(192, 1,1), dim3(32*2,1,1), fused_kernel_args, 13056 * sizeof(half)));

	cudaDeviceSynchronize();
	cudaMemcpy(query, d_query, sizeof(half)*query_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(key, d_key, sizeof(half)*key_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(value, d_value, sizeof(half)*value_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(query_key_output, d_query_key_output, sizeof(half)*query_key_output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(sum, d_sum, sizeof(half)*sum_size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(qv_value_output, d_qv_value_output, sizeof(half)*qv_value_output_size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(attn_fc_output, d_attn_fc_output, sizeof(half)*attn_fc_output_size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(attn_output_layer_norm_sum, d_attn_output_layer_norm_sum, sizeof(half)*attn_output_layer_norm_sum_size, cudaMemcpyDeviceToHost);
	// cudaMemcpy(attn_output_layer_norm_variance, d_attn_output_layer_norm_variance, sizeof(half)*attn_output_layer_norm_variance_size, cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();

  // if (err != cudaSuccess){
  //   fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
  //   exit(EXIT_FAILURE);
  // }

  delete[] input;
	delete[] weight_qkv;
	delete[] qkv_output;
	delete[] query;
	delete[] key;
	delete[] value;
	delete[] output;
	delete[] query_key_output;
	delete[] sum;
	// delete[] qv_value_output;
	// delete[] attn_fc_weight;
	// delete[] attn_fc_output;
	// delete[] attn_output_layer_norm_sum;
	// delete[] attn_output_layer_norm_variance;
	cudaFree(d_input);
	cudaFree(d_weight_qkv);
	cudaFree(d_qkv_output);
	cudaFree(d_query);
	cudaFree(d_key);
	cudaFree(d_value);
	cudaFree(d_output);
	cudaFree(d_query_key_output);
	cudaFree(d_sum);
	// cudaFree(d_qv_value_output);
	// cudaFree(d_attn_fc_weight);
	// cudaFree(d_attn_fc_output);
	// cudaFree(d_attn_output_layer_norm_sum);
	// cudaFree(d_attn_output_layer_norm_variance);
	return 0;
}