#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include "../../utils.h"
#include "../../cuda_utils.h"

#include "bert_query_key_matmul_softmax.h"

// #define FUSED_FUNC checkCuda(cudaLaunchCooperativeKernel((void*)fused_query_key_matmul_softmax, dim3(4, 4,12), dim3(32,1,1), kernel_args, 8704*sizeof(half)));
// #define FUSED_FUNC checkCuda(cudaLaunchCooperativeKernel((void*)fused_query_key_matmul_softmax_v2, dim3(4, 4,12), dim3(32,8,1), kernel_args, 8704*sizeof(half)));
#define FUSED_FUNC checkCuda(cudaLaunchCooperativeKernel((void*)fused_query_key_matmul_softmax_v3, dim3(4, 4,12), dim3(32,1,1), kernel_args, 8704*sizeof(half)));

int num_head = 12, m = 128, n = 128, k = 64;

void init_values(half* query, half* key, half* sum, half* output){
  hf_init_values<half>(query, {num_head * m * k}, 0.1, 0);
  hf_init_values<half>(key, {num_head * n * k}, 0.1, 0);
	hf_init_values<half>(sum, {num_head*m}, 0, 0);
  hf_init_values<half>(output, {num_head*m*n}, 1, 0);
}

int main(){
	const int qeury_size=98304;	half *query = new half[qeury_size];
	const int key_size=98304;	half *key = new half[key_size];
	const int output_size=196608;	half *output = new half[output_size];
	const int sum_size=1536;	half *reduce_sum = new half[sum_size];
	init_values(query, key, reduce_sum, output);
	cudaError_t err = cudaSuccess;
	half *d_qeury=NULL;
	half *d_key=NULL;
	half *d_output=NULL;
	half *d_sum=NULL;
	err=cudaMalloc((void **)&d_qeury, sizeof(half)*qeury_size);
	err=cudaMalloc((void **)&d_key, sizeof(half)*key_size);
	err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);
	err=cudaMalloc((void **)&d_sum, sizeof(half)*sum_size);

	cudaMemcpy(d_qeury, query, sizeof(half)*qeury_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_key, key, sizeof(half)*key_size, cudaMemcpyHostToDevice);
	void *kernel_args[] = { (void *)&(d_qeury), (void *)&(d_key), (void *)&(d_output), (void *)&(d_sum) };
	FUSED_FUNC

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(reduce_sum, d_sum, sizeof(half)*sum_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

	int error_cnt = 0;

	// for(int i=0; i<num_head*m*n; ++i){
	// 	if(__half2float(output[i])-0.007812 > 0.0001){
	// 		printf("(%d, %d, %d): ours: %f\n", i/(m*n), (i%(m*n))/n, i%n, __half2float(output[i]));
	// 	}
  // }printf("\n");


	cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  const int round_cout = 3, loop = 10000;
  float ms = 0, sum = 0;
  // 1. For original pointwise conv
  for(int round =0; round<round_cout; ++round){
    ms = 0, sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      FUSED_FUNC
      checkCuda( cudaEventRecord(stopEvent,0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      sum += ms;
    }printf("Before fuse avg time %f\n", sum / loop);
  }


  delete[] query;
	delete[] key;
	delete[] output;
	delete[] reduce_sum;
	cudaFree(d_qeury);
	cudaFree(d_key);
	cudaFree(d_output);
	cudaFree(d_sum);
	return 0;
}