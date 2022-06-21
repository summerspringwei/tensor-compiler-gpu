#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include "../../utils.h"
#include "../../cuda_utils.h"


int main(){
	const int qeury_size=98304;	half *query = new half[qeury_size];
	const int key_size=98304;	half *key = new half[key_size];
	const int output_size=196608;	half *output = new half[output_size];
	const int sum_size=1536;	half *sum = new half[sum_size];

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

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(sum, d_sum, sizeof(half)*sum_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  	delete[] query;
	delete[] key;
	delete[] output;
	delete[] sum;
	cudaFree(d_qeury);
	cudaFree(d_key);
	cudaFree(d_output);
	cudaFree(d_sum);
	return 0;
}