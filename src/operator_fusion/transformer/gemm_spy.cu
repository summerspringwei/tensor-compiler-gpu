
// We use a small gemm example to inspect the tensor core gemm instruction 
// and asychronization memory copy to enable the pipeline
// M, N, K = 64, 16, 16

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../utils.h"
#include "../../cuda_utils.h"
#include "half.hpp"
#include "gemm_spy.h"

void init_values(half* A, half* B, half* C){
  hf_init_values<half>(A, {1024}, 1, 0);
  hf_init_values<half>(B, {1024}, 1, 0);
  hf_init_values<half>(C, {256}, 0, 0);
}

void gemm_cpu(half* A, half* B, half* C, const int M, const int N, const int K){
  for(int i=0; i<M; i++){
    for(int j=0; j<N; ++j){
      C[i*N+j]=0;
      half_float::half hc(C[i*N+j]);
      for(int k=0; k<K; ++k){
        half_float::half ha(A[i*K+k]);
        half_float::half hb(B[j*K+k]);
        hc+=(ha*hb);
      }
      // C[i*N+j] = half_float::half_cast<half, half_float::half>(hc);
      C[i*N+j] = hc;
    }
  }
}

bool check(half* A, half *B){
  bool equal=true;
  for(int i=0; i<M; i++){
    for(int j=0; j<N; ++j){
      if(half_float::half(A[i*N+j])!=half_float::half(B[i*N+j])){
        equal=false;
        printf("%f %f\n", __half2float(A[i*N+j]), __half2float(B[i*N+j]));
        return equal;      }
    }
  }
  return equal;
}

// #define FUNC gemm_spy_baseline<half><<<dim3(1, 1, 1), dim3(32, 1, 1)>>>(d_input, d_weight, d_output);
// #define FUNC gemm_spy_pipline<half><<<dim3(1, 1, 1), dim3(32, 1, 1)>>>(d_input, d_weight, d_output);
#define FUNC gemm_spy_pipline_v2<half><<<dim3(1, 1, 1), dim3(32, 1, 1)>>>(d_input, d_weight, d_output);

int main(){
	const int input_size=1024;	half *input = new half[input_size];
	const int weight_size=1024;	half *weight = new half[weight_size];
	const int output_size=256;	half *output = new half[output_size];
  half *output_cpu = new half[output_size];

  init_values(input, weight, output);

  cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight=NULL;
	half *d_output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
	err=cudaMalloc((void **)&d_weight, sizeof(half)*weight_size);
	err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight, sizeof(half)*weight_size, cudaMemcpyHostToDevice);
  FUNC
	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  gemm_cpu(input, weight, output_cpu, M, N, K);

  if(check(output, output_cpu)){
    printf("Check pass\n");
  }else{
    printf("Check failed\n");
  }

  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  const int round_cout = 10, loop = 10000;
  float ms = 0, sum = 0;
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
	delete[] weight;
	delete[] output;
	cudaFree(d_input);
	cudaFree(d_weight);
	cudaFree(d_output);
	return 0;
}