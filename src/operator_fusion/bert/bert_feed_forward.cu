
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include "../../utils.h"
#include "../../cuda_utils.h"

#include "bert_feed_forward_fc1.h"
#include "bert_feed_forward_fc2.h"
#include "bert_fused_fc_fc.h"

// Two fc configuration
int m1 = 128, n1 = 3072, k1 = 768;
int m2 = m1, n2 = 768, k2= n1;

void init_values(half* input, half* weight1, half* weight2, half* output){
  hf_init_values<half>(input, {m1*k1}, 0.1, 0);
  hf_init_values<half>(weight1, {n1*k1}, 0.1, 0);
	hf_init_values<half>(weight2, {n2*k2}, 0.1, 0);
  hf_init_values<half>(output, {m2*n2}, 1, 0);
}
 
#define FUNC1 checkCuda(cudaLaunchCooperativeKernel((void*)fc1_128_768_3072, dim3(2, 96,1), dim3(32,2,2), kernel_args1, 13056*sizeof(half)));
#define FUNC2 checkCuda(cudaLaunchCooperativeKernel((void*)fc2, dim3(4, 24,1), dim3(32,4,1), kernel_args2, 8704*sizeof(half)));
#define FUSED_FUNC checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v2, dim3(192, 1, 1), dim3(128, 1, 1), fused_kernel_args, 13056*sizeof(half)));

int main(){
  int error_cnt_threshold = 100000;
	
	const int input_size=m1*k1;	half *input = new half[input_size];
	const int weight1_size=n1*k1;	half *weight1 = new half[weight1_size];
	const int weight2_size=n2*k2;	half *weight2 = new half[weight2_size];
	const int output1_size=m1*n1;	half *output1 = new half[output1_size];
	const int output2_size=m2*n2;	half *output2 = new half[output2_size];
	half *output_tmp = new half[output1_size]; 
	half *output_cmp = new half[output2_size]; 

	init_values(input, weight1, weight2, output1);
  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  if(supportsCoopLaunch){
      printf("Device support CoopLaunch\n");
  }

  
	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight1=NULL;
	half *d_weight2=NULL;
	half *d_output1=NULL;
	half *d_output2=NULL;
	half *d_output_tmp=NULL;
	half *d_output_cmp=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
	err=cudaMalloc((void **)&d_weight1, sizeof(half)*weight1_size);
	err=cudaMalloc((void **)&d_weight2, sizeof(half)*weight2_size);
	err=cudaMalloc((void **)&d_output1, sizeof(half)*output1_size);
	err=cudaMalloc((void **)&d_output2, sizeof(half)*output2_size);
	err=cudaMalloc((void **)&d_output_tmp, sizeof(half)*output1_size);
	err=cudaMalloc((void **)&d_output_cmp, sizeof(half)*output2_size);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight1, weight1, sizeof(half)*weight1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight2, weight2, sizeof(half)*weight2_size, cudaMemcpyHostToDevice);

  cudaDeviceProp deviceProp; \
  cudaGetDeviceProperties(&deviceProp, dev); \
  int numBlocksPerSm;
  int numThreads = 128;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void*)fc1_128_768_3072, numThreads, 0); \
  printf("fc1: OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);\
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void*)fc2, numThreads, 0); \
  printf("fc2: OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);\
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void*)fused_fc_fc_v2, numThreads, 0); \
  printf("fused_fc_fc: OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);\
  
  void *kernel_args1[] = { (void *)&(d_input), (void *)&(d_weight1), (void *)&(d_output_tmp) };
	void *kernel_args2[] = { (void *)&(d_output_tmp), (void *)&(d_weight2), (void *)&(d_output_cmp) };
	void *fused_kernel_args[] = { (void *)&(d_input), (void *)&(d_weight1), (void *)&(d_output1), (void *)&(d_weight2), (void *)&(d_output2) };
	FUNC1
	FUNC2
	FUSED_FUNC
	cudaDeviceSynchronize();
	cudaMemcpy(output1, d_output1, sizeof(half)*output1_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(output2, d_output2, sizeof(half)*output2_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_tmp, d_output_tmp, sizeof(half)*output1_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_cmp, d_output_cmp, sizeof(half)*output2_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
  printf("output1: %f, output_tmp: %f\n", __half2float(output1[0]), __half2float(output_tmp[0]));
  // Compare results
	int error_cnt = 0;

	for(int i=0; i<m1*n1; ++i){
		if( abs(__half2float(output1[i]) - __half2float(output_tmp[i])) > ((__half2float(output1[i]) + __half2float(output_tmp[i])) / 1000)){
			// printf("(%d, %d): %f\n", i/256, i%256, __half2float(output[i]));
			printf("(%d, %d): ours: %f, tvm: %f\n", i/n1, i%n1, __half2float(output1[i]), __half2float(output_tmp[i]));
		}
  }printf("\n");

	for(int i=0; i<m2*n2; ++i){
		if( abs(__half2float(output2[i]) - __half2float(output_cmp[i])) > ((__half2float(output2[i]) + __half2float(output_cmp[i])) / 1000) || true){
			// printf("(%d, %d): %f\n", i/256, i%256, __half2float(output[i]));
			printf("(%d, %d): ours: %f, tvm: %f\n", i/n2, i%n2, __half2float(output2[i]), __half2float(output_cmp[i]));
			error_cnt++;
			if(error_cnt>error_cnt_threshold){
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
  const int round_cout = 3, loop = 10000;
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

  delete[] input;
	delete[] weight1;
	delete[] weight2;
	delete[] output1;
	delete[] output2;
	delete[] output_cmp;
	delete[] output_tmp;
	cudaFree(d_input);
	cudaFree(d_weight1);
	cudaFree(d_weight2);
	cudaFree(d_output1);
	cudaFree(d_output2);
	cudaFree(d_output_tmp);
	cudaFree(d_output_cmp);
	return 0;
}
