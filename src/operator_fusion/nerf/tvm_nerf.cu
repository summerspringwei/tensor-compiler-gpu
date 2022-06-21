

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include "../../utils.h"
#include "../../cuda_utils.h"


#include "tvm_fused_fc_fc.h"
#include "cublas_helper.h"

void init_values(half* input, half* weight1, half* weight2, half* weight3, half* weight4, half* output){
  hf_init_values<half>(input, {16777216}, 0.1, 0);
  hf_init_values<half>(weight1, {65536}, 0.1, 0);
	hf_init_values<half>(weight2, {65536}, 0.1, 0);
	hf_init_values<half>(weight3, {65536}, 1, 0);
	hf_init_values<half>(weight4, {65536}, 1, 0);
  hf_init_values<half>(output, {16777216}, 1, 0);
}

#define FUNC1 cudaLaunchCooperativeKernel((void*)tvm_fc, dim3(54, 4, 1),dim3(32, 4, 1), kernel_args1, 27648);
#define FUNC2 cudaLaunchCooperativeKernel((void*)tvm_fc, dim3(54, 4, 1),dim3(32, 4, 1), kernel_args2, 27648);
#define FUSED_FUNC cudaLaunchCooperativeKernel((void*)tvm_fused_fc, dim3(54, 4, 1),dim3(32, 4, 1), fused_kernel_args, 27648);
float expect = 2.56;
// float expect = 65.750000;

int main(){
  int error_cnt_threshold = 100000;
	
	
  int m = 108 * 4 * 16, n = 256, k = 256;
	const int input_size=16777216*2;	half *input = new half[input_size];
	const int weight1_size=65536;	half *weight1 = new half[weight1_size];
	const int weight2_size=65536;	half *weight2 = new half[weight2_size];
	const int weight3_size=65536;	half *weight3 = new half[weight3_size];
	const int weight4_size=65536;	half *weight4 = new half[weight4_size];
	const int output_size=input_size;	half *output = new half[output_size];
	half *output_cmp = new half[output_size]; half *output_tmp = new half[output_size]; 

	init_values(input, weight1, weight2, weight3, weight4, output);
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
	half *d_weight3=NULL;
	half *d_weight4=NULL;
	half *d_output=NULL;
	half *d_output_cmp=NULL;
	half *d_output_tmp=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
	err=cudaMalloc((void **)&d_weight1, sizeof(half)*weight1_size);
	err=cudaMalloc((void **)&d_weight2, sizeof(half)*weight2_size);
	err=cudaMalloc((void **)&d_weight3, sizeof(half)*weight3_size);
	err=cudaMalloc((void **)&d_weight4, sizeof(half)*weight4_size);
	err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);
	err=cudaMalloc((void **)&d_output_cmp, sizeof(half)*output_size);
	err=cudaMalloc((void **)&d_output_tmp, sizeof(half)*output_size);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight1, weight1, sizeof(half)*weight1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight2, weight2, sizeof(half)*weight2_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight3, weight3, sizeof(half)*weight3_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight4, weight4, sizeof(half)*weight4_size, cudaMemcpyHostToDevice);

  cudaDeviceProp deviceProp; \
  cudaGetDeviceProperties(&deviceProp, dev); \
  int numBlocksPerSm;
  int numThreads = 128;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void*)tvm_fused_fc, numThreads, 0); \
  printf("OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);\
  // void *kernel_args[] = { (void *)&(d_input), (void *)&(d_weight1), (void *)&(d_weight2), 
  //   (void *)&(d_weight3), (void *)&(d_weight4),  (void *)&(d_output)
  // };
  void *kernel_args1[] = { (void *)&(d_input), (void *)&(d_weight1), (void *)&(d_output_tmp) };
	void *kernel_args2[] = { (void *)&(d_output_tmp), (void *)&(d_weight2), (void *)&(d_output_cmp) };
	void *fused_kernel_args[] = { (void *)&(d_input), (void *)&(d_weight1), (void *)&(d_weight2), (void *)&(d_output) };
	// FUNC1
	// FUNC2
	// FUSED_FUNC
	cudnn_matmul_wrapper<half, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N>(d_input, d_weight1, d_output_tmp, m, n, k);
	cudnn_matmul_wrapper<half, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N>(d_output_tmp, d_weight2, d_output_cmp, m, n, k);
	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(output_cmp, d_output_cmp, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
  
  // Compare results
	int error_cnt = 0;
	for(int i=0; i<m*n; ++i){
		if( abs(__half2float(output[i]) - __half2float(output_cmp[i])) > ((__half2float(output[i]) + __half2float(output_cmp[i])) / 1000)){
			// printf("(%d, %d): %f\n", i/256, i%256, __half2float(output[i]));
			printf("(%d, %d): ours: %f, cudnn: %f\n", i/256, i%256, __half2float(output[i]), __half2float(output_cmp[i]));
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
  const int round_cout = 0, loop = 1;
  float ms = 0, sum = 0;
  // 1. For original pointwise conv
  for(int round =0; round<round_cout; ++round){
    ms = 0, sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      // FUNC
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
	delete[] output_cmp;
	cudaFree(d_input);
	cudaFree(d_weight1);
	cudaFree(d_weight2);
	cudaFree(d_weight3);
	cudaFree(d_weight4);
	cudaFree(d_output);
	cudaFree(d_output_tmp);
	cudaFree(d_output_cmp);
	return 0;
}



