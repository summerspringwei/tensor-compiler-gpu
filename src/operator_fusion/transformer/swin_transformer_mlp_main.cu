
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../utils.h"
#include "../../cuda_utils.h"

#include "swin_transformer_mlp.h"

void init_values(half* input, half* weight){
  hf_init_values<half>(input, {524288}, 1, 0);
  hf_init_values<half>(weight, {1048576}, 1, 0);
}

int main(){
	const int input_size=524288;	half *input = new half[input_size];
	const int weight_size=1048576;	half *weight = new half[weight_size];
	const int output_size=131072;	half *output = new half[output_size];

  init_values(input, weight);

	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight=NULL;
	half *d_output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
	err=cudaMalloc((void **)&d_weight, sizeof(half)*weight_size);
	err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight, sizeof(half)*weight_size, cudaMemcpyHostToDevice);
	// fc2_16_16_2048_512<<<dim3(8,8,1), dim3(32, 1, 4)>>>(d_input, d_weight, d_output, (half*)nullptr);
  fc2_16_16_2048_512_v2<<<dim3(16, 16, 1), dim3(32, 4, 1)>>>(d_input, d_weight, d_output, (half*)nullptr);

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  const int round_cout = 1, loop = 1;
  float ms = 0, sum = 0;
  // 1. For original pointwise conv
  for(int round =0; round<round_cout; ++round){
    ms = 0, sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      fc2_16_16_2048_512_v2<<<dim3(16, 16, 1), dim3(32, 4, 1)>>>(d_input, d_weight, d_output, (half*)nullptr);
      checkCuda( cudaEventRecord(stopEvent,0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      sum += ms;
    }printf("After fuse avg time %f\n", sum / loop);
  }

  for(int i=0; i<output_size; ++i){
    // printf("%f ", __half2float(output[i]));
		if( __half2float(output[i]) != 2048){
			printf("(%d, %d): %f\n", i/512, i%512, __half2float(output[i]));
			// break;
		}
  }printf("\n");
  
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  delete[] input;
	delete[] weight;
	delete[] output;
	cudaFree(d_input);
	cudaFree(d_weight);
	cudaFree(d_output);
	return 0;
}