
#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

#include "../utils.h"
#include "../cuda_utils.h"

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define WMMA_M1 32
#define WMMA_N1 8
#define WMMA_K1 16

#define WMMA_M2 32
#define WMMA_N1 
#define WMMA_K1 8

using namespace nvcuda;

__device__ int updiv(int a, int b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

int host_updiv(int a, int b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}


template<int height, int width, int in_channel, int out_channel, int num_warp>
__global__ void fused_pointwise_pointwise_conv(half* input, half* weight_1, half* weight_2, float* output){
  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  // int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M1, WMMA_N1, WMMA_K1, half, wmma::row_major> a1_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M1, WMMA_N1, WMMA_K1, half, wmma::col_major> b1_frag;
  wmma::fragment<wmma::accumulator, WMMA_M1, WMMA_N1, WMMA_K1, float> acc1_frag;
  
  // TODO(Chunwei Xia) for now we assume weight is small to fit into shared memory
  // Load weight to shared memory and pad with 0
  // Note that weight is in column major
  const int pad_out_channel = 8;
  __shared__ half shared_weight_1[in_channel * pad_out_channel];
  __shared__ float shared_output_1[num_warp * WMMA_M1 * pad_out_channel];
  #pragma unroll
  for(int i=0; i<updiv(out_channel * in_channel, blockDim.x); ++i){
    int idx = i*blockDim.x + threadIdx.x;
    if(idx >= in_channel * out_channel){continue;}
    shared_weight_1[idx] = weight_1[idx];
  }

  // Pad weight
  #pragma unroll
  for(int i=0; i<updiv((pad_out_channel-out_channel) * in_channel, blockDim.x); ++i){
    int idx = i*blockDim.x + threadIdx.x;
    if(idx >= (pad_out_channel-out_channel) * in_channel){continue;}
    shared_weight_1[idx + out_channel * in_channel] = 0;
  }
  __syncthreads();
  assert(height*width % WMMA_M1 == 0);
  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  #pragma unroll
  for(int i=0; i<in_channel; i+=WMMA_K){
    int a_col = i, a_row = warpM * WMMA_M1;
    int b_col = i, b_row = 0; // As all the weights are cached in the shared memory

    if(a_row < height*width && a_col < in_channel && b_row < pad_out_channel && b_col < in_channel){
      wmma::load_matrix_sync(a_frag, input+a_row*in_channel+a_col, in_channel);
      wmma::load_matrix_sync(b_frag, shared_weight_1+b_row*in_channel+b_col, in_channel);
      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Store back to shared memory
  wmma::store_matrix_sync(shared_output + (threadIdx.x / warpSize) * WMMA_M1 * pad_out_channel, 
    acc_frag, pad_out_channel, wmma::mem_row_major);
  __syncthreads();

  // Do sigmoid and mul here

  // Now start the next pointwise conv
  // 1. Load transposed weight to shared memory,
  __shared__ half shared_weight_2[in_channel*pad_out_channel];
  #pragma unroll
  for(int i=0; i<updiv(in_channel*out_channel, blockDim.x); ++i){
    int idx = i*blockDim.x + threadIdx.x;
    if(idx>=in_channel*out_channel){continue;}
    if(idx % pad_out_channel < out_channel){
      shared_weight_2[idx] = weight_2[idx/pad_out_channel*out_channel + (idx%pad_out_channel)];
    }else{
      shared_weight_2[idx] = 0;
    }
  }
  // 2. Declare the fragments

}

// Convert elements of array from float to half on device
__global__ void float2half(float* a, half* b, const int len){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx > len){return;}
  b[idx] = __float2half_rd(a[idx]);
}

int main() {
  // Declare size
  const int height = 112;
  const int width = 112;
  const int in_channel=144;
  const int out_channel=6;
  const int kernel_height = 3, kernel_width = 3;
  const int input_size = in_channel * height * width;
  const int dw_weight_size = in_channel * kernel_height * kernel_width;
  const int pw_weight_size = in_channel * out_channel;
  const int output_size = out_channel * height * width;

  // Declare arrays
  float *input = new float[input_size];
  float *dw_weight = new float[dw_weight_size];
  float *pw_weight = new float[pw_weight_size];
  float *output = new float[output_size];
  float *cpu_output = new float[output_size];
  float* d_input = NULL, *d_dw_weight = NULL, *d_pw_weight = NULL, *d_output = NULL;
  half* dh_input = NULL, *dh_dw_weight = NULL, *dh_pw_weight = NULL;

  // Allocate space on device
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&d_input, sizeof(float)*input_size);
  err = cudaMalloc((void **)&d_dw_weight, sizeof(float)*dw_weight_size);
  err = cudaMalloc((void **)&d_pw_weight, sizeof(float)*pw_weight_size);
  err = cudaMalloc((void **)&dh_input, sizeof(half)*input_size);
  err = cudaMalloc((void **)&dh_dw_weight, sizeof(half)*dw_weight_size);
  err = cudaMalloc((void **)&dh_pw_weight, sizeof(half)*pw_weight_size);
  err = cudaMalloc((void **)&d_output, sizeof(float)*output_size);

  // Check error
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  // Copy data and convert data type
  init_conv_conv_fusion_data(input, dw_weight, pw_weight, output, 
    height, width, 3, 3, in_channel, 1, 1, 1, in_channel, out_channel);
  cudaMemcpy(d_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dw_weight, dw_weight, sizeof(float)*dw_weight_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pw_weight, pw_weight, sizeof(float)*pw_weight_size, cudaMemcpyHostToDevice);
  
  float2half<<<host_updiv(input_size, 128), 128>>>(d_input, dh_input, input_size);
  float2half<<<host_updiv(pw_weight_size, 128), 128>>>(d_pw_weight, dh_pw_weight, pw_weight_size);
  cudaDeviceSynchronize();

  // Warm up
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  dim3 threadsPerBlock(128, 1, 1);
  dim3 numBlocks(height*width / threadsPerBlock.x, 1, 1);  
  simple_depthwise_conv<height, width, in_channel, out_channel, 128/32>
    <<<numBlocks, threadsPerBlock>>>(dh_input, dh_pw_weight, d_output);
  err = cudaMemcpy(output, d_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess){
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Benchmark
  int loop = 1000;
  float ms = 0, sum = 0;
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    simple_depthwise_conv<height, width, in_channel, out_channel, 128/32>
      <<<numBlocks, threadsPerBlock>>>(dh_input, dh_pw_weight, d_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
  }printf("avg time %f\n", sum / loop);
  sum = 0;
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    simple_depthwise_conv<height, width, in_channel, out_channel, 128/32>
      <<<numBlocks, threadsPerBlock>>>(dh_input, dh_pw_weight, d_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
  }printf("avg time %f\n", sum / loop);
  // Compute on CPU
  pointwise_conv(input, pw_weight, cpu_output, height, width, in_channel, out_channel);
  // Print result
  printf("outputs:->\n");
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int oc = 0; oc < out_channel; ++oc) {
        // printf("%.2f ", output[h*width*out_channel + w*out_channel+oc]);
        int idx = h*width*out_channel + w*out_channel+oc;
        if(std::abs(output[idx] - cpu_output[idx]) > 0.1 ){
          printf("<%d, %d, %d> %.2f, %.2f\n",h, w, oc, output[idx], cpu_output[idx]);
        }
      };
    }
  }


  // Free
  cudaFree(d_input);
  cudaFree(d_dw_weight);
  cudaFree(d_pw_weight);
  cudaFree(dh_input);
  cudaFree(dh_dw_weight);
  cudaFree(dh_pw_weight);
  cudaFree(d_output);
  delete[] input;
  delete[] dw_weight;
  delete[] pw_weight;
  delete[] output;
  delete[] cpu_output;
  return 0;
}
