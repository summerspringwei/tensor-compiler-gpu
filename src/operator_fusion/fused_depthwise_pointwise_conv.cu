

#include <stdio.h>
#include <cuda_runtime.h>


#include "../utils.h"

const int height = 56, width = 56, in_channel=24, out_channel=24, 
  kernel_height = 3, kernel_width = 3;


__global__ void cu_fused_avgpool_pointwise_conv(float *input, float *pw_weight,
                                  float *output) {
  const int oc = blockIdx.y;
  const int h = blockIdx.x * blockDim.y + threadIdx.y;
  const int w = threadIdx.x;
  if(oc>=out_channel || h>=height || w>=width){
    return;
  }
  float reduce_sum = 0;
  for (int ic = 0; ic < in_channel; ++ic) { // reduce
    float sum = 0;
    // Do 3x3 avg_pool to produce on element
    for (int wi = 0; wi < 3; ++wi) {
      for (int wj = 0; wj < 3; ++wj) {
        if (!(h - 1 + wi < 0 || w - 1 + wj < 0 || h - 1 + wi >= height ||
            w - 1 + wj >= width)) {
          sum += input[ic * height * width +
                                  (h - 1 + wi) * width + (w - 1 + wj)];
        }
      }
    }
    float avg = sum / 9;
    reduce_sum += (avg * pw_weight[oc * in_channel + ic]);
  }
  output[oc * height * width + h * width + w] = reduce_sum;
}


__global__ void cu_fused_depthwise_pointwise_conv_v1(float *input, float *dw_weight,
                                  float *pw_weight, float *output) {
  const int oc = blockIdx.y;
  const int h = blockIdx.x * blockDim.y + threadIdx.y;
  const int w = threadIdx.x;
  if(oc>=out_channel || h>=height || w>=width){
    return;
  }
  // __shared__ float s_input[width * 4];
  // __shared__ float s_dw_weight[in_channel * kernel_height * kernel_width];
  // __shared__ float s_pw_weight[out_channel*in_channel];
  // Load input and weights to shared memory
  // s_input[threadIdx.y * blockDim.x + threadIdx.x] = input[oc*height*width + h*width + w];
  // int stride = in_channel * kernel_height * kernel_width / (blockDim.x * blockDim.y) + 1;
  // for(int i=0; i<stride; ++i){
  //   int index = i * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
  //   if(index < in_channel * kernel_height * kernel_width){
  //     s_dw_weight[index] = dw_weight[index];
  //   }
  // }
  // stride = out_channel * in_channel / (blockDim.x * blockDim.y) + 1;
  // for(int i=0; i<stride; ++i){
  //   int index = i * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
  //   if(index < out_channel * in_channel){
  //     s_dw_weight[index] = dw_weight[index];
  //   }
  // }
  // __syncthreads();
  float reduce_sum = 0;
  for (int ic = 0; ic < in_channel; ++ic) { // reduce
    float dw_sum = 0;
    // Do depthwise 3x3 conv to produce on element
    #pragma unroll
    for (int wi = 0; wi < kernel_height; ++wi) {
      #pragma unroll
      for (int wj = 0; wj < kernel_width; ++wj) {
        if (!(h - 1 + wi < 0 || w - 1 + wj < 0 || h - 1 + wi >= height ||
            w - 1 + wj >= width)) {
          // window[wi][wj] = s_input[(threadIdx.y-1+wj) * blockDim.x + (threadIdx.x-1+wi)];
          dw_sum += (input[(h - 1 + wi)*width + w - 1 + wj] *
                      dw_weight[ic * kernel_height * kernel_width +
                                wi * kernel_width + wj]);
        }
      }
    }
    reduce_sum += (dw_sum * pw_weight[oc * in_channel + ic]);
  }
  output[oc * height * width + h * width + w] = reduce_sum;
}

/**
 * @brief fusion of depthwise and pointwise conv
 * For each block, we compute tile_x*tile_y*out_channels output,
 * We cache the depthwise conv output in shared memory
 * thus we do not need to recompute for each output_channel's computation
 * 
 * @param input 
 * @param dw_weight 
 * @param pw_weight 
 * @param output 
 * @return __global__ 
 */
__global__ void cu_fused_depthwise_pointwise_conv_v2(float *input, float *dw_weight,
                                  float *pw_weight, float *output) {
  const int oc = threadIdx.z;
  const int tidx_h = threadIdx.y;
  const int tidx_w = threadIdx.x;
  const int tile_h = blockDim.y;
  const int tile_w = blockDim.x;
  const int tile_oc = blockDim.z;
  int h = blockIdx.x * tile_w + tidx_w;
  int w = blockIdx.y * tile_h + tidx_h;
  if(oc>=out_channel || h>=height || w>=width){
    return;
  }
  // __shared__ float cache_dw_output[tile_h*tile_w*tile_oc];
  // __shared__ float s_input[width * 4];
  // __shared__ float s_dw_weight[in_channel * kernel_height * kernel_width];
  // __shared__ float s_pw_weight[out_channel*in_channel];
  // Load input and weights to shared memory
  // s_input[threadIdx.y * blockDim.x + threadIdx.x] = input[oc*height*width + h*width + w];
  // int stride = in_channel * kernel_height * kernel_width / (blockDim.x * blockDim.y) + 1;
  // for(int i=0; i<stride; ++i){
  //   int index = i * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
  //   if(index < in_channel * kernel_height * kernel_width){
  //     s_dw_weight[index] = dw_weight[index];
  //   }
  // }
  // stride = out_channel * in_channel / (blockDim.x * blockDim.y) + 1;
  // for(int i=0; i<stride; ++i){
  //   int index = i * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
  //   if(index < out_channel * in_channel){
  //     s_dw_weight[index] = dw_weight[index];
  //   }
  // }
  // First compute the depthwise
  

  __syncthreads();
  float reduce_sum = 0;
  for (int ic = 0; ic < in_channel; ++ic) { // reduce
    float dw_sum = 0;
    // Do depthwise 3x3 conv to produce on element
    #pragma unroll
    for (int wi = 0; wi < kernel_height; ++wi) {
      #pragma unroll
      for (int wj = 0; wj < kernel_width; ++wj) {
        if (!(h - 1 + wi < 0 || w - 1 + wj < 0 || h - 1 + wi >= height ||
            w - 1 + wj >= width)) {
          // window[wi][wj] = s_input[(threadIdx.y-1+wj) * blockDim.x + (threadIdx.x-1+wi)];
          dw_sum += (input[(h - 1 + wi)*width + w - 1 + wj] *
                      dw_weight[ic * kernel_height * kernel_width +
                                wi * kernel_width + wj]);
        }
      }
    }
    reduce_sum += (dw_sum * pw_weight[oc * in_channel + ic]);
  }
  output[oc * height * width + h * width + w] = reduce_sum;
}



// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result){
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

int main() {
  float *input = new float[in_channel * height * width];
  float *dw_weight = new float[in_channel * kernel_height * kernel_width];
  float *pw_weight = new float[out_channel * in_channel];
  float *output = new float[in_channel * height * width];
  float* d_input = NULL, *d_dw_weight = NULL, *d_pw_weight = NULL, *d_output = NULL;
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&d_input, sizeof(float)*in_channel * height * width);
  err = cudaMalloc((void **)&d_dw_weight, sizeof(float)*in_channel * kernel_height * kernel_width);
  err = cudaMalloc((void **)&d_pw_weight, sizeof(float)*out_channel*in_channel);
  err = cudaMalloc((void **)&d_output, sizeof(float)*in_channel * height * width);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  init_conv_conv_fusion_data(input, dw_weight, pw_weight, output, 
    height, width, 3, 3, in_channel, 1, 1, 1, in_channel, out_channel);
  cudaMemcpy(d_input, input, sizeof(float)*in_channel * height * width, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dw_weight, dw_weight, sizeof(float)*in_channel * kernel_height * kernel_width, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pw_weight, pw_weight, sizeof(float)*out_channel*in_channel, cudaMemcpyHostToDevice);
  
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  dim3 threadsPerBlock(56, 4, 1);
  dim3 numBlocks(14, 24, 1);
  // cu_fused_avgpool_pointwise_conv<<<numBlocks, threadsPerBlock>>>(d_input, d_pw_weight, d_output);
  cu_fused_depthwise_pointwise_conv_v1<<<numBlocks, threadsPerBlock>>>(d_input, d_dw_weight, d_pw_weight, d_output);
  err = cudaMemcpy(output, d_output, sizeof(float)*in_channel * height * width, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess){
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  int loop = 1000;
  float ms = 0, sum = 0;
  
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    // cu_fused_avgpool_pointwise_conv<<<numBlocks, threadsPerBlock>>>(d_input, d_pw_weight, d_output);
    cu_fused_depthwise_pointwise_conv_v1<<<numBlocks, threadsPerBlock>>>(d_input, d_dw_weight, d_pw_weight, d_output);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    sum += ms;
  }

  for (int oc = 0; oc < out_channel; ++oc) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        printf("%f ", output[oc * height * width + h * width + w]);
      }
      printf("\n");
    }
  }

  printf("avg time %f\n", sum / loop);
  cudaFree(d_input);
  cudaFree(d_dw_weight);
  cudaFree(d_pw_weight);
  cudaFree(d_output);
  delete[] input;
  delete[] dw_weight;
  delete[] pw_weight;
  delete[] output;
  return 0;
}