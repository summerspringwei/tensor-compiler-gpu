

#include <stdio.h>
#include <cuda_runtime.h>

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
    float window[3][3];
    float sum = 0;
    // Do 3x3 avg_pool to produce on element
    for (int wi = 0; wi < 3; ++wi) {
      for (int wj = 0; wj < 3; ++wj) {
        if (h - 1 + wi < 0 || w - 1 + wj < 0 || h - 1 + wi >= height ||
            w - 1 + wj >= width) {
          window[wi][wj] = 0;
          sum += window[wi][wj];
        } else {
          window[wi][wj] = input[ic * height * width +
                                  (h - 1 + wi) * width + (w - 1 + wj)];
          sum += window[wi][wj];
        }
      }
    }
    float avg = sum / 9;
    reduce_sum += (avg * pw_weight[oc * in_channel + ic]);
  }
  output[oc * height * width + h * width + w] = reduce_sum;
}

__global__ void cu_fused_depthwise_pointwise_conv(float *input, float *dw_weight,
                                  float *pw_weight, float *output) {
  const int oc = blockIdx.y;
  const int h = blockIdx.x * blockDim.y + threadIdx.y;
  const int w = threadIdx.x;
  if(oc>=out_channel || h>=height || w>=width){
    return;
  }
  float reduce_sum = 0;
  for (int ic = 0; ic < in_channel; ++ic) { // reduce
    float window[3][3];
    float dw_sum = 0;
    // Do depthwise 3x3 conv to produce on element
    #pragma unroll
    for (int wi = 0; wi < kernel_height; ++wi) {
      #pragma unroll
      for (int wj = 0; wj < kernel_width; ++wj) {
        if (h - 1 + wi < 0 || w - 1 + wj < 0 || h - 1 + wi >= height ||
            w - 1 + wj >= width) {
          window[wi][wj] = 0;
          dw_sum += (window[wi][wj] *
                      dw_weight[ic * kernel_height * kernel_width +
                                wi * kernel_width + wj]);
            
        } else {
          window[wi][wj] = input[ic * height * width +
                                  (h - 1 + wi) * width + (w - 1 + wj)];
          dw_sum += (window[wi][wj] *
                      dw_weight[ic * kernel_height * kernel_width +
                                wi * kernel_width + wj]);
        }
      }
    }
    reduce_sum += (dw_sum * pw_weight[oc * in_channel + ic]);
  }
  output[oc * height * width + h * width + w] = reduce_sum;
}


void init_data(float* input, float* dw_weight, float* pw_weight, float* output){
  for (int ic = 0; ic < out_channel; ++ic) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        input[ic * height * width + h * width + w] = 1;
      }
    }
  }

  for (int ic = 0; ic < in_channel; ++ic) {
    for (int h = 0; h < kernel_height; ++h) {
      for (int w = 0; w < kernel_width; ++w) {
        dw_weight[ic * kernel_height * kernel_width + h * kernel_width + w] = 1;
      }
    }
  }

  for (int oc = 0; oc < out_channel; ++oc) {
    for (int ic = 0; ic < out_channel; ++ic) {
      pw_weight[oc * in_channel + ic] = 1;
    }
  }

  for (int oc = 0; oc < out_channel; ++oc) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        output[oc * height * width + h * width + w] = 0;
      }
    }
  }  
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
  
  init_data(input, dw_weight, pw_weight, output);
  cudaMemcpy(d_input, input, sizeof(float)*in_channel * height * width, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dw_weight, dw_weight, sizeof(float)*in_channel * kernel_height * kernel_width, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pw_weight, pw_weight, sizeof(float)*out_channel*in_channel, cudaMemcpyHostToDevice);
  
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  dim3 threadsPerBlock(56, 4, 1);
  dim3 numBlocks(14, 24, 1);
  // cu_fused_avgpool_pointwise_conv<<<numBlocks, threadsPerBlock>>>(d_input, d_pw_weight, d_output);
  cu_fused_depthwise_pointwise_conv<<<numBlocks, threadsPerBlock>>>(d_input, d_dw_weight, d_pw_weight, d_output);
  err = cudaMemcpy(output, d_output, sizeof(float)*in_channel * height * width, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess){
      fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  int loop = 10000;
  float ms = 0, sum = 0;
  
  for(int i=0; i<loop; ++i){
    checkCuda( cudaEventRecord(startEvent,0) );
    // cu_fused_avgpool_pointwise_conv<<<numBlocks, threadsPerBlock>>>(d_input, d_pw_weight, d_output);
    cu_fused_depthwise_pointwise_conv<<<numBlocks, threadsPerBlock>>>(d_input, d_dw_weight, d_pw_weight, d_output);
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