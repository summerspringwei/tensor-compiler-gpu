

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


void init_data(float* input, float* pw_weight, float* output){
  // Init data
  for (int oc = 0; oc < out_channel; ++oc) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        input[oc * height * width + h * width + w] = 1;
        output[oc * height * width + h * width + w] = 0;
      }
    }
  }
  for (int oc = 0; oc < out_channel; ++oc) {
    for (int ic = 0; ic < out_channel; ++ic) {
      pw_weight[oc * in_channel + ic] = 1;
    }
  }
}

int main() {
  float *input = new float[in_channel * height * width];
  float *pw_weight = new float[out_channel * in_channel];
  float *output = new float[in_channel * height * width];
  float* d_input = NULL, *d_pw_weight = NULL, *d_output = NULL;
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **)&d_input, sizeof(float)*in_channel * height * width);
  err = cudaMalloc((void **)&d_pw_weight, sizeof(float)*out_channel*in_channel);
  err = cudaMalloc((void **)&d_output, sizeof(float)*in_channel * height * width);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  init_data(input, pw_weight, output);
  cudaMemcpy(d_input, input, sizeof(float)*in_channel * height * width, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pw_weight, pw_weight, sizeof(float)*out_channel*in_channel, cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock(56, 4, 1);
  dim3 numBlocks(14, 24, 1);
  cu_fused_avgpool_pointwise_conv<<<numBlocks, threadsPerBlock>>>(d_input, d_pw_weight, d_output);
  err = cudaMemcpy(output, d_output, sizeof(float)*in_channel * height * width, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int oc = 0; oc < out_channel; ++oc) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        printf("%f ", output[oc * height * width + h * width + w]);
      }
      printf("\n");
    }
  }

  delete input;
  delete pw_weight;
  delete output;
  return 0;
}