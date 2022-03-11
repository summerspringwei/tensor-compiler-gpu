
#include <stdio.h>
#include "../utils.h"

const int height = 56, width = 56, in_channel = 24, out_channel = 24,
          kernel_height = 3, kernel_width = 3;
// input: NCHW, weight: OCICHW
void fused_avgpool_pointwise_conv(float *input, float *pw_weight,
                                  float *output) {
  // Omit N as batch size is 1
  for (int oc = 0; oc < out_channel; ++oc) { // parallel (blockIdx.y = 24)
    for (int h = 0; h < height; ++h) {       // parallel (blockIdx.x = 14, threadIdx.y = 4)
      for (int w = 0; w < width; ++w) {      // parallel (threadIdx.x = 56)
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
    }
  }
}

// input: NCHW, weight: OCICHW
void fused_depthwise_pointwise_conv(float *input, float *dw_weight,
                                    float *pw_weight, float *output) {
  // Omit N as batch size is 1
  for (int oc = 0; oc < out_channel; ++oc) { // parallel
    for (int h = 0; h < height; ++h) {       // parallel
      for (int w = 0; w < width; ++w) {      // parallel
        float reduce_sum = 0;
        for (int ic = 0; ic < in_channel; ++ic) { // reduce
          float window[3][3];
          float dw_sum = 0;
          // Do depthwise 3x3 conv to produce on element
          for (int wi = 0; wi < kernel_height; ++wi) {
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
    }
  }
}


int main() {
  float *input = new float[in_channel * height * width];
  float *dw_weight = new float[in_channel * kernel_height * kernel_width];
  float *pw_weight = new float[out_channel * in_channel];
  float *output = new float[in_channel * height * width];

  init_conv_conv_fusion_data(input, dw_weight, pw_weight, output);
  // fused_avgpool_pointwise_conv(input, pw_weight, output);
  fused_depthwise_pointwise_conv(input,dw_weight, pw_weight, output);

  for (int oc = 0; oc < out_channel; ++oc) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        printf("%f ", output[oc * height * width + h * width + w]);
      }
      printf("\n");
    }
  }

  delete[] input;
  delete[] pw_weight;
  delete[] output;
  return 0;
}