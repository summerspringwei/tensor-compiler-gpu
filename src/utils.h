#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cstdlib>
#include <time.h> 
typedef unsigned long uint64_t;

uint64_t get_shape_size(std::vector<uint64_t> shape){
    uint64_t shape_size = 1;
    for(auto s: shape){
        shape_size *= s;
    }
    return shape_size;
}



void print_and_check(std::vector<float>& output, float expected){
  for(int i=0; i<16; ++i){
    printf("%f ", output[i]);
  }printf("\n");
  for(int i=0; i<output.size(); ++i){
    auto d = output[i];
    if(d!=expected){
      printf("error %d output is %f\n", i, d);
      break;
    }
  }
}

/**
 * @brief Init input and weights for two continuos conv
 * 
 * @param input 
 * @param dw_weight weight for first depthwise conv
 * @param pw_weight weight for second pointwise conv
 * @param output the output
 * @param height the input img height
 * @param width the input img width
 * @param in_channel the input_channel of the img
 * @param out_channel the output_channel of the output
 * @param kernel_height the depthwise kernel_height
 * @param kernel_width the depthwise kernel width
 */
void init_conv_conv_fusion_data(float* input, float* weight1, float* weight2, float* output,
  const int height, const int width,
  const int kernel1_height, const int kernel1_width, const int kernel1_in_channel, const int kernel1_out_channel, 
  const int kernel2_height, const int kernel2_width, const int kernel2_in_channel, const int kernel2_out_channel){
    srand (time(NULL));
  for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int ic = 0; ic < kernel1_in_channel; ++ic) {
        input[h*width*kernel1_in_channel + w*kernel1_in_channel + ic] = rand() % 10;
        // input[h*width*kernel1_in_channel + w*kernel1_in_channel + ic] = 1;
        // if(ic%2==0){input[h*width*in_channel + w*in_channel + ic] = 1;}
        // else{input[h*width*in_channel + w*in_channel + ic] = 2;}
        // input[h*width*kernel1_in_channel + w*kernel1_in_channel + ic] = rand() % 10;
      }
    }
  }
  
  for (int h = 0; h < kernel1_height; ++h) {
    for (int w = 0; w < kernel1_width; ++w) {
      for (int ic = 0; ic < kernel1_in_channel; ++ic) {
        for(int oc = 0; oc < kernel1_out_channel; ++oc) {
          weight1[h * kernel1_width * kernel1_in_channel * kernel1_out_channel 
            + w * kernel1_in_channel * kernel1_out_channel + ic * kernel1_out_channel + oc] = rand() % 10;
        }
      }
    }
  }

  for (int h = 0; h < kernel2_height; ++h) {
    for (int w = 0; w < kernel2_width; ++w) {
      for (int ic = 0; ic < kernel2_in_channel; ++ic) {
        for(int oc = 0; oc < kernel2_out_channel; ++oc) {
          weight2[h * kernel2_width * kernel2_in_channel * kernel2_out_channel 
            + w * kernel2_in_channel * kernel2_out_channel + ic * kernel2_out_channel + oc] = rand() % 10;
        }
      }
    }
  }

  for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int oc = 0; oc < kernel2_out_channel; ++oc) {
        output[h*width*kernel2_out_channel + w*kernel2_out_channel + oc] = 0;
      }
    }
  }
}


void pointwise_conv(float* input, float* pw_weight, float* output,
  const int height, const int width, const int in_channel, const int  out_channel){
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for(int oc = 0; oc<out_channel; ++oc){
        float sum = 0;
        for (int ic = 0; ic < in_channel; ++ic) {
          sum += input[h*width*in_channel + w*in_channel + ic] * pw_weight[oc*in_channel + ic];
        }
        output[h*width*out_channel+ w*out_channel + oc] = sum;
      }
    }
  }
}
  

#endif