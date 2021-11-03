#ifndef UTILS_H
#define UTILS_H

#include <vector>

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



#endif