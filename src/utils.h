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


#endif