
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>

#include <assert.h>
#include <stdio.h>


#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

int64_t shape_size(std::vector<int> shape){
    int64_t total_size = 1;
    for(auto s: shape){
        total_size *= s;
    }
    return total_size;
}


void benchmark_block_fusion_group_conv(){
    int num_inputs = 1;
    int num_weights = 64;
    std::vector<int> input_shape = {1, 256, 56, 56};
    std::vector<int> weight_shape = {4,256,1,1};
    std::vector<int> output_shape = {1, 4, 56, 56};
    std::vector<std::vector<int>> all_input_shapes;
    std::vector<std::vector<int>> all_weight_shapes;
    std::vector<std::vector<int>> all_output_shapes;
    for(int i=0; i<num_inputs; ++i){
        all_input_shapes.push_back(input_shape);
    }
    for(int i=0; i<num_weights; ++i){
        all_weight_shapes.push_back(weight_shape);
        all_output_shapes.push_back(output_shape);
    }
    // Generate host and device inputs
    float** inputs = (float**)malloc(num_inputs*sizeof(float*));
    float** weights = (float**)malloc(num_weights*sizeof(float*));
    float** outputs = (float**)malloc(num_weights*sizeof(float*));
    float** bias = (float**)malloc(num_weights*sizeof(float*));
    for(int i=0; i<num_inputs; ++i){
        inputs[i] = (float*)malloc(shape_size(all_input_shapes[i]) * sizeof(float));
    }
    for(int i=0; i<num_weights; ++i){
        weights[i] = (float*)malloc(shape_size(all_weight_shapes[i]) * sizeof(float));
        outputs[i] = (float*)malloc(shape_size(all_output_shapes[i]) * sizeof(float));
        bias[i] = (float*)malloc(shape_size(all_output_shapes[i]) * sizeof(float));
    }
    // Allocate device
    float** d_inputs = (float**)malloc(num_inputs*sizeof(float*));
    float** d_weights = (float**)malloc(num_weights*sizeof(float*));
    float** d_outputs = (float**)malloc(num_weights*sizeof(float*));
    float** d_bias = (float**)malloc(num_weights*sizeof(float*));
    for(int i=0; i<num_inputs; ++i){
        cudaMalloc((void**)&(d_inputs[i]), shape_size(all_input_shapes[i]) * sizeof(float));
        checkCuda(cudaMemcpy(d_inputs[i], inputs[i], shape_size(all_input_shapes[i]) * sizeof(float), cudaMemcpyHostToDevice));
    }
    for(int i=0; i<num_weights; ++i){
        cudaMalloc((void**)&(d_weights[i]), shape_size(all_weight_shapes[i]) * sizeof(float));
        checkCuda(cudaMemcpy(d_weights[i], weights[i], shape_size(all_weight_shapes[i]) * sizeof(float), cudaMemcpyHostToDevice));
        cudaMalloc((void**)&(d_outputs[i]), shape_size(all_output_shapes[i]) * sizeof(float));
        cudaMalloc((void**)&(d_bias[i]), shape_size(all_output_shapes[i]) * sizeof(float));
        checkCuda(cudaMemcpy(d_bias[i], bias[i], shape_size(all_outputs_shapes[i]) * sizeof(float), cudaMemcpyHostToDevice));
    }
    

}