
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>

#include <assert.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "cuda_runtime.h"

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

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif

const size_t num_layer=10, num_hidden=96, num_timestep=100, batch=1;
const int kNumGatesInLstmCell = 8;


__global__ void __launch_bounds__(96) default_function_kernel0(float* __restrict__ compute, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  compute[(((((int)blockIdx.x) * 96) + ((int)threadIdx.x)))] = 0.000000e+00f;
  for (int rc = 0; rc < 96; ++rc) {
    compute[(((((int)blockIdx.x) * 96) + ((int)threadIdx.x)))] = (compute[(((((int)blockIdx.x) * 96) + ((int)threadIdx.x)))] + (placeholder[((((((int)blockIdx.x) >> 2) * 96) + rc))] * 
        placeholder1[((((((int)blockIdx.x) * 9216) + (rc * 96)) + ((int)threadIdx.x)))]));
  }
}

__device__ void lstm_wavefront(float* __restrict__ compute, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  compute[(((((int)blockIdx.x) * 96) + ((int)threadIdx.x)))] = 0.000000e+00f;
  for (int rc = 0; rc < 96; ++rc) {
    compute[(((((int)blockIdx.x) * 96) + ((int)threadIdx.x)))] = (compute[(((((int)blockIdx.x) * 96) + ((int)threadIdx.x)))] + (placeholder[((((((int)blockIdx.x) >> 2) * 96) + rc))] * 
        placeholder1[((((((int)blockIdx.x) * 9216) + (rc * 96)) + ((int)threadIdx.x)))]));
  }
}

__global__ void __launch_bounds__(96) lstm_wavefront_timesteps(float* __restrict__ compute, float* __restrict__ placeholder, float* __restrict__ placeholder1){
    for(int i=0; i<num_timestep; ++i){
        lstm_wavefront(compute, placeholder, placeholder1);
        // __threadfence();
    }
}


#define FUNC_CALL default_function_kernel0<<<numBlocks, threadsPerBlock>>>(d_output_buffer, d_output_buffer, d_weight_state_wavefront);
// #define FUNC_CALL lstm_wavefront_timesteps<<<numBlocks, threadsPerBlock>>>(d_output_buffer, d_output_buffer, d_weight_state_wavefront);

// num_layers: 10, timesteps: 100
// inputs_timestep: [1, 100, 128], outputs_timestep[1, 100, 128]
// input_wavefront: [1, 10, 128], state_wavefront: [1, 10, 128], weight_*_wavefront [40, 128, 128]
// c: [1, 10, 128], output_buffer:[1, 80, 128]
void benchmark_lstm_wavefront_magic(int argc, char** argv){
  
  // Allocate host data
  std::vector<float> input_timestep(num_timestep * num_hidden);
  std::vector<float> output_timestep(num_timestep * num_hidden);
  std::vector<float> input_wavefront(batch*num_layer*num_hidden);
  std::vector<float> c_wavefront(batch*num_layer*num_hidden);
  std::vector<float> h_wavefront(batch*num_layer*num_hidden);
  std::vector<float> weight_input_wavefront(4*num_layer*num_hidden*num_hidden);
  std::vector<float> weight_state_wavefront(8*num_layer*num_hidden*num_hidden);
  std::vector<float> output_buffer(8*num_layer*num_hidden);
  std::vector<float> bias(num_hidden);
  // Set host data
  for(int i=0; i<input_timestep.size(); ++i){
    input_timestep[i] = 1.0;
  }
  for(int i=0;i<weight_input_wavefront.size(); ++i){
    weight_input_wavefront[i] = 1.0/num_hidden;
    weight_state_wavefront[i] = 1.0/num_hidden;
  }
  for(int i=0; i<c_wavefront.size(); ++i){
    c_wavefront[i]=1.0;
    h_wavefront[i]=1.0;
  }
  // Allocate GPU
  float* d_inputs_timestep=nullptr, *d_outputs_timestep=nullptr;
  float* d_input_wavefront=nullptr, *d_c_wavefront=nullptr, *d_h_wavefront=nullptr;
  float* d_weight_input_wavefront=nullptr, *d_weight_state_wavefront=nullptr, *d_output_buffer=nullptr, *d_bias=nullptr;
  cudaMalloc((void**)&d_inputs_timestep, sizeof(float) * input_timestep.size());
  cudaMalloc((void**)&d_outputs_timestep, sizeof(float) * output_timestep.size());
  cudaMalloc((void**)&d_input_wavefront, sizeof(float) * input_wavefront.size());
  cudaMalloc((void**)&d_c_wavefront, sizeof(float) * c_wavefront.size());
  cudaMalloc((void**)&d_h_wavefront, sizeof(float) * h_wavefront.size());
  cudaMalloc((void**)&d_weight_input_wavefront, sizeof(float) * weight_input_wavefront.size());
  cudaMalloc((void**)&d_weight_state_wavefront, sizeof(float) * weight_state_wavefront.size());
  cudaMalloc((void**)&d_output_buffer, sizeof(float) * output_buffer.size());
  cudaMalloc((void**)&d_bias, sizeof(float) * bias.size());
  
  //GPU time measurement
  float ms_max = std::numeric_limits<float>::min();
  float ms_min = std::numeric_limits<float>::max();
  float ms_total, ms_i;
  cudaEvent_t start_i, stop_i;
  cudaEventCreate(&start_i);
  cudaEventCreate(&stop_i);

  checkCuda(cudaMemcpy(d_inputs_timestep, input_timestep.data(), sizeof(float) * input_timestep.size(), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_input_wavefront, input_wavefront.data(), sizeof(float) * input_wavefront.size() , cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_c_wavefront, c_wavefront.data(), sizeof(float) * c_wavefront.size() , cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_h_wavefront, h_wavefront.data(), sizeof(float) * h_wavefront.size() , cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_weight_input_wavefront, weight_input_wavefront.data(), sizeof(float) * weight_input_wavefront.size() , cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_weight_state_wavefront, weight_state_wavefront.data(), sizeof(float) * weight_state_wavefront.size() , cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_bias, bias.data(), sizeof(float) * bias.size() , cudaMemcpyHostToDevice));
  
  
  // Set shared memory for SM
  // int maxbytes = 1024*64; // 96 KB
  // cudaFuncSetAttribute(lstm_wavefront_magic, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
  // int carveout = 50; // prefer shared memory capacity 50% of maximum
  // Named Carveout Values:
  // carveout = cudaSharedmemCarveoutDefault;   //  (-1)
  // carveout = cudaSharedmemCarveoutMaxL1;     //   (0)
  // auto carveout = cudaSharedmemCarveoutMaxShared; // (100)
  // cudaFuncSetAttribute(lstm_wavefront_magic, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
  dim3 numBlocks(8*num_layer, 1, 1);
  dim3 threadsPerBlock(num_hidden, 1, 1);
  
  for(int t=0; t<num_timestep; ++t){
    FUNC_CALL
  }
  cudaDeviceSynchronize();
  
  auto result = cudaGetLastError();                                                   
  if (result != cudaSuccess)                                                                 
  {                                                                                          
      const char* msg = cudaGetErrorString(result);                                          
      std::stringstream safe_call_ss;                                                        
      safe_call_ss << "\nerror: " << " failed with error"                                    
                    << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  
      throw std::runtime_error(safe_call_ss.str());                                          
  }
  
  
  int steps = 1000;
  // Warm up
  for (int i=0; i<steps; i++) {
    for(int t=0; t<num_timestep; ++t){
        FUNC_CALL
    }
    cudaDeviceSynchronize();
  }
  result = cudaGetLastError();                                                   
  if (result != cudaSuccess)                                                                 
  {                                                                                          
      const char* msg = cudaGetErrorString(result);                                          
      std::stringstream safe_call_ss;                                                        
      safe_call_ss << "\nerror: " << " failed with error"                                    
                    << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  
      throw std::runtime_error(safe_call_ss.str());                                          
  }

  //time measurement
  ms_total = 0;
  
  cudaProfilerStart();
  for (int i_=0; i_<steps; i_++)
  {
    cudaEventRecord(start_i, 0);
    for(int t=0; t<num_timestep; ++t){
        FUNC_CALL
    }
    cudaEventRecord(stop_i, 0);
    cudaEventSynchronize(stop_i);
    cudaEventElapsedTime(&ms_i, start_i, stop_i);
    cudaDeviceSynchronize();
    
    ms_total += ms_i;
    if (ms_i > ms_max)  ms_max = ms_i;
    if (ms_i < ms_min) ms_min = ms_i;
  }
  cudaProfilerStop();
  cudaDeviceSynchronize();
  printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total / steps);
  result = cudaGetLastError();                                                   
  if (result != cudaSuccess)                                                                 
  {                                                                                          
      const char* msg = cudaGetErrorString(result);                                          
      std::stringstream safe_call_ss;                                                        
      safe_call_ss << "\nerror: " << " failed with error" 
                    << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  
      throw std::runtime_error(safe_call_ss.str());                                          
  }
  
  cudaFree(d_inputs_timestep);
  cudaFree(d_outputs_timestep);
  cudaFree(d_c_wavefront);
  cudaFree(d_h_wavefront);
  cudaFree(d_input_wavefront);
  cudaFree(d_weight_input_wavefront);
  cudaFree(d_weight_state_wavefront);
  cudaFree(d_bias);
  cudaFree(d_output_buffer);
}

int main(int argc, char** argv) {
  benchmark_lstm_wavefront_magic(argc, argv);
  return 0;
}