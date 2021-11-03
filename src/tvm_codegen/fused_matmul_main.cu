
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



 extern void BlockFusion_matmul_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, 
  float* d_input, float* d_filter1, float* d_output1, float* d_filter2, float* d_output2);

  extern "C" __global__ void g_d2l_matmul_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute);
  extern "C" __global__ void __launch_bounds__(256) fused_matmul_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);
  extern "C" __global__ void mmult_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);


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

void benchmark_tvmgen_matmul(int argc, char** argv){
  int type = 0;
  if (argc > 1){
    type = atoi(argv[1]);
  }
  // const size_t  batch=1, in_height = 73, in_width = 73, in_channel = 160;
  // const size_t kernel_height = 1, kernel_width = 1, out_channel = 64;
  const size_t  batch=1, in_height = 56, in_width = 56, in_channel = 64;
  const size_t kernel_height = 1, kernel_width = 1, out_channel = 512;
  std::vector<float> input(batch * in_height * in_width * in_channel);
  std::vector<float> filter1(kernel_height * kernel_width * in_channel * out_channel);
  std::vector<float> filter2(kernel_height * kernel_width * in_channel * out_channel);
  std::vector<float> output1(batch * in_height * in_width * out_channel);
  std::vector<float> output2(batch * in_height * in_width * out_channel);
  std::vector<float> fused_filter(kernel_height * kernel_width * in_channel * out_channel * 2);
  std::vector<float> fused_output(batch * in_height * in_width * out_channel * 2);
  for(size_t i=0; i<input.size(); ++i){
    input[i] = 1;
  }
  for(size_t i=0; i<filter1.size(); ++i){
    filter1[i] = 1;
    filter2[i] = 1;
    fused_filter[2*i] = 1;
    fused_filter[2*i+1] = 1;
  }
  
  float* d_input = nullptr;
  float* d_filter1 = nullptr, *d_output1 = nullptr;
  float* d_filter2 = nullptr, *d_output2 = nullptr;
  float* d_fused_filter = nullptr, *d_fused_output = nullptr;
  cudaMalloc((void**)&d_input, sizeof(float) * input.size());
  cudaMalloc((void**)&d_filter1, sizeof(float) * filter1.size());
  cudaMalloc((void**)&d_output1, sizeof(float) * output1.size());
  cudaMalloc((void**)&d_filter2, sizeof(float) * filter2.size());
  cudaMalloc((void**)&d_output2, sizeof(float) * output2.size());
  cudaMalloc((void**)&d_fused_filter, sizeof(float) * fused_filter.size());
  cudaMalloc((void**)&d_fused_output, sizeof(float) * fused_output.size());

  //GPU time measurement
  float ms_max = std::numeric_limits<float>::min();
  float ms_min = std::numeric_limits<float>::max();
  float ms_total, ms_i;
  cudaEvent_t start_i, stop_i;
  cudaEventCreate(&start_i);
  cudaEventCreate(&stop_i);
  
  checkCuda(cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_filter1, filter1.data(), sizeof(float) * filter1.size() , cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_filter2, filter2.data(), sizeof(float) * filter2.size() , cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_fused_filter, fused_filter.data(), sizeof(float) * fused_filter.size() , cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(4, 4, 1);
  dim3 numBlocks(8, 1, 5329);
  int steps = 100;
  // Warm up
  for (int i_=0; i_<steps; i_++) {
    // Run in serial default
    if(type==0){
      g_d2l_matmul_kernel0<<<dim3(64/16/4*2, 73*73/16/8+1, 1), dim3(16, 16, 1)>>>(d_input, d_filter1, d_output1);
      g_d2l_matmul_kernel0<<<dim3(64/16/4*2, 73*73/16/8+1, 1), dim3(16, 16, 1)>>>(d_input, d_filter2, d_output2);
    }
    // Run in block fusion
    else if(type==1){
      BlockFusion_matmul_Call(dim3(64/16/4*2, (73*73/16/8+1)*2, 1), dim3(16, 16, 1), 0, 0, d_input, d_filter1, d_output1, d_filter2, d_output2);
    }
    // Run in fused
    else if (type==2){
      // fused_matmul_kernel0<<<dim3(64/16/4*2, 73*73/16/8+1, 1), dim3(16, 16, 1)>>>(d_input, d_fused_filter, d_fused_output);
      // mmult_kernel0<<<dim3(2, 84, 1), dim3(8, 8, 1)>>>(d_input, d_fused_filter, d_fused_output);
      mmult_kernel0<<<dim3(8, 98, 1), dim3(32, 4, 1)>>>(d_input, d_filter1, d_output1);
    }
    cudaDeviceSynchronize();
  }
  //time measurement
  ms_total = 0;
  
  cudaProfilerStart();
  for (int i_=0; i_<steps; i_++)
  {
    cudaEventRecord(start_i, 0);
    // tvmgen_thread_fused_nn_conv2d_kernel0<<<numBlocks, threadsPerBlock>>>(d_input, d_filter1, d_output1, d_filter2, d_output2);
    // tvmgen_default_fused_nn_conv2d_kernel0<<<numBlocks, threadsPerBlock>>>(d_input, d_filter1, d_output1);
    // tvmgen_default_fused_nn_conv2d_kernel0<<<numBlocks, threadsPerBlock>>>(d_input, d_filter2, d_output2);
    // mmult_kernel0<<<dim3(42, 1, 1), dim3(64, 8, 1)>>>(d_input, d_filter2, d_output2);
    // Run in serial default
    if(type==0){
      g_d2l_matmul_kernel0<<<dim3(64/16/4*2, 73*73/16/8+1, 1), dim3(16, 16, 1)>>>(d_input, d_filter1, d_output1);
      g_d2l_matmul_kernel0<<<dim3(64/16/4*2, 73*73/16/8+1, 1), dim3(16, 16, 1)>>>(d_input, d_filter2, d_output2);
    }
    // Run in block fusion
    else if(type==1){
      BlockFusion_matmul_Call(dim3(64/16/4*2, (73*73/16/8+1)*2, 1), dim3(16, 16, 1), 0, 0, d_input, d_filter1, d_output1, d_filter2, d_output2);
    }
    // Run in fused
    else if (type==2){
      // fused_matmul_kernel0<<<dim3(64/16/4*2, 73*73/16/8+1, 1), dim3(16, 16, 1)>>>(d_input, d_fused_filter, d_fused_output);
      // mmult_kernel0<<<dim3(2, 84, 1), dim3(8, 8, 1)>>>(d_input, d_fused_filter, d_fused_output);
      mmult_kernel0<<<dim3(8, 98, 1), dim3(32, 4, 1)>>>(d_input, d_filter1, d_output1);
    }
    cudaEventRecord(stop_i, 0);
    cudaEventSynchronize(stop_i);
    cudaEventElapsedTime(&ms_i, start_i, stop_i);
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();                                                   
    if (result != cudaSuccess)                                                                 
    {                                                                                          
        const char* msg = cudaGetErrorString(result);                                          
        std::stringstream safe_call_ss;                                                        
        safe_call_ss << "\nerror: " << " failed with error"                                    
                      << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  
        throw std::runtime_error(safe_call_ss.str());                                          
    }
    // printf("Iteration time %f ms\n", ms_i);
    ms_total += ms_i;
    if (ms_i > ms_max)  ms_max = ms_i;
    if (ms_i < ms_min) ms_min = ms_i;
  }
  if(type==0 || type==1){
    checkCuda(cudaMemcpy(output1.data(), d_output1, sizeof(float) * output1.size() , cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(output2.data(), d_output2, sizeof(float) * output2.size() , cudaMemcpyDeviceToHost));
    print_and_check(output1, 160.0);
    print_and_check(output2, 160.0);
  }else{
    // checkCuda(cudaMemcpy(fused_output.data(), d_fused_output, sizeof(float) * fused_output.size() , cudaMemcpyDeviceToHost));
    // print_and_check(fused_output, 160.0);
  }
  cudaProfilerStop();
  cudaDeviceSynchronize();
  printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total / steps);
  cudaFree(d_input);
  cudaFree(d_filter1);
  cudaFree(d_filter2);
  cudaFree(d_output1);
  cudaFree(d_output2);
  cudaFree(d_fused_filter);
  cudaFree(d_fused_output);
}

int main(int argc, char** argv) {
  benchmark_tvmgen_matmul(argc, argv);
  return 0;
}

