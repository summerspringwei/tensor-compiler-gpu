
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


#define CUBLAS_SAFE_CALL(func)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t e = (func);                                                                 \
        if (e != CUBLAS_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
   #define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)

#define CUDNN_SAFE_CALL(func)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            const char* msg = cudnnGetErrorString(e);                                              \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
  
     

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
extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_concatenate_kernel0(float* __restrict__ T_concat, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 2048) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) >> 7)) < 5329) {
      if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)) < 682112) {
        T_concat[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)blockIdx.x) * 1024)) + ((int)threadIdx.x)))] = ((64 <= (((int)threadIdx.x) & 127)) ? placeholder[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 131072) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 7) * 64)) + (((int)threadIdx.x) & 127)) - 64))] : placeholder1[(((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 131072) + (((int)blockIdx.x) * 512)) + ((((int)threadIdx.x) >> 7) * 64)) + (((int)threadIdx.x) & 127)))]);
      }
    }
  }
}
// input: (1, 73, 73, 160), filter (1, 1, 160, 64), output (1, 73, 73, 64)
// <<< (8, 1, 5329), (4, 4, 1)>>>
extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_nn_conv2d_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ Conv2dOutput) {
  // printf("%d %d\n", threadIdx.x, threadIdx.y);
  float Conv2dOutput_local[2];
  __shared__ float PaddedInput_shared[24];
  __shared__ float placeholder_shared[256];
  float PaddedInput_shared_local[1];
  float placeholder_shared_local[2];
  for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
    Conv2dOutput_local[(ff_c_init)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 10; ++rc_outer) {// Every thread compute 2 output
    __syncthreads();
    // each thread get continues 10 input channel elements
    PaddedInput_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)))] = 
        placeholder[(((((((int)blockIdx.z) * 160) + (rc_outer * 16)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)))];
    for (int ax2_ax3_fused_outer_outer = 0; ax2_ax3_fused_outer_outer < 8; ++ax2_ax3_fused_outer_outer) {
      // ([0,8) * 32 + (threadIdx.y*4 + threadIdx) >> 3 * 16) +  ((threadIdx.y * 4 + threadIdx.x) & 7)
      // [0,8) * 32 + {[0,8] | 16 + [0, 8]} 
      // Every iteration jump 32
      placeholder_shared[((((ax2_ax3_fused_outer_outer * 32) + ((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) >> 3) * 16)) 
          + (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) & 7)))] = 
      // (([0,10) * 1024 + [0,8) * 128) + (threadIdx.y*4 + threadIdx) * 8) + blockIdx.x * 8 + ((threadIdx.y * 4 + threadIdx.x) & 7)
      // [0, 9*1024] + [0, 1024) + [0, 16) >> 3 * 16 + [0, 16) * 8 + [0, 16) & 7
      // [0, 9*1024] + [0, 1024) + { (threadIdx.x, threadIdx.y) in [0,8)  varies 64 | (threadIdx.x, threadIdx.y) in [8,16) varies 16+ [64, 128) + [8, 16)}
      // Every iteration jump 128 
      placeholder1[((((((rc_outer * 1024) + (ax2_ax3_fused_outer_outer * 128)) + ((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) >> 3) * 64)) 
          + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) & 7)))];
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      if (((int)threadIdx.y) < 1) {
        PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 24) + rc_inner))];
      }
      for (int ax3 = 0; ax3 < 2; ++ax3) {
        placeholder_shared_local[(ax3)] = placeholder_shared[((((rc_inner * 16) + (((int)threadIdx.x) * 2)) + ax3))];
      }
      for (int ff_c = 0; ff_c < 2; ++ff_c) {
        if (((int)threadIdx.y) < 1) {
          Conv2dOutput_local[(ff_c)] = (Conv2dOutput_local[(ff_c)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(ff_c)]));
        }
      }
    }
  }
  // Reduce here
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    if (((int)threadIdx.y) < 1) {
      Conv2dOutput[((((((((int)threadIdx.y) * 341056) + (((int)blockIdx.z) * 64)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + ff_inner))] = Conv2dOutput_local[(ff_inner)];
    }
  }
}



extern "C" __global__ void __launch_bounds__(16) tvmgen_thread_fused_nn_conv2d_kernel0(
  float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ Conv2dOutput1,
  float* __restrict__ placeholder2, float* __restrict__ Conv2dOutput2) {
  // printf("%d %d\n", threadIdx.x, threadIdx.y);
  float Conv2dOutput_local1[2];
  float Conv2dOutput_local2[2];
  __shared__ float PaddedInput_shared[24];
  __shared__ float placeholder_shared1[256];
  __shared__ float placeholder_shared2[256];
  float PaddedInput_shared_local[1];
  float placeholder_shared_local1[2];
  float placeholder_shared_local2[2];
  for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
    Conv2dOutput_local1[(ff_c_init)] = 0.000000e+00f;
    Conv2dOutput_local2[(ff_c_init)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 10; ++rc_outer) {
    __syncthreads();
    PaddedInput_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)))] = 
        placeholder[(((((((int)blockIdx.z) * 160) + (rc_outer * 16)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)))];
    for (int ax2_ax3_fused_outer_outer = 0; ax2_ax3_fused_outer_outer < 8; ++ax2_ax3_fused_outer_outer) {
      int idx_placeholder_shared = ((((ax2_ax3_fused_outer_outer * 32) + ((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) >> 3) * 16)) 
          + (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) & 7)));
      int idx_placeholder = ((((((rc_outer * 1024) + (ax2_ax3_fused_outer_outer * 128)) + ((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) >> 3) * 64)) 
      + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) & 7)));
      placeholder_shared1[idx_placeholder_shared] = placeholder1[idx_placeholder];
      placeholder_shared2[idx_placeholder_shared] = placeholder2[idx_placeholder];
      // printf("%f\n", placeholder1[idx_placeholder]);
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      if (((int)threadIdx.y) < 1) {
        PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 24) + rc_inner))];
      }
      for (int ax3 = 0; ax3 < 2; ++ax3) {
        int idx_ax3_shared = ((((rc_inner * 16) + (((int)threadIdx.x) * 2)) + ax3));
        placeholder_shared_local1[(ax3)] = placeholder_shared1[idx_ax3_shared];
        placeholder_shared_local2[(ax3)] = placeholder_shared2[idx_ax3_shared];
      }
      for (int ff_c = 0; ff_c < 2; ++ff_c) {
        if (((int)threadIdx.y) < 1) {
          Conv2dOutput_local1[(ff_c)] = (Conv2dOutput_local1[(ff_c)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local1[(ff_c)]));
          Conv2dOutput_local2[(ff_c)] = (Conv2dOutput_local2[(ff_c)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local2[(ff_c)]));
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    if (((int)threadIdx.y) < 1) {
      int idx_conv2d_output = ((((((((int)threadIdx.y) * 341056) + (((int)blockIdx.z) * 64)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + ff_inner));
      Conv2dOutput1[idx_conv2d_output] = Conv2dOutput_local1[(ff_inner)];
      Conv2dOutput2[idx_conv2d_output] = Conv2dOutput_local2[(ff_inner)];
    }
  }
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

//  __forceinline__ __device__ unsigned lane_id()
// {
//     unsigned ret; 
//     asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
//     return ret;
// }

// __forceinline__ __device__ unsigned warp_id()
// {
//     // this is not equal to threadIdx.x / 32
//     unsigned ret; 
//     asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
//     return ret;
// }

 extern "C" __global__ void __launch_bounds__(512) mmult_kernel0(float* __restrict__ C, float* __restrict__ A, float* __restrict__ B) {
  //  printf("lane id: %d, warp id: %d threadIdx(%d %d)\n", lane_id(), warp_id(), threadIdx.x, threadIdx.y);
   for (int i_inner_outer = 0; i_inner_outer < 16; ++i_inner_outer) {
     if ((((((int)blockIdx.x) * 128) + (i_inner_outer * 8)) + ((int)threadIdx.y)) < 5329) {
       C[(((((((int)blockIdx.x) * 8192) + (i_inner_outer * 512)) + (((int)threadIdx.y) * 64)) + ((int)threadIdx.x)))] = 0.000000e+00f;
     }
     if ((((((int)blockIdx.x) * 128) + (i_inner_outer * 8)) + ((int)threadIdx.y)) < 5329) {
       C[((((((((int)blockIdx.x) * 8192) + (i_inner_outer * 512)) + (((int)threadIdx.y) * 64)) + ((int)threadIdx.x)) + 341056))] = 0.000000e+00f;
     }
     for (int rik = 0; rik < 160; ++rik) {
       if ((((((int)blockIdx.x) * 128) + (i_inner_outer * 8)) + ((int)threadIdx.y)) < 5329) {
         int a_idx = (((((((int)blockIdx.x) * 20480) + (i_inner_outer * 1280)) + (((int)threadIdx.y) * 160)) + rik));
         int c_idx = (((((((int)blockIdx.x) * 8192) + (i_inner_outer * 512)) + (((int)threadIdx.y) * 64)) + ((int)threadIdx.x)));
         C[c_idx] = (C[c_idx] + (A[a_idx] * B[(((rik *   64) + ((int)threadIdx.x)))]));
         C[c_idx + 341056] = C[c_idx + 341056] + (A[a_idx] * B[((((rik * 64) + ((int)threadIdx.x)) + 10240))]);
       }
     }
   }
 }


 extern void BlockFusion_matmul_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, 
  float* d_input, float* d_filter1, float* d_output1, float* d_filter2, float* d_output2);

  extern "C" __global__ void g_d2l_matmul_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute);
  extern "C" __global__ void __launch_bounds__(256) fused_matmul_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);

void benchmark_tvmgen_thread_fused_nn_conv2d_kernel0(){
  const size_t  batch=1, in_height = 73, in_width = 73, in_channel = 160;
  const size_t kernel_height = 1, kernel_width = 1, out_channel = 64;
  std::vector<float> input(batch * in_height * in_width * in_channel);
  std::vector<float> filter1(kernel_height * kernel_width * in_channel * out_channel);
  std::vector<float> filter2(kernel_height * kernel_width * in_channel * out_channel);
  std::vector<float> output1(batch * in_height * in_width * out_channel);
  std::vector<float> output2(batch * in_height * in_width * out_channel);
  for(size_t i=0; i<input.size(); ++i){
    input[i] = 1;
  }
  for(size_t i=0; i<filter1.size(); ++i){
    filter1[i] = 1;
    filter2[i] = 1;
  }
  
  float* d_input = nullptr;
  float* d_filter1 = nullptr, *d_output1 = nullptr;
  float* d_filter2 = nullptr, *d_output2 = nullptr;
  cudaMalloc((void**)&d_input, sizeof(float) * input.size());
  cudaMalloc((void**)&d_filter1, sizeof(float) * filter1.size());
  cudaMalloc((void**)&d_output1, sizeof(float) * output1.size());
  cudaMalloc((void**)&d_filter2, sizeof(float) * filter2.size());
  cudaMalloc((void**)&d_output2, sizeof(float) * output2.size());

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

  dim3 threadsPerBlock(4, 4, 1);
  dim3 numBlocks(8, 1, 5329);
  //time measurement
  ms_total = 0;
  int steps = 10000;
  cudaProfilerStart();
  for (int i_=0; i_<steps; i_++)
  {
    cudaEventRecord(start_i, 0);
    // tvmgen_thread_fused_nn_conv2d_kernel0<<<numBlocks, threadsPerBlock>>>(d_input, d_filter1, d_output1, d_filter2, d_output2);
    // tvmgen_default_fused_nn_conv2d_kernel0<<<numBlocks, threadsPerBlock>>>(d_input, d_filter1, d_output1);
    // tvmgen_default_fused_nn_conv2d_kernel0<<<numBlocks, threadsPerBlock>>>(d_input, d_filter2, d_output2);
    // mmult_kernel0<<<dim3(42, 1, 1), dim3(64, 8, 1)>>>(d_input, d_filter2, d_output2);
    // Run in serial default
    // g_d2l_matmul_kernel0<<<dim3(73*73/16/8+1, 64/16/4*2, 1), dim3(16, 16, 1)>>>(d_input, d_filter1, d_output1);
    // g_d2l_matmul_kernel0<<<dim3(73*73/16/8+1, 64/16/4*2, 1), dim3(16, 16, 1)>>>(d_input, d_filter2, d_output2);
    // Run in block fusion
    BlockFusion_matmul_Call(dim3(73*73/16/8+1, 64/16/4*2, 1), dim3(16, 16, 1), 0, 0, d_input, d_filter1, d_output1, d_filter2, d_output2);
    // Run in fused
    // fused_matmul_kernel0<<<dim3(73*73/16/8+1, 64/16/4, 1), dim3(16, 16, 1)>>>(d_input, d_filter2, d_output2);

    
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
    printf("Iteration time %f ms\n", ms_i);
    ms_total += ms_i;
    if (ms_i > ms_max)  ms_max = ms_i;
    if (ms_i < ms_min) ms_min = ms_i;
  }
  checkCuda(cudaMemcpy(output1.data(), d_output1, sizeof(float) * output1.size() , cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(output2.data(), d_output2, sizeof(float) * output2.size() , cudaMemcpyDeviceToHost));
    for(int i=0; i<16; ++i){
      printf("%f ", output1[i]);
    }printf("\n");

  cudaProfilerStop();
  cudaDeviceSynchronize();
  printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total / steps);
  cudaFree(d_input);
  cudaFree(d_filter1);
  cudaFree(d_filter2);
  cudaFree(d_output1);
  cudaFree(d_output2);
}

int main(int argc, char** argv) {
  benchmark_tvmgen_thread_fused_nn_conv2d_kernel0();
  return 0;
}

