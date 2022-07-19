
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <iostream>

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
// #if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
// #endif
  return result;
}

__inline__ __device__
half warpReduceSum(half val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
    // val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

#define CUBLAS_CHECK(func)                                                     \
    do {                                                                       \
        cublasStatus_t e = (func);                                             \
        if (e != CUBLAS_STATUS_SUCCESS) {                                      \
            std::stringstream safe_call_ss;                                    \
            safe_call_ss << "\nerror: " #func " failed with error"             \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__   \
                         << "\nmsg: " << e;                                    \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

// void my_compare(half* src, half* dst, int m, int n, float rotl, float aotl){
//   int error_cnt = 0;
//   for(int i=0; i<m; ++i){
//     for(int j=0; j<n; ++j){
//       float a = __half2float(src[i*n+j]);
//       float b = __half2float(dst[i*n+j]);
//       if(std::abs(a - b) > 
//         rotl * std::abs(a) + aotl){
//         printf("diff: <%d, %d> %f %f\n", i, j, a, b);
//         error_cnt++;
//         if(error_cnt > 100000){
//           return;
//         }
//       }
//     }
//   }
// }



// void create_my_query_key(){
//   // auto v_query_cpu = torch::empty({num_heads, max_seq_length, hidden_size}, options_fp16_cpu);
//   // auto v_key_cpu = torch::empty({num_heads, max_seq_length, hidden_size}, options_fp16_cpu);
//   float* v_query = (float*)malloc(max_seq_length*d_model*sizeof(float));
//   float* v_key = (float*)malloc(max_seq_length*d_model*sizeof(float));
//   for(int i=0; i<num_heads; ++i){
//     for(int j=0; j<max_seq_length; ++j){
//       for(int k=0; k<hidden_size; ++k){
//         // v_query_cpu[i][j][k] =__float2half((j%10)/10.0);
//         // v_key_cpu[i][j][k] =__float2half((j%10)/10.0); 
//         int idx = i*max_seq_length*hidden_size + j * hidden_size + k;
//         // v_query[idx] = __float2half((j%10)/10.0);
//         // v_key[idx] = __float2half((j%384)/10.0);
//         // v_key[idx] = (j%10)/10.0;
//         v_key[idx] = 1;
//         v_query[idx] = 1;
//         // if(j<max_seq_length/2){
//         //   v_query[idx] = (j%10)/10.0;
//         // }else{
//         //   v_query[idx] = 1;
//         // }
//       }
//     }
//   }
//   // auto v_query = v_query_cpu.to(torch::kCUDA);
//   // auto v_key = v_key_cpu.to(torch::kCUDA);
// torch::Tensor my_query = torch::from_blob(v_query, {num_heads, max_seq_length, hidden_size})
//     .toType(torch::kFloat16).to(torch::kCUDA);
//   torch::Tensor my_key = torch::from_blob(v_key, {num_heads, max_seq_length, hidden_size})
//     .toType(torch::kFloat16).to(torch::kCUDA);
// }


void check_compatability(int numThreads, int sharedMemSize, void* cuda_kernel){
  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  if(supportsCoopLaunch){
    printf("Device support CoopLaunch\n");
  }
  cudaDeviceProp deviceProp; \
  cudaGetDeviceProperties(&deviceProp, dev); \
  int numBlocksPerSm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, cuda_kernel, numThreads, sharedMemSize); 
  printf("fused_fc_fc: OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);
}


#endif
