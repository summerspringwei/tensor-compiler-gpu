
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


#endif
