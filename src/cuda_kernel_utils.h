#ifndef CUDA_KERNEL_UTILS
#define CUDA_KERNEL_UTILS

#define FULL_MASK 0xffffffff
#define warpSize 32

__inline__ __device__ 
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}

__inline__ __device__
half warpReduceSum(half val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val = __hadd(val, __shfl_down_sync(FULL_MASK, val, offset));
  return val;
}

__inline__ __device__
half2 warpReduceSum(half2 val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val = __hadd2(val, __shfl_down_sync(FULL_MASK, val, offset));
  return val;
}

__inline__ __device__
float blockReduceSum(float val) {
  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  val = warpReduceSum(val);     // Each warp performs partial reduction
  if (lane==0) shared[wid]=val; // Write reduced value to shared memory
  __syncthreads();              // Wait for all partial reductions
  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp
  return val;
}

__device__ __forceinline__ float sigmoid(float x){
    return (1.0f / (1+exp(-x)));
}

#endif