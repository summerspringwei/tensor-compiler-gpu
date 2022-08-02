#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>

#include "torch/all.h"

#include "../../cuda_utils.h"
#include "torch_utils.h"


// __inline__ __device__ float warpReduceSum(float val) {
//     for (int mask = 16; mask > 0; mask >>= 1)
//         val += __shfl_xor_sync(0xffffffff, val, mask, 32);
//     return val;
// }

__inline__ __device__ half2 warpReduceSumHalf2_v2(half2 val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    return val;
}

__inline__ __device__
half2 warpReduceSumHalf2_v1(half2 val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val = __hadd2(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}


// One thread reduce one line
// dim3(6*12, 1, 1), dim3(128, 1, 1)
__global__ void fused_layer_norm_v0(const half *__restrict__ src, 
  float * reduce_sum, float * reduce_variance, 
  const half * src_layer_norm, int64_t* profile_grid_clock){
    
  extern __shared__ half all_shared_mem[];
  half* acc_shared = all_shared_mem;
  

  const int row_tile_size = 64;
  const int col_tile_size = 64;
  const int max_seq_length = 384;
  const int d_model = 768;
  const int block_id_x = blockIdx.x / 12;
  const int block_id_y = blockIdx.x % 12;
  const int pad_row_tile_size = 64 + 8;
  int clock_idx = 0;
  const int warpIdx = threadIdx.x >> 5;
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  float* sum_shared = (float*)(acc_shared) + row_tile_size * pad_row_tile_size;
  float* sum_x2_shared = sum_shared + row_tile_size;
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();

  // Load to shared memory
  const half* src_base = src + block_id_x * d_model + block_id_y * col_tile_size;
  const int src_stride = (threadIdx.x >> 3) * d_model + (threadIdx.x & 7) * sizeof(float4);
  const int shared_stride = (threadIdx.x >> 3) * pad_row_tile_size + (threadIdx.x & 7) * sizeof(float4);
  const int kLoadRowsPerIter = 128 * sizeof(float4) / sizeof(half) / row_tile_size;
  for(int i=0; i<col_tile_size / kLoadRowsPerIter; ++i){
    ((float4*)(acc_shared + i * shared_stride))[0] = ((float4*)(src_base + i * src_stride))[0];
  }
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;

  // Compute mean and variance
  float sum = 0;
  float sum_x2 = 0;
  const int offset = (threadIdx.x >> 6) * 32;
  const int cmp_shared_stride = threadIdx.x * pad_row_tile_size + offset;
  for(int i=0; i<16; i++){
    half2 value = ((half2*)(acc_shared + cmp_shared_stride + i * 2))[0];
    sum += (__half2float(value.x) + __half2float(value.y));
    sum_x2 += (__half2float(value.x) * __half2float(value.x) + __half2float(value.y) * __half2float(value.y));
  }
  atomicAdd(reduce_sum + block_id_x * row_tile_size + threadIdx.x, sum);
  atomicAdd(reduce_variance + block_id_x * row_tile_size + threadIdx.x, sum_x2);
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;

  // grid.sync();
  // const half eps = 0.00001;
  // const half h_gama = 1.0;
  // const half h_beta = 0;
  // // First load to shared
  // float* shared_arr[] = {sum_shared, sum_x2_shared};
  // float* gm_arr[] = {reduce_sum, reduce_variance};
  // shared_arr[(threadIdx.x >> 6)][threadIdx.x & 63] = gm_arr[(threadIdx.x >> 6)][threadIdx.x & 63];
  // __syncthreads();
  // // Normalize
  // const int norm_stride = block_id_x * col_tile_size + (threadIdx.x >> 5);
  // for(int i=0; i<col_tile_size / kReduceRowsPerIter; ++i){
  //   half2 value = ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0];
  //   const int idx = norm_stride + i * kReduceRowsPerIter;
  //   float sum_x = sum_shared[idx];
  //   float sum_x_2 = sum_x2_shared[idx];
  //   half standard_deviation = __float2half(sqrt((sum_x_2 - (sum_x * sum_x)/d_model) / d_model + __half2float(eps)));
  //   half mean = __float2half(sum_x / d_model);
  //   half2 norm;
  //   norm.x = ((value.x - mean) / standard_deviation) * h_gama + h_beta;
  //   norm.y = ((value.y - mean) / standard_deviation) * h_gama + h_beta;
  //   ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0] = norm;
  // }
  // __syncthreads();
  // profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  for(int i=0; i<col_tile_size / kLoadRowsPerIter; ++i){
    ((float4*)(src_layer_norm + i * src_stride))[0] = ((float4*)(acc_shared + i * shared_stride))[0];
  }
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
}

// Using warp reduce
// dim3(6*12, 1, 1), dim3(128, 1, 1)
__global__ void fused_layer_norm_v1(const half *__restrict__ src, 
  float * reduce_sum, float * reduce_variance, 
  const half * src_layer_norm, int64_t* profile_grid_clock){
  int clock_idx = 0;
  const int warpIdx = threadIdx.x / 32;
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;  
  extern __shared__ half all_shared_mem[];
  half* acc_shared = all_shared_mem;
  const int row_tile_size = 64;
  const int col_tile_size = 64;
  const int max_seq_length = 384;
  const int d_model = 768;
  const int block_id_x = blockIdx.x / 12;
  const int block_id_y = blockIdx.x % 12;
  const int pad_row_tile_size = 64 + 0;
  

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  

  // Load to shared memory
  const half* src_base = src + block_id_x * d_model + block_id_y * col_tile_size;
  const int src_stride = (threadIdx.x >> 3) * d_model + (threadIdx.x & 7) * sizeof(float4);
  const int shared_stride = (threadIdx.x >> 3) * pad_row_tile_size + (threadIdx.x & 7) * sizeof(float4);
  const int kLoadRowsPerIter = 128 * sizeof(float4) / sizeof(half) / row_tile_size;
  for(int i=0; i<col_tile_size / kLoadRowsPerIter; ++i){
    ((float4*)(acc_shared + i * shared_stride))[0] = ((float4*)(src_base + i * src_stride))[0];
  }
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  // Compute mean and variance
  const int kReduceRowsPerIter = 128 * sizeof(half2) / sizeof(half) / row_tile_size;
  const int load_stride = (threadIdx.x >> 5) * pad_row_tile_size + (threadIdx.x & 31) * 2;
  half2 sum_x(half(0), half(0));
  half2 sum_x_2(half(0), half(0));
  for(int i=0; i<col_tile_size / kReduceRowsPerIter; ++i){
  // for(int i=0; i<4; ++i){
    half2 value = ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0];
    sum_x = warpReduceSumHalf2_v2(value);
    sum_x_2 = warpReduceSumHalf2_v2(value*value);
    if((threadIdx.x & 31) == 0){
      atomicAdd(reduce_sum + block_id_x * col_tile_size + (threadIdx.x >> 5), __half2float(sum_x.x + sum_x.y));
      atomicAdd(reduce_variance + block_id_x * col_tile_size + (threadIdx.x >> 5), __half2float(sum_x_2.x + sum_x_2.y));
    }
  }
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  // grid.sync();
  // const half eps = 0.00001;
  // const half h_gama = 1.0;
  // const half h_beta = 0;
  // // Normalize
  // for(int i=0; i<col_tile_size / kReduceRowsPerIter; ++i){
  //   half2 value = ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0];
  //   float sum_x = reduce_sum[block_id_x * col_tile_size + i * kReduceRowsPerIter];
  //   float sum_x_2 = reduce_variance[block_id_x * col_tile_size + i * kReduceRowsPerIter];
  //   half standard_deviation = __float2half(sqrt((sum_x_2 - (sum_x * sum_x)/d_model) / d_model + __half2float(eps)));
  //   half mean = __float2half(sum_x / d_model);
  //   half2 norm;
  //   norm.x = ((value.x - mean) / standard_deviation) * h_gama + h_beta;
  //   norm.y = ((value.y - mean) / standard_deviation) * h_gama + h_beta;
  //   ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0] = norm;
  // }

  __syncthreads();
  for(int i=0; i<col_tile_size / kLoadRowsPerIter; ++i){
    ((float4*)(src_layer_norm + i * src_stride))[0] = ((float4*)(acc_shared + i * shared_stride))[0];
  }
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
}


// Reduce with atomicAdd on shared memory
// dim3(6*12, 1, 1), dim3(128, 1, 1)
__global__ void fused_layer_norm_v2(const half *__restrict__ src, 
  float * reduce_sum, float * reduce_variance, 
  const half * src_layer_norm, int64_t* profile_grid_clock){
  int clock_idx = 0;
  const int warpIdx = threadIdx.x / 32;
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;  
  extern __shared__ half all_shared_mem[];
  half* acc_shared = all_shared_mem;
  
  const int row_tile_size = 64;
  const int col_tile_size = 64;
  const int max_seq_length = 384;
  const int d_model = 768;
  const int block_id_x = blockIdx.x / 12;
  const int block_id_y = blockIdx.x % 12;
  const int pad_row_tile_size = 64 + 0;

  float* sum_shared = (float*)(acc_shared) + row_tile_size * pad_row_tile_size;
  float* sum_x2_shared = sum_shared + row_tile_size;
  

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  

  // Load to shared memory
  const half* src_base = src + block_id_x * d_model + block_id_y * col_tile_size;
  const int src_stride = (threadIdx.x >> 3) * d_model + (threadIdx.x & 7) * sizeof(float4);
  const int shared_stride = (threadIdx.x >> 3) * pad_row_tile_size + (threadIdx.x & 7) * sizeof(float4);
  const int kLoadRowsPerIter = 128 * sizeof(float4) / sizeof(half) / row_tile_size;
  for(int i=0; i<col_tile_size / kLoadRowsPerIter; ++i){
    ((float4*)(acc_shared + i * shared_stride))[0] = ((float4*)(src_base + i * src_stride))[0];
  }
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  // Compute mean and variance
  const int kReduceRowsPerIter = 128 * sizeof(half2) / sizeof(half) / row_tile_size;
  const int load_stride = (threadIdx.x >> 5) * pad_row_tile_size + (threadIdx.x & 31) * 2;
  half2 sum_x(half(0), half(0));
  half2 sum_x_2(half(0), half(0));
  for(int i=0; i<col_tile_size / kReduceRowsPerIter; ++i){
    half2 value = ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0];
    atomicAdd(sum_shared + i * kReduceRowsPerIter + (threadIdx.x >> 5), __half2float(value.x) + __half2float(value.y));
    half2 value_x2 = value*value;
    atomicAdd(sum_x2_shared + i * kReduceRowsPerIter + (threadIdx.x >> 5), __half2float(value_x2.x) + __half2float(value_x2.y));
  }
  __syncthreads();
  if(threadIdx.x < 64){
    atomicAdd(reduce_sum + blockIdx.x * row_tile_size + threadIdx.x, sum_shared[threadIdx.x]);
  }else{
    atomicAdd(reduce_variance + blockIdx.x * row_tile_size + threadIdx.x - 64, sum_x2_shared[threadIdx.x - 64]);
  }
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  // grid.sync();
  // const half eps = 0.00001;
  // const half h_gama = 1.0;
  // const half h_beta = 0;
  // // Normalize
  // for(int i=0; i<col_tile_size / kReduceRowsPerIter; ++i){
  //   half2 value = ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0];
  //   float sum_x = reduce_sum[block_id_x * col_tile_size + i * kReduceRowsPerIter];
  //   float sum_x_2 = reduce_variance[block_id_x * col_tile_size + i * kReduceRowsPerIter];
  //   half standard_deviation = __float2half(sqrt((sum_x_2 - (sum_x * sum_x)/d_model) / d_model + __half2float(eps)));
  //   half mean = __float2half(sum_x / d_model);
  //   half2 norm;
  //   norm.x = ((value.x - mean) / standard_deviation) * h_gama + h_beta;
  //   norm.y = ((value.y - mean) / standard_deviation) * h_gama + h_beta;
  //   ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0] = norm;
  // }

  __syncthreads();
  for(int i=0; i<col_tile_size / kLoadRowsPerIter; ++i){
    ((float4*)(src_layer_norm + i * src_stride))[0] = ((float4*)(acc_shared + i * shared_stride))[0];
  }
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
}

// Reduce using atomicAdd on global memory
__global__ void fused_layer_norm_v3(const half *__restrict__ src, 
  float * reduce_sum, float * reduce_variance, 
  const half * src_layer_norm, int64_t* profile_grid_clock){
  int clock_idx = 0;
  const int warpIdx = threadIdx.x / 32;
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;  
  extern __shared__ half all_shared_mem[];
  half* acc_shared = all_shared_mem;
  
  const int row_tile_size = 64;
  const int col_tile_size = 64;
  const int max_seq_length = 384;
  const int d_model = 768;
  const int block_id_x = blockIdx.x / 12;
  const int block_id_y = blockIdx.x % 12;
  const int pad_row_tile_size = 64 + 0;
  
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  
  // Load to shared memory
  const half* src_base = src + block_id_x * d_model + block_id_y * col_tile_size;
  const int src_stride = (threadIdx.x >> 3) * d_model + (threadIdx.x & 7) * sizeof(float4);
  const int shared_stride = (threadIdx.x >> 3) * pad_row_tile_size + (threadIdx.x & 7) * sizeof(float4);
  const int kLoadRowsPerIter = 128 * sizeof(float4) / sizeof(half) / row_tile_size;
  for(int i=0; i<col_tile_size / kLoadRowsPerIter; ++i){
    ((float4*)(acc_shared + i * shared_stride))[0] = ((float4*)(src_base + i * src_stride))[0];
  }
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  // Compute mean and variance
  const int kReduceRowsPerIter = 128 * sizeof(half2) / sizeof(half) / row_tile_size;
  const int load_stride = (threadIdx.x >> 5) * pad_row_tile_size + (threadIdx.x & 31) * 2;
  const int global_stride = block_id_x * row_tile_size + (threadIdx.x >> 5);
  for(int i=0; i<col_tile_size / kReduceRowsPerIter; ++i){
    half2 value = ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0];
    const int g_idx = global_stride + i * kReduceRowsPerIter;
    atomicAdd(reduce_sum + g_idx, __half2float(value.x) + __half2float(value.y));
    half2 value_x2 = value*value;
    atomicAdd(reduce_variance + g_idx, __half2float(value_x2.x) + __half2float(value_x2.y));
  }
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  // grid.sync();
  // const half eps = 0.00001;
  // const half h_gama = 1.0;
  // const half h_beta = 0;
  // // Normalize
  // for(int i=0; i<col_tile_size / kReduceRowsPerIter; ++i){
  //   half2 value = ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0];
  //   float sum_x = reduce_sum[block_id_x * col_tile_size + i * kReduceRowsPerIter];
  //   float sum_x_2 = reduce_variance[block_id_x * col_tile_size + i * kReduceRowsPerIter];
  //   half standard_deviation = __float2half(sqrt((sum_x_2 - (sum_x * sum_x)/d_model) / d_model + __half2float(eps)));
  //   half mean = __float2half(sum_x / d_model);
  //   half2 norm;
  //   norm.x = ((value.x - mean) / standard_deviation) * h_gama + h_beta;
  //   norm.y = ((value.y - mean) / standard_deviation) * h_gama + h_beta;
  //   ((half2*)(acc_shared + i * kReduceRowsPerIter * pad_row_tile_size + load_stride))[0] = norm;
  // }
  // __syncthreads();
  
  for(int i=0; i<col_tile_size / kLoadRowsPerIter; ++i){
    ((float4*)(src_layer_norm + i * src_stride))[0] = ((float4*)(acc_shared + i * shared_stride))[0];
  }
  __syncthreads();
  profile_grid_clock[clock_idx * gridDim.x * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
}


void test_reduce(int round_count, int loop, int func_version){
  const int d_model = 768, max_seq_length = 384;
  auto options_fp16 = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_fp32 = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_int64 = torch::TensorOptions()
    .dtype(torch::kInt64)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  // auto src = torch::ones({max_seq_length, d_model}, options_fp16);
  auto src = torch::nn::init::uniform_(torch::rand({max_seq_length, d_model}, options_fp16), 0, 1);
  auto t_reduce_mean = torch::sum(src, 1, false, torch::kFloat32);
  auto t_reduce_variance = torch::sum(src*src, 1, false, torch::kFloat32);
  auto t_src_layer_norm = torch::layer_norm(src, {d_model,});
  auto reduce_mean = torch::zeros({max_seq_length,}, options_fp32);
  auto reduce_variance = torch::zeros({max_seq_length,}, options_fp32);
  auto src_layer_norm = torch::zeros({max_seq_length, d_model}, options_fp16);
  const int kProfileStages = 4;
  auto profile_clock = torch::zeros({kProfileStages, 72, 4}, options_int64);

  auto ptr_src = src.data<at::Half>();
  float* ptr_reduce_mean = reduce_mean.data<float>();
  float* ptr_reduce_variance = reduce_variance.data<float>();
  auto ptr_src_layer_norm = src_layer_norm.data<at::Half>();
  auto ptr_profile_clock = profile_clock.data<int64_t>();

  void * fused_kernel_args[] = {
    (void *)&(ptr_src), 
    (void *)&(ptr_reduce_mean), 
    (void *)&(ptr_reduce_variance), 
    (void *)&(ptr_src_layer_norm), 
    (void*)&(ptr_profile_clock)
  };
  auto multi_func = [&](int func_version) {
    switch (func_version)
    {
    case 0:
      checkCuda(cudaLaunchCooperativeKernel((void*)fused_layer_norm_v0, dim3(72, 1, 1), dim3(128, 1, 1), fused_kernel_args, 48*1024));
      break;
    case 1:
      checkCuda(cudaLaunchCooperativeKernel((void*)fused_layer_norm_v1, dim3(72, 1, 1), dim3(128, 1, 1), fused_kernel_args, 48*1024));
      break;
    case 2:
      checkCuda(cudaLaunchCooperativeKernel((void*)fused_layer_norm_v2, dim3(72, 1, 1), dim3(128, 1, 1), fused_kernel_args, 48*1024));
      break;
    case 3:
      checkCuda(cudaLaunchCooperativeKernel((void*)fused_layer_norm_v3, dim3(72, 1, 1), dim3(128, 1, 1), fused_kernel_args, 48*1024));
      break;
    default:
      break;
    }
  };
  multi_func(func_version);
  cudaDeviceSynchronize();

  my_compare(t_reduce_mean, reduce_mean, 1/16, 1024, 1);
  my_compare(t_reduce_variance, reduce_variance, 1/16, 1024, 1);
  // my_compare(t_src_layer_norm, src_layer_norm, 1/16, 1024);

  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  float min_avg = 1e10;
  for(int round =0; round<round_count; ++round){
    float ms = 0, latency_sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      multi_func(func_version);
      checkCuda( cudaEventRecord(stopEvent,0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      latency_sum += ms;
    }
    auto avg = latency_sum/loop;
    if(avg<min_avg){
      min_avg = avg;
    }
    printf("Run iter %d loops %d finished, avg %f us\n", round, loop, min_avg);
  }
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));

  for(int i=0; i<kProfileStages; ++i){
    for(int j=0; j<72; ++j){
      for(int k=0; k<4; ++k){
        if(i>0){
          printf("stage: %d, block: %d, warp: %d, latency: %ld\n", 
            i-1, j, k, profile_clock[i][j][k].item().toLong() - profile_clock[i-1][j][k].item().toLong());
        }
      }
    }
  }
}

int main(int argc, char** argv){
  int round = 1, loop = 1, type=0;
  if(argc>3){
    round = atoi(argv[1]);
    loop = atoi(argv[2]);
    type = atoi(argv[3]);
  }
  test_reduce(round, loop, type);
  return 0;
}