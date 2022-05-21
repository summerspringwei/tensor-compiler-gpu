#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>
#include <cuda/barrier>
#include "cooperative_groups/memcpy_async.h"


extern "C" __global__ void __launch_bounds__(128) default_function_kernel0(half* __restrict__ x, half* __restrict__ weight, half* __restrict__ add, half* __restrict__ short_cut) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half x_shared[4352];
  __shared__ half weight_shared[8704];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[1];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4*)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = ((uint4*)(x + (((((((((int)blockIdx.x) * 65536) + (ax0_ax1_fused_outer_outer_outer_outer * 16384)) + (((int)threadIdx.z) * 4096)) + ((((int)threadIdx.x) >> 4) * 2048)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(weight_shared + (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) + (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = ((uint4*)(weight + (((((((((int)blockIdx.y) * 131072) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16384)) + (((int)threadIdx.z) * 4096)) + ((((int)threadIdx.x) >> 4) * 2048)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[ax0_outer], ((half *)x_shared + (((ax0_outer * 2176) + (k_outer_inner * 16)))), 136);
      }
      (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], ((half *)weight_shared + (((((int)threadIdx.z) * 2176) + (k_outer_inner * 16)))), 136);
      for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
        (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[i_c_outer], x_shared_wmma_matrix_a[i_c_outer], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[i_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax0_outer_inner = 0; ax0_outer_inner < 2; ++ax0_outer_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + (((ax0_outer_inner * 1152) + (((int)threadIdx.z) * 16)))), T_dense_wmma_accumulator[ax0_outer_inner], 72, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    uint4 _1;
      uint4 _2 = ((uint4*)(x_shared + (((((i_inner_j_inner_fused_outer_outer_outer_outer * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
      uint4 _3 = ((uint4*)(short_cut + (((((((((int)blockIdx.x) * 65536) + (i_inner_j_inner_fused_outer_outer_outer_outer * 32768)) + (((int)threadIdx.z) * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
      ((half2*)(&(_1.x)))->x = (((half2*)(&(_2.x)))->x+((half2*)(&(_3.x)))->x);
      ((half2*)(&(_1.x)))->y = (((half2*)(&(_2.x)))->y+((half2*)(&(_3.x)))->y);
      ((half2*)(&(_1.y)))->x = (((half2*)(&(_2.y)))->x+((half2*)(&(_3.y)))->x);
      ((half2*)(&(_1.y)))->y = (((half2*)(&(_2.y)))->y+((half2*)(&(_3.y)))->y);
      ((half2*)(&(_1.z)))->x = (((half2*)(&(_2.z)))->x+((half2*)(&(_3.z)))->x);
      ((half2*)(&(_1.z)))->y = (((half2*)(&(_2.z)))->y+((half2*)(&(_3.z)))->y);
      ((half2*)(&(_1.w)))->x = (((half2*)(&(_2.w)))->x+((half2*)(&(_3.w)))->x);
      ((half2*)(&(_1.w)))->y = (((half2*)(&(_2.w)))->y+((half2*)(&(_3.w)))->y);
    ((uint4*)(add + (((((((((int)blockIdx.x) * 16384) + (i_inner_j_inner_fused_outer_outer_outer_outer * 8192)) + (((int)threadIdx.z) * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = _1;
  }
};

// <8, 8, 1>,<32, 1, 4>
__global__ void __launch_bounds__(128) fc2_16_16_2048_512_tvm(half* __restrict__ x, half* __restrict__ weight, half* __restrict__ add, half* __restrict__ short_cut) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half x_shared[4352];
  __shared__ half weight_shared[8704];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[1];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
  }
  
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4*)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = 
      ((uint4*)(x + (((((((((int)blockIdx.x) * 65536) + 
        (ax0_ax1_fused_outer_outer_outer_outer * 16384)) + 
        (((int)threadIdx.z) * 4096)) + 
        ((((int)threadIdx.x) >> 4) * 2048)) + 
        (k_outer_outer * 128)) + 
        ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(weight_shared + (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) + (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = ((uint4*)(weight + (((((((((int)blockIdx.y) * 131072) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16384)) + (((int)threadIdx.z) * 4096)) + ((((int)threadIdx.x) >> 4) * 2048)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        
      }
      (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], ((half *)weight_shared + (((((int)threadIdx.z) * 2176) + (k_outer_inner * 16)))), 136);
      for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
        (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[i_c_outer], x_shared_wmma_matrix_a[i_c_outer], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[i_c_outer]);
      }
    }
  }
  __syncthreads();
  
  for (int ax0_outer_inner = 0; ax0_outer_inner < 2; ++ax0_outer_inner) {
    // Guess through stride
    // ax0 * (16*72) + threadIdx.z * 16, thus, the x_shared is (32*64);
    // (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + (((ax0_outer_inner * 1152) + (((int)threadIdx.z) * 16)))), T_dense_wmma_accumulator[ax0_outer_inner], 72, nvcuda::wmma::mem_row_major);
    (void)nvcuda::wmma::store_matrix_sync(((half *)add + blockIdx.x * 32*512 + ax0_outer_inner * 16 * 512 + blockIdx.y * 64 + threadIdx.z * 16), T_dense_wmma_accumulator[ax0_outer_inner], 512, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  // Each warp computes 16*32 results, each thread needs to save 16 results,
  // each uint4 equals to 8 half, thus each thread save 2 times
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
  uint4 _1;
  uint4 _2 = ((uint4*)(x_shared + (((((i_inner_j_inner_fused_outer_outer_outer_outer * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
  uint4 _3 = ((uint4*)(short_cut + (((((((((int)blockIdx.x) * 65536) + (i_inner_j_inner_fused_outer_outer_outer_outer * 32768)) + (((int)threadIdx.z) * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) &  7) * 8)))))[0];
  ((half2*)(&(_1.x)))->x = (((half2*)(&(_2.x)))->x+((half2*)(&(_3.x)))->x);
  ((half2*)(&(_1.x)))->y = (((half2*)(&(_2.x)))->y+((half2*)(&(_3.x)))->y);
  ((half2*)(&(_1.y)))->x = (((half2*)(&(_2.y)))->x+((half2*)(&(_3.y)))->x);
  ((half2*)(&(_1.y)))->y = (((half2*)(&(_2.y)))->y+((half2*)(&(_3.y)))->y);
  ((half2*)(&(_1.z)))->x = (((half2*)(&(_2.z)))->x+((half2*)(&(_3.z)))->x);
  ((half2*)(&(_1.z)))->y = (((half2*)(&(_2.z)))->y+((half2*)(&(_3.z)))->y);
  ((half2*)(&(_1.w)))->x = (((half2*)(&(_2.w)))->x+((half2*)(&(_3.w)))->x);
  ((half2*)(&(_1.w)))->y = (((half2*)(&(_2.w)))->y+((half2*)(&(_3.w)))->y);
  // row = blockIdx.x * 32 + ijxxx * 16 + threadIdx.z * 4 + threadIdx.x/8  
  //    (blockIdx.x * (32 * 512) + ijxxx * (16*512) + threadIdx.z * (4*512) + threadIdx.x / 8 * 512)
  // col = blockIdx.y * 64 + (threadIdx.x % 8)*8
  ((uint4*)(add + (((((((((int)blockIdx.x) * 16384) + 
  (i_inner_j_inner_fused_outer_outer_outer_outer * 8192)) + 
  (((int)threadIdx.z) * 2048)) + 
  ((((int)threadIdx.x) >> 3) * 512)) + 
  (((int)blockIdx.y) * 64)) + 
  ((((int)threadIdx.x) & 7) * 8)))))[0] = _1;
  }
};



// <8, 8, 1>,<32, 1, 4>
__global__ void __launch_bounds__(128) fc2_16_16_2048_512_tvm_v2(half* __restrict__ x, half* __restrict__ weight, half* __restrict__ add, half* __restrict__ short_cut) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half x_shared[4352];
  __shared__ half weight_shared[8704];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[1];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
  }
  // 在k轴做了tiling，更好的数据局部性
  // output: (16*16, 512) / (16, 16) -> (16, 32), 每个warp算2个16*16块，一个block 算4个，共64个block
  // 我们的：每个block算2个，每个block 4个warp，每个warp reduce 1/4个k轴，最后再做同步，需要256个block
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
      __pipeline_memcpy_async((x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))), 
        (x + (((((((((int)blockIdx.x) * 65536) + 
        (ax0_ax1_fused_outer_outer_outer_outer * 16384)) + 
        (((int)threadIdx.z) * 4096)) + 
        ((((int)threadIdx.x) >> 4) * 2048)) + 
        (k_outer_outer * 128)) + 
        ((((int)threadIdx.x) & 15) * 8)))), 
        sizeof(uint4));
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      __pipeline_memcpy_async((weight_shared + (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) + (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))), 
        (weight + (((((((((int)blockIdx.y) * 131072) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16384)) + (((int)threadIdx.z) * 4096)) + ((((int)threadIdx.x) >> 4) * 2048)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))),
        sizeof(uint4));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[ax0_outer], ((half *)x_shared + (((ax0_outer * 2176) + (k_outer_inner * 16)))), 136);
      }
      (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], ((half *)weight_shared + (((((int)threadIdx.z) * 2176) + (k_outer_inner * 16)))), 136);
      for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
        (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[i_c_outer], x_shared_wmma_matrix_a[i_c_outer], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[i_c_outer]);
      }
    }
  }
  __syncthreads();
  
  for (int ax0_outer_inner = 0; ax0_outer_inner < 2; ++ax0_outer_inner) {
    // Guess through stride
    // ax0 * (16*72) + threadIdx.z * 16, thus, the x_shared is (32*64);
    (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + (((ax0_outer_inner * 1152) + (((int)threadIdx.z) * 16)))), T_dense_wmma_accumulator[ax0_outer_inner], 72, nvcuda::wmma::mem_row_major);
    // (void)nvcuda::wmma::store_matrix_sync(((half *)add + blockIdx.x * 32*512 + ax0_outer_inner * 16 * 512 + blockIdx.y * 64 + threadIdx.z * 16), T_dense_wmma_accumulator[ax0_outer_inner], 512, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  // Each warp computes 16*32 results, each thread needs to save 16 results,
  // each uint4 equals to 8 half, thus each thread save 2 times
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    uint4 _1;
    uint4 _2 = ((uint4*)(x_shared + (((((i_inner_j_inner_fused_outer_outer_outer_outer * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
    uint4 _3 = ((uint4*)(short_cut + (((((((((int)blockIdx.x) * 65536) + (i_inner_j_inner_fused_outer_outer_outer_outer * 32768)) + (((int)threadIdx.z) * 8192)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) &  7) * 8)))))[0];
    ((half2*)(&(_1.x)))->x = (((half2*)(&(_2.x)))->x+((half2*)(&(_3.x)))->x);
    ((half2*)(&(_1.x)))->y = (((half2*)(&(_2.x)))->y+((half2*)(&(_3.x)))->y);
    ((half2*)(&(_1.y)))->x = (((half2*)(&(_2.y)))->x+((half2*)(&(_3.y)))->x);
    ((half2*)(&(_1.y)))->y = (((half2*)(&(_2.y)))->y+((half2*)(&(_3.y)))->y);
    ((half2*)(&(_1.z)))->x = (((half2*)(&(_2.z)))->x+((half2*)(&(_3.z)))->x);
    ((half2*)(&(_1.z)))->y = (((half2*)(&(_2.z)))->y+((half2*)(&(_3.z)))->y);
    ((half2*)(&(_1.w)))->x = (((half2*)(&(_2.w)))->x+((half2*)(&(_3.w)))->x);
    ((half2*)(&(_1.w)))->y = (((half2*)(&(_2.w)))->y+((half2*)(&(_3.w)))->y);
    // row = blockIdx.x * 32 + ijxxx * 16 + threadIdx.z * 4 + threadIdx.x/8  
    //    (blockIdx.x * (32 * 512) + ijxxx * (16*512) + threadIdx.z * (4*512) + threadIdx.x / 8 * 512)
    // col = blockIdx.y * 64 + (threadIdx.x % 8)*8
    ((uint4*)(add + (((((((((int)blockIdx.x) * 16384) + 
    (i_inner_j_inner_fused_outer_outer_outer_outer * 8192)) + 
    (((int)threadIdx.z) * 2048)) + 
    ((((int)threadIdx.x) >> 3) * 512)) + 
    (((int)blockIdx.y) * 64)) + 
    ((((int)threadIdx.x) & 7) * 8)))))[0] = _1;
  }
}



// grid dim: <8, 8, 1>, block dim: <32, 1, 4>
// Double buffer version
__global__ void __launch_bounds__(128) fc2_16_16_2048_512_tvm_v3(half* __restrict__ x, half* __restrict__ weight, half* __restrict__ add, half* __restrict__ short_cut) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[2];
  extern half __shared__ shared_memory[];
  half* x_shared_0 = (half*)&shared_memory[0];
  half* x_shared_1 = (half*)&shared_memory[0+4352];
  half* weight_shared_0 = (half*)&shared_memory[0+4352+4352];
  half* weight_shared_1 = (half*)&shared_memory[0+4352+4352+8704];
  
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[1];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
  }
  // 在k轴做了tiling，更好的数据局部性
  // output: (16*16, 512) / (16, 16) -> (16, 32), 每个warp算2个16*16块，一个block 算8个 16x16，共64个block
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  pipe.producer_acquire();

  for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
    cuda::memcpy_async((x_shared_0 + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))), 
      (x + (((((((((int)blockIdx.x) * 65536) + (ax0_ax1_fused_outer_outer_outer_outer * 16384)) + (((int)threadIdx.z) * 4096)) + ((((int)threadIdx.x) >> 4) * 2048)) + 
      (0 * 128)) + ((((int)threadIdx.x) & 15) * 8)))), shape, pipe);
  }
  for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer_outer_outer1) {
    cuda::memcpy_async((weight_shared_0 + (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) + 
      (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))), 
      (weight + (((((((((int)blockIdx.y) * 131072) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16384)) + 
      (((int)threadIdx.z) * 4096)) + ((((int)threadIdx.x) >> 4) * 2048)) + 
      (0 * 128)) + ((((int)threadIdx.x) & 15) * 8)))), shape, pipe);
  }
  pipe.producer_commit();
  pipe.consumer_wait();

  half* load_x_shared=x_shared_1, *load_weight_shared=weight_shared_1;
  half* gemm_x_shared=x_shared_0, *gemm_weight_shared=weight_shared_0;
  for (int k_outer_outer = 1; k_outer_outer < 16; ++k_outer_outer) {
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
      cuda::memcpy_async(((load_x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8))))),  
      ((x + (((((((((int)blockIdx.x) * 65536) + 
        (ax0_ax1_fused_outer_outer_outer_outer * 16384)) + 
        (((int)threadIdx.z) * 4096)) + 
        ((((int)threadIdx.x) >> 4) * 2048)) + 
        (k_outer_outer * 128)) + 
        ((((int)threadIdx.x) & 15) * 8))))), shape, pipe);
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer_outer_outer1) { 
      // Why this will cause some load_x_shared value by zero when k_outer_outer is even?
      // if(ax0_ax1_fused_outer_outer_outer_outer1 & 0x1 == 0){
      //   cuda::memcpy_async((load_x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer1/2 * 1088) + (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))), 
      //   (x + (((((((((int)blockIdx.x) * 65536) + (ax0_ax1_fused_outer_outer_outer_outer1/2 * 16384)) + (((int)threadIdx.z) * 4096)) + ((((int)threadIdx.x) >> 4) * 2048)) + 
      //   (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))), 
      //   shape, pipe);
      // }
      (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[0], 
          ((half *)gemm_x_shared + (((0 * 2176) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16)))), 136);
      (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], 
        ((half *)gemm_weight_shared + (((((int)threadIdx.z) * 2176) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16)))), 136);
      (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[1], 
          ((half *)gemm_x_shared + (((1 * 2176) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16)))), 136);
      cuda::memcpy_async((load_weight_shared + (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) + 
        (((int)threadIdx.z) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))), 
        (weight + (((((((((int)blockIdx.y) * 131072) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16384)) + (((int)threadIdx.z) * 4096)) + ((((int)threadIdx.x) >> 4) * 2048)) + 
        (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))), shape, pipe);
      (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], 
          x_shared_wmma_matrix_a[0], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
      (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[1], 
          x_shared_wmma_matrix_a[1], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[1]);
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    
    // Swap pointers
    if(threadIdx.x==0 && threadIdx.z==0){
      half* tmp_x = load_x_shared; load_x_shared=gemm_x_shared; gemm_x_shared=tmp_x;
      half* tmp_weight=load_weight_shared; load_weight_shared=gemm_weight_shared; gemm_weight_shared=tmp_weight;
    }
    __syncthreads();
  }
  pipe.consumer_release();
  
  // Last computation for double buffer
  for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer_outer_outer1) {
    for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
      (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[ax0_outer], 
        ((half *)gemm_x_shared + (((ax0_outer * 2176) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16)))), 136);
    }
    (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], 
      ((half *)gemm_weight_shared + (((((int)threadIdx.z) * 2176) + (ax0_ax1_fused_outer_outer_outer_outer1 * 16)))), 136);
    for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
      (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[i_c_outer], 
        x_shared_wmma_matrix_a[i_c_outer], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[i_c_outer]);
    }
  }
  __syncthreads();
  
  for (int ax0_outer_inner = 0; ax0_outer_inner < 2; ++ax0_outer_inner) {
    // Guess through stride, ax0 * (16*72) + threadIdx.z * 16, thus, the x_shared is (32*64);
    (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared_0 + (((ax0_outer_inner * 1152) + (((int)threadIdx.z) * 16)))), T_dense_wmma_accumulator[ax0_outer_inner], 72, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  // Each warp computes 16*32 results, each thread needs to save 16 results,
  // each uint4 equals to 8 half, thus each thread save 2 times
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    uint4 _1;
    uint4 _2 = ((uint4*)(x_shared_0 + (((((i_inner_j_inner_fused_outer_outer_outer_outer * 1152) + (((int)threadIdx.z) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
    // 65536=128*512; 32768=64*512; 8192=16*512; 2048=4*512; 
    uint4 _3 =  ((uint4*)(short_cut + (((((((((int)blockIdx.x) * 65536/4) + (i_inner_j_inner_fused_outer_outer_outer_outer * 16384/4)) + (((int)threadIdx.z) * 4096/4)) + ((((int)threadIdx.x) >> 3) * 2048/4)) + (blockIdx.y * 64)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
    ((half2*)(&(_1.x)))->x = (((half2*)(&(_2.x)))->x+((half2*)(&(_3.x)))->x);
    ((half2*)(&(_1.x)))->y = (((half2*)(&(_2.x)))->y+((half2*)(&(_3.x)))->y);
    ((half2*)(&(_1.y)))->x = (((half2*)(&(_2.y)))->x+((half2*)(&(_3.y)))->x);
    ((half2*)(&(_1.y)))->y = (((half2*)(&(_2.y)))->y+((half2*)(&(_3.y)))->y);
    ((half2*)(&(_1.z)))->x = (((half2*)(&(_2.z)))->x+((half2*)(&(_3.z)))->x);
    ((half2*)(&(_1.z)))->y = (((half2*)(&(_2.z)))->y+((half2*)(&(_3.z)))->y);
    ((half2*)(&(_1.w)))->x = (((half2*)(&(_2.w)))->x+((half2*)(&(_3.w)))->x);
    ((half2*)(&(_1.w)))->y = (((half2*)(&(_2.w)))->y+((half2*)(&(_3.w)))->y);
    // row = blockIdx.x * 32 + ijxxx * 16 + threadIdx.z * 4 + threadIdx.x/8  
    //    (blockIdx.x * (32 * 512) + ijxxx * (16*512) + threadIdx.z * (4*512) + threadIdx.x / 8 * 512)
    // col = blockIdx.y * 64 + (threadIdx.x % 8)*8
    ((uint4*)(add + (((((((((int)blockIdx.x) * 16384) + 
      (i_inner_j_inner_fused_outer_outer_outer_outer * 8192)) + 
      (((int)threadIdx.z) * 2048)) + 
      ((((int)threadIdx.x) >> 3) * 512)) + 
      (((int)blockIdx.y) * 64)) + 
      ((((int)threadIdx.x) & 7) * 8)))))[0] = _1;
  }
}
