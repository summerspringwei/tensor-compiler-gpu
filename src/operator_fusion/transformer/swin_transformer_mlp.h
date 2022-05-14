#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

// <8, 8, 1>,<32, 1, 4>
extern "C" __global__ void __launch_bounds__(128) fc2_16_16_2048_512(half* __restrict__ x, half* __restrict__ weight, half* __restrict__ add, half* __restrict__ short_cut) {
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
}


// output: (16*16, 512) / (16, 16) -> (16, 32), 每个warp算2个16*16块，一个block 算4个，共64个block
// 我们的：每个block算2个，每个block 4个warp，每个warp reduce 1/4个k轴，最后再做同步，需要256个block
// <16, 16, 1>, <32, 4, 1>
extern "C" __global__ void __launch_bounds__(128) fc2_16_16_2048_512_v2(half* __restrict__ x, half* __restrict__ weight, half* __restrict__ add, half* __restrict__ short_cut) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[2];
  //for one warp, K is 512, we tile 128 for one
  const int M=16*16, N=512, K=2048;
  const int warp_size=32;
  
  const int block_dim_y = 4;
  const int num_k_outer_tile = 8;
  const int tile_size_m = 16;
  const int tile_size_n = 32;
  const int tile_size_k = K/num_k_outer_tile;
  __shared__ half x_shared[tile_size_m*tile_size_k];
  __shared__ half weight_shared[tile_size_n*tile_size_k];
  // __shared__ half output_shared[tile_size_m*tile_size_n*block_dim_y];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> weight_shared_wmma_matrix_b[2];
  #pragma unroll
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
  }
  const int vec_size = 8; // sizeof(uint4) / sizeof(half)
  const int warpId = threadIdx.y;
  #pragma unroll
  for(int ko=0; ko<num_k_outer_tile; ++ko){
    __syncthreads();
    // Each block load tile_m * tile_size_k, consider vec_size=8, we only need load tile_m*(tile_size_k/vec_size)/warp_size/block_dim_y
    const int vec_col_size = (tile_size_k/vec_size); // 256/8=32
    const int x_num_iter = tile_size_m*vec_col_size/warp_size/block_dim_y; // 16*(256/8)/32/4=8
    const int each_iter_process_rows = warp_size*block_dim_y / vec_col_size; // 32*4/32
    // Load x
    #pragma unroll
    for(int i=0; i<x_num_iter; ++i){
      const int x_shared_row = i * each_iter_process_rows + (threadIdx.y * warp_size + threadIdx.x) / vec_col_size; // 15
      const int x_shared_col = (threadIdx.y * warp_size + threadIdx.x) % vec_col_size; // 0
      const int x_gm_row = blockIdx.x * tile_size_m + x_shared_row; // 15*16+15
      const int x_gm_col = ko * vec_col_size + x_shared_col; // 1 * 256 + 0
      ((uint4*)x_shared + x_shared_row * vec_col_size + x_shared_col)[0] = ((uint4*)x + x_gm_row * (K/vec_size) + x_gm_col)[0];
      // half* x_ptr = reinterpret_cast<half*>((uint4*)x_shared + x_shared_row * vec_col_size + x_shared_col);
      // if(x_ptr[0]!=__float2half(1)){
      //   x_ptr[0]=__float2half(1);
      // }
    }
    // As weight has the same col as x
    const int weight_num_iter = tile_size_n*vec_col_size/warp_size/block_dim_y;
    #pragma unroll
    for(int i=0; i<weight_num_iter; ++i){
      int weight_shared_row = i * each_iter_process_rows + (threadIdx.y * warp_size + threadIdx.x) / vec_col_size;
      int weight_shared_col = (threadIdx.y * warp_size + threadIdx.x) % vec_col_size;
      int weight_gm_row = blockIdx.y * tile_size_m + weight_shared_row;
      int weight_gm_col = ko * vec_col_size + weight_shared_col;
      ((uint4*)weight_shared + weight_shared_row * vec_col_size + weight_shared_col)[0] = ((uint4*)weight + weight_gm_row * (K/vec_size) + weight_gm_col)[0];
      // half* weight_ptr = reinterpret_cast<half*>((uint4*)weight_shared + weight_shared_row * vec_col_size + weight_shared_col);
      // if(weight_ptr[0]!=__float2half(1)){
      //   weight_ptr[0]=__float2half(1);
      // }
    }
    __syncthreads();
    #pragma unroll
    for(int i=0;i<(K/num_k_outer_tile/16/block_dim_y); ++i){
      const int row = 0;
      const int col = i * block_dim_y * 16 + warpId * 16;
      (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[0], ((half *)x_shared + col), tile_size_k);
      (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], ((half *)weight_shared + 0 * 16 * tile_size_k + col), tile_size_k);
      (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
      (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[1], ((half *)weight_shared + 1 * 16 * tile_size_k + col), tile_size_k);
      (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[1], x_shared_wmma_matrix_a[0], weight_shared_wmma_matrix_b[1], T_dense_wmma_accumulator[1]);
    }
    __syncthreads();
  }
  // Reduce all the acc between warps and save to global memory 
  (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + (warpId * tile_size_n + 0 * 16)), T_dense_wmma_accumulator[0], tile_size_n*block_dim_y, nvcuda::wmma::mem_row_major);
  (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + (warpId * tile_size_n + 1 * 16)), T_dense_wmma_accumulator[1], tile_size_n*block_dim_y, nvcuda::wmma::mem_row_major);
  __syncthreads();

  #pragma unroll
  for(int i=0; i<16/block_dim_y; ++i){
    x_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x] += x_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+1*tile_size_n+threadIdx.x];
    x_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x] += x_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+2*tile_size_n+threadIdx.x];
    x_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x] += x_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+3*tile_size_n+threadIdx.x];
    // __syncthreads();
    add[(blockIdx.x*tile_size_m+(i*block_dim_y + warpId)) * N + (blockIdx.y * tile_size_n + threadIdx.x)] = x_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x];
  }
  // (void)nvcuda::wmma::store_matrix_sync(((half *)output_shared + (warpId * tile_size_n + 0 * 16)), T_dense_wmma_accumulator[0], tile_size_n*block_dim_y, nvcuda::wmma::mem_row_major);
  // (void)nvcuda::wmma::store_matrix_sync(((half *)output_shared + (warpId * tile_size_n + 1 * 16)), T_dense_wmma_accumulator[1], tile_size_n*block_dim_y, nvcuda::wmma::mem_row_major);
  // __syncthreads();
  
  // for(int i=0; i<16/block_dim_y; ++i){
  //   output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x] += output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+1*tile_size_n+threadIdx.x];
  //   output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x] += output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+2*tile_size_n+threadIdx.x];
  //   output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x] += output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+3*tile_size_n+threadIdx.x];
  //   __syncthreads();
  //   if(output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x] < __float2half(2000.0))
  //     add[(blockIdx.x*tile_size_m+(i*block_dim_y + warpId)) * N + (blockIdx.y * tile_size_n + threadIdx.x)] = output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x];
  //   if(output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x] != (half)2048){
  //     printf("(%d %d, %d),(%d, %d, %d) %f\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, __half2float(output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x]));
  //   }
  //   add[(blockIdx.x*tile_size_m+(i*block_dim_y + warpId)) * N + (blockIdx.y * tile_size_n + threadIdx.x)] = output_shared[(i*block_dim_y + warpId)*tile_size_n*block_dim_y+threadIdx.x];
  // }
}