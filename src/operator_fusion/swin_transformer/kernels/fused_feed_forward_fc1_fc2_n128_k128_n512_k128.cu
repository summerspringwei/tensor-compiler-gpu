#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#ifndef TVM_HELPER_FUNC
#define TVM_HELPER_FUNC
// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y)
{
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// fix undefined fp16 match function

static inline __device__ __host__ half hpow(half x, half y)
{
  float tmp_x = __half2float(x);
  float tmp_y = __half2float(y);
  float result = powf(tmp_x, tmp_y);
  return __float2half(result);
}

static inline __device__ __host__ half htanh(half x)
{
  float tmp_x = __half2float(x);
  float result = tanhf(tmp_x);
  return __float2half(result);
}
#endif

#include <mma.h>
// dim3(64, 1, 1), dim3(32, 2, 4)
extern "C" __global__ void __launch_bounds__(256) fused_attn_fc1(half *__restrict__ x, half *__restrict__ weight,
                                                                           half *__restrict__ x_second, half *__restrict__ x_mean, half *__restrict__ x_variance_sum, half *__restrict__ weight_second, half *__restrict__ compute)
{
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(uint2)>(sizeof(uint2));
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> T_dense_wmma_accumulator[4];
  extern __shared__ float all_shared_mem[];
  half *reshape_permute_shared = (half *)all_shared_mem;
  half *weight_shared = ((half *)all_shared_mem) + 2560;
  half *weight_shared_second = ((half *)weight_shared) + 8704;
  half *compute_shared = reshape_permute_shared;
  // __shared__ half weight_shared[8704];
  // __shared__ half compute_shared[2304]; // A
  // __shared__ half weight_shared_second[9216];  // B __shared__ half compute_shared[2304]; // A
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> reshape_permute_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[1];
  for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init)
  {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer)
  {
    if (blockIdx.x < 64)
    {
      __syncthreads();
      for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 8; ++ax0_ax1_fused_outer_outer_outer_outer)
      {
        reshape_permute_shared[(((((ax0_ax1_fused_outer_outer_outer_outer * 320) + (((int)threadIdx.z) * 80)) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)))] = x[(((((((((int)blockIdx.x) * 8192) + (ax0_ax1_fused_outer_outer_outer_outer * 1024)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + k_outer_outer))];
      }
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 16; ++ax0_ax1_fused_outer_outer_outer_outer1)
      {
        weight_shared[(((((ax0_ax1_fused_outer_outer_outer_outer1 * 320) + (((int)threadIdx.z) * 80)) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)))] = weight[((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (k_outer_outer * 32)) + ((int)threadIdx.x)))];
      }
      __syncthreads();
    }
    if (k_outer_outer == 3)
    {
      pipe.producer_acquire();
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer_outer_outer1)
      {
        ((uint2 *)(weight_shared + ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1152) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = [0];
      }
      cuda::memcpy_async(((uint2 *)(weight_shared_second + ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1152) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))))),
                         ((uint2 *)(weight_second + ((((((((((int)blockIdx.y) * 16384) + (ax0_ax1_fused_outer_outer_outer_outer1 * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4))))),
                         shape, pipe);
      pipe.producer_commit();
    }
    //
    if (blockIdx.x < 64)
    {
      for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner)
      {
        for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer)
        {
          (void)nvcuda::wmma::load_matrix_sync(reshape_permute_shared_wmma_matrix_a[ax0_outer], ((half *)reshape_permute_shared + ((((((int)threadIdx.y) * 1280) + (ax0_outer * 320)) + (k_outer_inner * 16)))), 40);
        }
        (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], ((half *)weight_shared + (((((int)threadIdx.z) * 1280) + (k_outer_inner * 16)))), 40);
        for (int i_c_outer = 0; i_c_outer < 4; ++i_c_outer)
        {
          (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[i_c_outer], reshape_permute_shared_wmma_matrix_a[i_c_outer], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[i_c_outer]);
        }
      }
    }
  }
  if (blockIdx.x < 64)
  {
    __syncthreads();
    for (int ax0_outer_inner = 0; ax0_outer_inner < 4; ++ax0_outer_inner)
    {
      (void)nvcuda::wmma::store_matrix_sync(((half *)weight_shared + ((((((int)threadIdx.y) * 4352) + (ax0_outer_inner * 1088)) + (((int)threadIdx.z) * 32)))), T_dense_wmma_accumulator[ax0_outer_inner], 136, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 32; ++i_inner_j_inner_fused_outer_outer_outer_outer)
    {
      x_second[((((((((int)blockIdx.x) * 8192) + (i_inner_j_inner_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = weight_shared[((((i_inner_j_inner_fused_outer_outer_outer_outer * 272) + (((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) >> 7) * 136)) + ((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) & 127)))];
    }
  }

  pipe.consumer_wait();
  __syncthreads();
  pipe.consumer_release();
  grid.sync();
  /*************************************************/
  if (blockIdx.x < 128)
  {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[2];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> compute_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[2];
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init)
    {
      (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init], 0.000000e+00f);
    }
    for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer)
    {
      __syncthreads();
      for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 2; ++ax0_ax1_fused_outer_outer_outer_outer)
      {
        uint2 _1;
        float4 _2;
        float4 _3;
        uint2 _4;
        uint2 _5 = ((uint2 *)(x_second + ((((((((((int)blockIdx.x) * 4096) + (ax0_ax1_fused_outer_outer_outer_outer * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
        int4 _6 = make_int4((((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_outer_outer_outer_outer * 16)) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)), (((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_outer_outer_outer_outer * 16)) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)), (((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_outer_outer_outer_outer * 16)) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)), (((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_outer_outer_outer_outer * 16)) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)));
        uint2 _7 = make_uint2(__pack_half2(x_mean[_6.x], x_mean[_6.y]), __pack_half2(x_mean[_6.z], x_mean[_6.w]));
        ((half2 *)(&(_4.x)))->x = (((half2 *)(&(_5.x)))->x - ((half2 *)(&(_7.x)))->x);
        ((half2 *)(&(_4.x)))->y = (((half2 *)(&(_5.x)))->y - ((half2 *)(&(_7.x)))->y);
        ((half2 *)(&(_4.y)))->x = (((half2 *)(&(_5.y)))->x - ((half2 *)(&(_7.y)))->x);
        ((half2 *)(&(_4.y)))->y = (((half2 *)(&(_5.y)))->y - ((half2 *)(&(_7.y)))->y);
        _3.x = (float)(((half2 *)(&(_4.x)))->x);
        _3.y = (float)(((half2 *)(&(_4.x)))->y);
        _3.z = (float)(((half2 *)(&(_4.y)))->x);
        _3.w = (float)(((half2 *)(&(_4.y)))->y);
        float4 _8;
        float4 _9;
        float4 _10;
        int4 _11 = make_int4((((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_outer_outer_outer_outer * 16)) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)), (((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_outer_outer_outer_outer * 16)) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)), (((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_outer_outer_outer_outer * 16)) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)), (((((((int)blockIdx.x) * 32) + (ax0_ax1_fused_outer_outer_outer_outer * 16)) + (((int)threadIdx.z) * 4)) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)));
        uint2 _12 = make_uint2(__pack_half2(x_variance_sum[_11.x], x_variance_sum[_11.y]), __pack_half2(x_variance_sum[_11.z], x_variance_sum[_11.w]));
        _10.x = (float)(((half2 *)(&(_12.x)))->x);
        _10.y = (float)(((half2 *)(&(_12.x)))->y);
        _10.z = (float)(((half2 *)(&(_12.y)))->x);
        _10.w = (float)(((half2 *)(&(_12.y)))->y);
        float4 _13 = make_float4(7.812500e-03f, 7.812500e-03f, 7.812500e-03f, 7.812500e-03f);
        _9.x = (_10.x * _13.x);
        _9.y = (_10.y * _13.y);
        _9.z = (_10.z * _13.z);
        _9.w = (_10.w * _13.w);
        float4 _14 = make_float4(1.000000e+05f, 1.000000e+05f, 1.000000e+05f, 1.000000e+05f);
        _8.x = (_9.x + _14.x);
        _8.y = (_9.y + _14.y);
        _8.z = (_9.z + _14.z);
        _8.w = (_9.w + _14.w);
        _2.x = (_3.x / _8.x);
        _2.y = (_3.y / _8.y);
        _2.z = (_3.z / _8.z);
        _2.w = (_3.w / _8.w);
        ((half2 *)(&(_1.x)))->x = (half)(_2.x);
        ((half2 *)(&(_1.x)))->y = (half)(_2.y);
        ((half2 *)(&(_1.y)))->x = (half)(_2.z);
        ((half2 *)(&(_1.y)))->y = (half)(_2.w);
        ((uint2 *)(compute_shared + ((((((ax0_ax1_fused_outer_outer_outer_outer * 1152) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = _1;
      }
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer_outer_outer1)
      {
        ((uint2 *)(weight_shared_second + ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1152) + (((int)threadIdx.z) * 288)) + (((int)threadIdx.y) * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((uint2 *)(weight_second + ((((((((((int)blockIdx.y) * 16384) + (ax0_ax1_fused_outer_outer_outer_outer1 * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
      }
      __syncthreads();
      //

      for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner)
      {
        (void)nvcuda::wmma::load_matrix_sync(compute_shared_wmma_matrix_a[0], ((half *)compute_shared + (((((int)threadIdx.y) * 1152) + (k_outer_inner * 16)))), 72);
        for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer)
        {
          (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[ax0_outer], ((half *)weight_shared_second + ((((((int)threadIdx.z) * 2304) + (ax0_outer * 1152)) + (k_outer_inner * 16)))), 72);
        }
        for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer)
        {
          (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[j_c_outer], compute_shared_wmma_matrix_a[0], weight_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[j_c_outer]);
        }
      }
    }
    __syncthreads();
    for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner)
    {
      (void)nvcuda::wmma::store_matrix_sync(((half *)weight_shared_second + ((((((int)threadIdx.y) * 2176) + (((int)threadIdx.z) * 32)) + (ax1_outer_inner * 16)))), T_dense_wmma_accumulator[ax1_outer_inner], 136, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    for (int i0_inner_i1_inner_fused_outer_outer_outer_outer = 0; i0_inner_i1_inner_fused_outer_outer_outer_outer < 4; ++i0_inner_i1_inner_fused_outer_outer_outer_outer)
    {
      uint2 _15;
      float4 _16;
      float4 _17;
      float4 _18 = make_float4(5.000000e-01f, 5.000000e-01f, 5.000000e-01f, 5.000000e-01f);
      float4 _19;
      uint2 _20 = ((uint2 *)(weight_shared_second + (((((i0_inner_i1_inner_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx.z) * 272)) + (((int)threadIdx.y) * 136)) + (((int)threadIdx.x) * 4)))))[0];
      _19.x = (float)(((half2 *)(&(_20.x)))->x);
      _19.y = (float)(((half2 *)(&(_20.x)))->y);
      _19.z = (float)(((half2 *)(&(_20.y)))->x);
      _19.w = (float)(((half2 *)(&(_20.y)))->y);
      _17.x = (_18.x * _19.x);
      _17.y = (_18.y * _19.y);
      _17.z = (_18.z * _19.z);
      _17.w = (_18.w * _19.w);
      float4 _21;
      float4 _22 = make_float4(1.000000e+00f, 1.000000e+00f, 1.000000e+00f, 1.000000e+00f);
      float4 _23;
      float4 _24;
      float4 _25 = make_float4(7.978846e-01f, 7.978846e-01f, 7.978846e-01f, 7.978846e-01f);
      float4 _26;
      float4 _27;
      uint2 _28 = ((uint2 *)(weight_shared_second + (((((i0_inner_i1_inner_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx.z) * 272)) + (((int)threadIdx.y) * 136)) + (((int)threadIdx.x) * 4)))))[0];
      _27.x = (float)(((half2 *)(&(_28.x)))->x);
      _27.y = (float)(((half2 *)(&(_28.x)))->y);
      _27.z = (float)(((half2 *)(&(_28.y)))->x);
      _27.w = (float)(((half2 *)(&(_28.y)))->y);
      float4 _29;
      float4 _30 = make_float4(4.471500e-02f, 4.471500e-02f, 4.471500e-02f, 4.471500e-02f);
      float4 _31;
      uint2 _32;
      uint2 _33 = make_uint2(__pack_half2(__float2half_rn(3.000000e+00f), __float2half_rn(3.000000e+00f)), __pack_half2(__float2half_rn(3.000000e+00f), __float2half_rn(3.000000e+00f)));
      ((half2 *)(&(_32.x)))->x = hpow(((half2 *)(&(_28.x)))->x, ((half2 *)(&(_33.x)))->x);
      ((half2 *)(&(_32.x)))->y = hpow(((half2 *)(&(_28.x)))->y, ((half2 *)(&(_33.x)))->y);
      ((half2 *)(&(_32.y)))->x = hpow(((half2 *)(&(_28.y)))->x, ((half2 *)(&(_33.y)))->x);
      ((half2 *)(&(_32.y)))->y = hpow(((half2 *)(&(_28.y)))->y, ((half2 *)(&(_33.y)))->y);
      _31.x = (float)(((half2 *)(&(_32.x)))->x);
      _31.y = (float)(((half2 *)(&(_32.x)))->y);
      _31.z = (float)(((half2 *)(&(_32.y)))->x);
      _31.w = (float)(((half2 *)(&(_32.y)))->y);
      _29.x = (_30.x * _31.x);
      _29.y = (_30.y * _31.y);
      _29.z = (_30.z * _31.z);
      _29.w = (_30.w * _31.w);
      _26.x = (_27.x + _29.x);
      _26.y = (_27.y + _29.y);
      _26.z = (_27.z + _29.z);
      _26.w = (_27.w + _29.w);
      _24.x = (_25.x * _26.x);
      _24.y = (_25.y * _26.y);
      _24.z = (_25.z * _26.z);
      _24.w = (_25.w * _26.w);
      _23.x = tanhf(_24.x);
      _23.y = tanhf(_24.y);
      _23.z = tanhf(_24.z);
      _23.w = tanhf(_24.w);
      _21.x = (_22.x + _23.x);
      _21.y = (_22.y + _23.y);
      _21.z = (_22.z + _23.z);
      _21.w = (_22.w + _23.w);
      _16.x = (_17.x * _21.x);
      _16.y = (_17.y * _21.y);
      _16.z = (_17.z * _21.z);
      _16.w = (_17.w * _21.w);
      ((half2 *)(&(_15.x)))->x = (half)(_16.x);
      ((half2 *)(&(_15.x)))->y = (half)(_16.y);
      ((half2 *)(&(_15.y)))->x = (half)(_16.z);
      ((half2 *)(&(_15.y)))->y = (half)(_16.w);
      ((uint2 *)(compute + (((((((((int)blockIdx.x) * 16384) + (i0_inner_i1_inner_fused_outer_outer_outer_outer * 4096)) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.x) * 4)))))[0] = _15;
    }
  }
}
