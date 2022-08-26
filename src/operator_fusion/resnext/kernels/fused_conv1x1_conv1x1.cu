#include <cuda_runtime.h>
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
extern "C" __global__ void __launch_bounds__(112) default_function_kernel0(float* __restrict__ fused_input, float* __restrict__ fused_weight, float* __restrict__ compute) {
  float compute_local[16];
  __shared__ float fused_input_shared[3808];
  __shared__ float fused_weight_shared[1088];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  for (int rk_outer_outer = 0; rk_outer_outer < 64; ++rk_outer_outer) {
    __syncthreads();
    fused_input_shared[(((int)threadIdx.x))] = fused_input[(((((((((int)blockIdx.x) / 224) * 250880) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((int)threadIdx.x) / 68) * 320)) + (rk_outer_outer * 4)) + (((int)threadIdx.x) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 112))] = fused_input[(((((((((int)blockIdx.x) / 224) * 250880) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 112) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 44) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 224))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 224) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 224) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 20) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 336))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 336) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 64) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 64) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 448))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 448) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 176) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 40) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 560))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 560) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 16) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 16) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 672))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 672) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 128) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 60) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 784))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 784) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 240) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 36) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 896))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 896) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 80) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 12) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 1008))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 1008) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 192) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 56) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 1120))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 1120) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 32) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 32) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 1232))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 1232) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 144) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 8) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 1344))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 1344) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 256) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 52) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 1456))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 1456) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 96) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 28) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 1568))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 1568) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 208) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 4) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 1680))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 1680) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 48) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 48) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 1792))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 1792) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 160) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 24) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 1904))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((int)threadIdx.x) / 68) * 320)) + (rk_outer_outer * 4)) + (((int)threadIdx.x) % 68)) + 125440))];
    fused_input_shared[((((int)threadIdx.x) + 2016))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 2016) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 112) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 44) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 2128))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 2128) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 224) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 20) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 2240))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 2240) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 64) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 64) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 2352))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 2352) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 176) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 40) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 2464))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 2464) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 16) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 16) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 2576))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 2576) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 128) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 60) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 2688))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 2688) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 240) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 36) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 2800))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 2800) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 80) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 12) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 2912))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 2912) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 192) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 56) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 3024))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 3024) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 32) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 32) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 3136))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 3136) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 144) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 8) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 3248))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 3248) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 256) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 52) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 3360))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 3360) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 96) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 28) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 3472))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 3472) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + ((((((int)threadIdx.x) + 208) % 272) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 4) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 3584))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 3584) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 48) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 48) % 68)))];
    fused_input_shared[((((int)threadIdx.x) + 3696))] = fused_input[((((((((((int)blockIdx.x) / 224) * 250880) + (((((int)threadIdx.x) + 3696) / 272) * 17920)) + (((((int)blockIdx.x) % 224) >> 4) * 1280)) + (((((int)threadIdx.x) + 160) / 68) * 320)) + (rk_outer_outer * 4)) + ((((int)threadIdx.x) + 24) % 68)))];
    fused_weight_shared[(((int)threadIdx.x))] = fused_weight[(((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)))];
    fused_weight_shared[((((int)threadIdx.x) + 112))] = fused_weight[((((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 1792))];
    fused_weight_shared[((((int)threadIdx.x) + 224))] = fused_weight[((((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 3584))];
    fused_weight_shared[((((int)threadIdx.x) + 336))] = fused_weight[((((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 5376))];
    fused_weight_shared[((((int)threadIdx.x) + 448))] = fused_weight[((((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 7168))];
    fused_weight_shared[((((int)threadIdx.x) + 560))] = fused_weight[((((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 8960))];
    fused_weight_shared[((((int)threadIdx.x) + 672))] = fused_weight[((((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 10752))];
    fused_weight_shared[((((int)threadIdx.x) + 784))] = fused_weight[((((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 12544))];
    fused_weight_shared[((((int)threadIdx.x) + 896))] = fused_weight[((((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 14336))];
    if (((int)threadIdx.x) < 80) {
      fused_weight_shared[((((int)threadIdx.x) + 1008))] = fused_weight[((((((rk_outer_outer * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 15) * 16)) + (((int)threadIdx.x) & 15)) + 16128))];
    }
    __syncthreads();
    compute_local[(0)] = (compute_local[(0)] + ((rk_outer_outer < 16) ? (fused_input_shared[(((((int)threadIdx.x) >> 2) * 136))] * fused_weight_shared[(((((int)threadIdx.x) & 3) * 2))]) : 0.000000e+00f));
    compute_local[(8)] = (compute_local[(8)] + ((rk_outer_outer < 16) ? (fused_input_shared[(((((int)threadIdx.x) >> 2) * 136))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 8))]) : 0.000000e+00f));
    compute_local[(1)] = (compute_local[(1)] + ((rk_outer_outer < 16) ? (fused_input_shared[(((((int)threadIdx.x) >> 2) * 136))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1))]) : 0.000000e+00f));
    compute_local[(9)] = (compute_local[(9)] + ((rk_outer_outer < 16) ? (fused_input_shared[(((((int)threadIdx.x) >> 2) * 136))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 9))]) : 0.000000e+00f));
    compute_local[(2)] = (compute_local[(2)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 68))] * fused_weight_shared[(((((int)threadIdx.x) & 3) * 2))]) : 0.000000e+00f));
    compute_local[(10)] = (compute_local[(10)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 68))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 8))]) : 0.000000e+00f));
    compute_local[(3)] = (compute_local[(3)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 68))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1))]) : 0.000000e+00f));
    compute_local[(11)] = (compute_local[(11)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 68))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 9))]) : 0.000000e+00f));
    compute_local[(0)] = (compute_local[(0)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 1))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 16))]) : 0.000000e+00f));
    compute_local[(8)] = (compute_local[(8)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 1))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 24))]) : 0.000000e+00f));
    compute_local[(1)] = (compute_local[(1)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 1))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 17))]) : 0.000000e+00f));
    compute_local[(9)] = (compute_local[(9)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 1))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 25))]) : 0.000000e+00f));
    compute_local[(2)] = (compute_local[(2)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 69))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 16))]) : 0.000000e+00f));
    compute_local[(10)] = (compute_local[(10)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 69))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 24))]) : 0.000000e+00f));
    compute_local[(3)] = (compute_local[(3)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 69))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 17))]) : 0.000000e+00f));
    compute_local[(11)] = (compute_local[(11)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 69))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 25))]) : 0.000000e+00f));
    compute_local[(0)] = (compute_local[(0)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 2))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 32))]) : 0.000000e+00f));
    compute_local[(8)] = (compute_local[(8)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 2))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 40))]) : 0.000000e+00f));
    compute_local[(1)] = (compute_local[(1)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 2))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 33))]) : 0.000000e+00f));
    compute_local[(9)] = (compute_local[(9)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 2))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 41))]) : 0.000000e+00f));
    compute_local[(2)] = (compute_local[(2)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 70))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 32))]) : 0.000000e+00f));
    compute_local[(10)] = (compute_local[(10)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 70))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 40))]) : 0.000000e+00f));
    compute_local[(3)] = (compute_local[(3)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 70))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 33))]) : 0.000000e+00f));
    compute_local[(11)] = (compute_local[(11)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 70))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 41))]) : 0.000000e+00f));
    compute_local[(0)] = (compute_local[(0)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 3))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 48))]) : 0.000000e+00f));
    compute_local[(8)] = (compute_local[(8)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 3))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 56))]) : 0.000000e+00f));
    compute_local[(1)] = (compute_local[(1)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 3))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 49))]) : 0.000000e+00f));
    compute_local[(9)] = (compute_local[(9)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 3))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 57))]) : 0.000000e+00f));
    compute_local[(2)] = (compute_local[(2)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 71))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 48))]) : 0.000000e+00f));
    compute_local[(10)] = (compute_local[(10)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 71))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 56))]) : 0.000000e+00f));
    compute_local[(3)] = (compute_local[(3)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 71))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 49))]) : 0.000000e+00f));
    compute_local[(11)] = (compute_local[(11)] + ((rk_outer_outer < 16) ? (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 71))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 57))]) : 0.000000e+00f));
    compute_local[(4)] = (compute_local[(4)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 64))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1024))]));
    compute_local[(12)] = (compute_local[(12)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 64))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1032))]));
    compute_local[(5)] = (compute_local[(5)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 64))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1025))]));
    compute_local[(13)] = (compute_local[(13)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 64))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1033))]));
    compute_local[(6)] = (compute_local[(6)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 132))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1024))]));
    compute_local[(14)] = (compute_local[(14)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 132))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1032))]));
    compute_local[(7)] = (compute_local[(7)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 132))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1025))]));
    compute_local[(15)] = (compute_local[(15)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 132))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1033))]));
    compute_local[(4)] = (compute_local[(4)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 65))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1040))]));
    compute_local[(12)] = (compute_local[(12)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 65))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1048))]));
    compute_local[(5)] = (compute_local[(5)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 65))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1041))]));
    compute_local[(13)] = (compute_local[(13)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 65))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1049))]));
    compute_local[(6)] = (compute_local[(6)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 133))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1040))]));
    compute_local[(14)] = (compute_local[(14)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 133))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1048))]));
    compute_local[(7)] = (compute_local[(7)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 133))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1041))]));
    compute_local[(15)] = (compute_local[(15)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 133))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1049))]));
    compute_local[(4)] = (compute_local[(4)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 66))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1056))]));
    compute_local[(12)] = (compute_local[(12)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 66))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1064))]));
    compute_local[(5)] = (compute_local[(5)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 66))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1057))]));
    compute_local[(13)] = (compute_local[(13)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 66))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1065))]));
    compute_local[(6)] = (compute_local[(6)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 134))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1056))]));
    compute_local[(14)] = (compute_local[(14)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 134))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1064))]));
    compute_local[(7)] = (compute_local[(7)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 134))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1057))]));
    compute_local[(15)] = (compute_local[(15)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 134))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1065))]));
    compute_local[(4)] = (compute_local[(4)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 67))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1072))]));
    compute_local[(12)] = (compute_local[(12)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 67))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1080))]));
    compute_local[(5)] = (compute_local[(5)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 67))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1073))]));
    compute_local[(13)] = (compute_local[(13)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 67))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1081))]));
    compute_local[(6)] = (compute_local[(6)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 135))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1072))]));
    compute_local[(14)] = (compute_local[(14)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 135))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1080))]));
    compute_local[(7)] = (compute_local[(7)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 135))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1073))]));
    compute_local[(15)] = (compute_local[(15)] + (fused_input_shared[((((((int)threadIdx.x) >> 2) * 136) + 135))] * fused_weight_shared[((((((int)threadIdx.x) & 3) * 2) + 1081))]));
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    for (int w_inner = 0; w_inner < 2; ++w_inner) {
      for (int o_inner = 0; o_inner < 2; ++o_inner) {
        compute[((((((((((i_inner * 802816) + ((((int)blockIdx.x) / 224) * 200704)) + ((((int)threadIdx.x) >> 3) * 14336)) + (((((int)blockIdx.x) % 224) >> 4) * 1024)) + (((((int)threadIdx.x) & 7) >> 2) * 512)) + (w_inner * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 2)) + o_inner))] = compute_local[((((i_inner * 4) + (w_inner * 2)) + o_inner))];
        compute[(((((((((((i_inner * 802816) + ((((int)blockIdx.x) / 224) * 200704)) + ((((int)threadIdx.x) >> 3) * 14336)) + (((((int)blockIdx.x) % 224) >> 4) * 1024)) + (((((int)threadIdx.x) & 7) >> 2) * 512)) + (w_inner * 256)) + ((((int)blockIdx.x) & 15) * 16)) + ((((int)threadIdx.x) & 3) * 2)) + o_inner) + 8))] = compute_local[(((((i_inner * 4) + (w_inner * 2)) + o_inner) + 8))];
      }
    }
  }
}
