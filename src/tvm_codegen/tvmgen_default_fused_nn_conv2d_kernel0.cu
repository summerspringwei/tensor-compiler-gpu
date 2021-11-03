
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

extern "C" __global__ void __launch_bounds__(16) tvmgen_default_fused_nn_conv2d_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ Conv2dOutput) {
  float Conv2dOutput_local[2];
  __shared__ float PaddedInput_shared[24];
  __shared__ float placeholder_shared[256];
  float PaddedInput_shared_local[1];
  float placeholder_shared_local[2];
  for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
    Conv2dOutput_local[(ff_c_init)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 10; ++rc_outer) {
    __syncthreads();
    PaddedInput_shared[(((((int)threadIdx.y) * 4) + ((int)threadIdx.x)))] = placeholder[(((((((int)blockIdx.z) * 160) + (rc_outer * 16)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)))];
    for (int ax2_ax3_fused_outer_outer = 0; ax2_ax3_fused_outer_outer < 8; ++ax2_ax3_fused_outer_outer) {
      placeholder_shared[((((ax2_ax3_fused_outer_outer * 32) + ((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) >> 3) * 16)) + (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) & 7)))] = placeholder1[((((((rc_outer * 1024) + (ax2_ax3_fused_outer_outer * 128)) + ((((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) >> 3) * 64)) + (((int)blockIdx.x) * 8)) + (((((int)threadIdx.y) * 4) + ((int)threadIdx.x)) & 7)))];
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
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    if (((int)threadIdx.y) < 1) {
      Conv2dOutput[((((((((int)threadIdx.y) * 341056) + (((int)blockIdx.z) * 64)) + (((int)blockIdx.x) * 8)) + (((int)threadIdx.x) * 2)) + ff_inner))] = Conv2dOutput_local[(ff_inner)];
    }
  }
}

