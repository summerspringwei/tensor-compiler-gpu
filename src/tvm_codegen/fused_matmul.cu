
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
extern "C" __global__ void __launch_bounds__(256) fused_matmul_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[64];
  __shared__ float A_shared[4096];
  __shared__ float B_shared[4096];
  float A_shared_local[8];
  float B_shared_local[8];
  for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
    for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
      C_local[(((i_c_init * 4) + j_c_init))] = 0.000000e+00f;
      C_local[((((i_c_init * 4) + j_c_init) + 32))] = 0.000000e+00f;
    }
  }
  for (int rik_outer = 0; rik_outer < 5; ++rik_outer) {
    __syncthreads();
    for (int ax0_outer = 0; ax0_outer < 8; ++ax0_outer) {
      for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
        if ((((((int)blockIdx.y) * 128) + (ax0_outer * 16)) + ((int)threadIdx.y)) < 5329) {
          A_shared[(((((ax0_outer * 512) + (((int)threadIdx.y) * 32)) + (ax1_outer * 16)) + ((int)threadIdx.x)))] = A[(((((((((int)blockIdx.y) * 20480) + (ax0_outer * 2560)) + (((int)threadIdx.y) * 160)) + (rik_outer * 32)) + (ax1_outer * 16)) + ((int)threadIdx.x)))];
        }
      }
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
      for (int ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
        B_shared[(((((ax1_outer1 * 1024) + (((int)threadIdx.y) * 64)) + (ax2_outer * 16)) + ((int)threadIdx.x)))] = B[((((((rik_outer * 2048) + (ax1_outer1 * 1024)) + (((int)threadIdx.y) * 64)) + (ax2_outer * 16)) + ((int)threadIdx.x)))];
        B_shared[((((((ax1_outer1 * 1024) + (((int)threadIdx.y) * 64)) + (ax2_outer * 16)) + ((int)threadIdx.x)) + 2048))] = B[(((((((rik_outer * 2048) + (ax1_outer1 * 1024)) + (((int)threadIdx.y) * 64)) + (ax2_outer * 16)) + ((int)threadIdx.x)) + 10240))];
      }
    }
    __syncthreads();
    for (int rik_inner = 0; rik_inner < 32; ++rik_inner) {
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + ax0) < 5329) {
          A_shared_local[(ax0)] = A_shared[((((((int)threadIdx.y) * 256) + (ax0 * 32)) + rik_inner))];
        }
      }
      for (int ax01 = 0; ax01 < 2; ++ax01) {
        for (int ax2 = 0; ax2 < 4; ++ax2) {
          B_shared_local[(((ax01 * 4) + ax2))] = B_shared[(((((ax01 * 2048) + (rik_inner * 64)) + (((int)threadIdx.x) * 4)) + ax2))];
        }
      }
      for (int j_c = 0; j_c < 4; ++j_c) {
        for (int i_c = 0; i_c < 8; ++i_c) {
          if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + i_c) < 5329) {
            C_local[(((i_c * 4) + j_c))] = (C_local[(((i_c * 4) + j_c))] + (A_shared_local[(i_c)] * B_shared_local[(j_c)]));
          }
          if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + i_c) < 5329) {
            C_local[((((i_c * 4) + j_c) + 32))] = (C_local[((((i_c * 4) + j_c) + 32))] + (A_shared_local[(i_c)] * B_shared_local[((j_c + 4))]));
          }
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    for (int j_inner_inner = 0; j_inner_inner < 4; ++j_inner_inner) {
      for (int offset = 0; offset < 2; ++offset) {
        if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + i_inner_inner) < 5329) {
          C[(((((((offset * 341056) + (((int)blockIdx.y) * 8192)) + (((int)threadIdx.y) * 512)) + (i_inner_inner * 64)) + (((int)threadIdx.x) * 4)) + j_inner_inner))] = C_local[((((offset * 32) + (i_inner_inner * 4)) + j_inner_inner))];
        }
      }
    }
  }
}

