
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
extern "C" __global__ void __launch_bounds__(32) mmult_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[4];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[64];
  float A_shared_local[4];
  float B_shared_local[1];
  for (int i_c_init = 0; i_c_init < 4; ++i_c_init) {
    C_local[(i_c_init)] = 0.000000e+00f;
  }
  for (int rik_outer = 0; rik_outer < 8; ++rik_outer) {
    __syncthreads();
    for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
      for (int ax1_outer = 0; ax1_outer < 16; ++ax1_outer) {
        A_shared[(((((ax0_outer * 512) + (((int)threadIdx.y) * 32)) + (ax1_outer * 2)) + ((int)threadIdx.x)))] = A[(((((((((int)blockIdx.y) * 16384) + (ax0_outer * 4096)) + (((int)threadIdx.y) * 256)) + (rik_outer * 32)) + (ax1_outer * 2)) + ((int)threadIdx.x)))];
      }
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
      B_shared[((((ax1_outer1 * 32) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)))] = B[((((((rik_outer * 8192) + (ax1_outer1 * 4096)) + (((int)threadIdx.y) * 256)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int rik_inner = 0; rik_inner < 32; ++rik_inner) {
      for (int ax0 = 0; ax0 < 4; ++ax0) {
        A_shared_local[(ax0)] = A_shared[((((((int)threadIdx.y) * 128) + (ax0 * 32)) + rik_inner))];
      }
      B_shared_local[(0)] = B_shared[(((rik_inner * 2) + ((int)threadIdx.x)))];
      for (int i_c = 0; i_c < 4; ++i_c) {
        C_local[(i_c)] = (C_local[(i_c)] + (A_shared_local[(i_c)] * B_shared_local[(0)]));
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 4; ++i_inner_inner) {
    C[((((((((int)blockIdx.y) * 16384) + (((int)threadIdx.y) * 1024)) + (i_inner_inner * 256)) + (((int)blockIdx.x) * 2)) + ((int)threadIdx.x)))] = C_local[(i_inner_inner)];
  }
}

