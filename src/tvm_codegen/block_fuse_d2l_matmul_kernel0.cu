#include <stdio.h>

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
__device__ void d2l_matmul_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute, 
  int threadIdx_x, int threadIdx_y, int blockIdx_x, int blockIdx_y) {
  const dim3 blockDim(16, 16, 1);
  if(threadIdx_x*blockDim.x + threadIdx_y > 256){
    return;
  }
  const dim3 blockIdx(blockIdx_x, blockIdx_y, 0);

  float compute_local[32];
  __shared__ float A_shared[4096];
  __shared__ float B_shared[2048];
  float A_shared_local[8];
  float B_shared_local[4];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
      compute_local[(((i_c_init * 4) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int rik_outer = 0; rik_outer < 5; ++rik_outer) {
    __syncthreads();
    for (int ax0_inner = 0; ax0_inner < 8; ++ax0_inner) {
      for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
        if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + ax0_inner) < 5329) {
          A_shared[(((((((int)threadIdx.y) * 256) + (ax0_inner * 32)) + (((int)threadIdx.x) * 2)) + ax1_inner))] = A[(((((((((int)blockIdx.y) * 20480) + (((int)threadIdx.y) * 1280)) + (ax0_inner * 160)) + (rik_outer * 32)) + (((int)threadIdx.x) * 2)) + ax1_inner))];
        }
      }
    }
    for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
      for (int ax1_inner1 = 0; ax1_inner1 < 4; ++ax1_inner1) {
        B_shared[(((((((int)threadIdx.y) * 128) + (ax0_inner1 * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner1))] = B[((((((rik_outer * 2048) + (((int)threadIdx.y) * 128)) + (ax0_inner1 * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner1))];
      }
    }
    __syncthreads();
    for (int rik_inner = 0; rik_inner < 32; ++rik_inner) {
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + ax0) < 5329) {
          A_shared_local[(ax0)] = A_shared[((((((int)threadIdx.y) * 256) + (ax0 * 32)) + rik_inner))];
        }
      }
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        B_shared_local[(ax1)] = B_shared[((((rik_inner * 64) + (((int)threadIdx.x) * 4)) + ax1))];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        for (int j_c = 0; j_c < 4; ++j_c) {
          if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + i_c) < 5329) {
            compute_local[(((i_c * 4) + j_c))] = (compute_local[(((i_c * 4) + j_c))] + (A_shared_local[(i_c)] * B_shared_local[(j_c)]));
          }
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    for (int j_inner_inner = 0; j_inner_inner < 4; ++j_inner_inner) {
      if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + i_inner_inner) < 5329) {
        compute[((((((((int)blockIdx.y) * 8192) + (((int)threadIdx.y) * 512)) + (i_inner_inner * 64)) + (((int)threadIdx.x) * 4)) + j_inner_inner))] = compute_local[(((i_inner_inner * 4) + j_inner_inner))];
      }
    }
  }
}

extern "C" __global__ void g_d2l_matmul_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ compute) {
  float compute_local[32];
  __shared__ float A_shared[4096];
  __shared__ float B_shared[2048];
  float A_shared_local[8];
  float B_shared_local[4];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 4; ++j_c_init) {
      compute_local[(((i_c_init * 4) + j_c_init))] = 0.000000e+00f;
    }
  }
  for (int rik_outer = 0; rik_outer < 5; ++rik_outer) {
    __syncthreads();
    for (int ax0_inner = 0; ax0_inner < 8; ++ax0_inner) {
      for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
        if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + ax0_inner) < 5329) {
          A_shared[(((((((int)threadIdx.y) * 256) + (ax0_inner * 32)) + (((int)threadIdx.x) * 2)) + ax1_inner))] = A[(((((((((int)blockIdx.y) * 20480) + (((int)threadIdx.y) * 1280)) + (ax0_inner * 160)) + (rik_outer * 32)) + (((int)threadIdx.x) * 2)) + ax1_inner))];
        }
      }
    }
    for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
      for (int ax1_inner1 = 0; ax1_inner1 < 4; ++ax1_inner1) {
        B_shared[(((((((int)threadIdx.y) * 128) + (ax0_inner1 * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner1))] = B[((((((rik_outer * 2048) + (((int)threadIdx.y) * 128)) + (ax0_inner1 * 64)) + (((int)threadIdx.x) * 4)) + ax1_inner1))];
      }
    }
    __syncthreads();
    for (int rik_inner = 0; rik_inner < 32; ++rik_inner) {
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + ax0) < 5329) {
          A_shared_local[(ax0)] = A_shared[((((((int)threadIdx.y) * 256) + (ax0 * 32)) + rik_inner))];
        }
      }
      for (int ax1 = 0; ax1 < 4; ++ax1) {
        B_shared_local[(ax1)] = B_shared[((((rik_inner * 64) + (((int)threadIdx.x) * 4)) + ax1))];
      }
      for (int i_c = 0; i_c < 8; ++i_c) {
        for (int j_c = 0; j_c < 4; ++j_c) {
          if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + i_c) < 5329) {
            compute_local[(((i_c * 4) + j_c))] = (compute_local[(((i_c * 4) + j_c))] + (A_shared_local[(i_c)] * B_shared_local[(j_c)]));
          }
        }
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    for (int j_inner_inner = 0; j_inner_inner < 4; ++j_inner_inner) {
      if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.y) * 8)) + i_inner_inner) < 5329) {
        compute[((((((((int)blockIdx.y) * 8192) + (((int)threadIdx.y) * 512)) + (i_inner_inner * 64)) + (((int)threadIdx.x) * 4)) + j_inner_inner))] = compute_local[(((i_inner_inner * 4) + j_inner_inner))];
        // if(compute_local[(((i_inner_inner * 4) + j_inner_inner))] !=160){
        //   printf("Error <<<dim3(%d, %d, %d), dim3(%d, %d, %d)>>> %f\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, compute_local[(((i_inner_inner * 4) + j_inner_inner))]);
        // }
      }
    }
  }
}

extern "C" __global__ void check_result(float* output, int size, float expected){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size){
    return;
  }
  if(output[idx] != expected){
    printf("Error <<<dim3(%d, %d, %d), dim3(%d, %d, %d)>>> %f\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, output[idx]);
  }
}

extern "C" __global__ void BlockFusion_matmul(float* d_input, float* d_filter1, 
  float* d_output1, float* d_filter2, float* d_output2){
    if(blockIdx.y < (73*73 / 16 / 8 + 1)){
      d2l_matmul_kernel0(d_input, d_filter1, d_output1, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    }else if(blockIdx.y >= (73*73 / 16 / 8 + 1)){
      d2l_matmul_kernel0(d_input, d_filter2, d_output2, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y-(73*73 / 16 / 8 + 1));
    }
}

extern void BlockFusion_matmul_Call(const dim3 &grids, const dim3 &blocks, unsigned mem, cudaStream_t stream, 
  float* d_input, float* d_filter1, float* d_output1, float* d_filter2, float* d_output2){
    BlockFusion_matmul<<<grids, blocks, mem, stream>>>(d_input, d_filter1, d_output1, d_filter2, d_output2);
}

