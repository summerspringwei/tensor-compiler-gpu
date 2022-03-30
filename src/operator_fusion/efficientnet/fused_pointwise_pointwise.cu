#include <assert.h>

#define UPDIV(x, y) (((x)%(y))==0? ((x)/(y)): (((x)/(y))+1))

// 112, 112, 32, 16, 96,
// Layout: input: NHWC, weight1: COCI, weight2: COCI, output: NHWC
template<int num_block, int block_size, int height, int width, int in_channels, int out_channels_1, int out_channels_2>
__global__ void __launch_bounds__(block_size) fused_pointwise_pointwise(float* input, float* weight1, float* weight2, float* output){
  const int tile_x = 32, tile_y = out_channels_2;
  // Load shared input 
  const int num_iter_bx = UPDIV(height*width, tile_x), num_iter_by = UPDIV(out_channels_2, tile_y);
  const int bx = blockIdx.x / num_iter_by, by = threadIdx.x % num_iter_by;

  __shared__ float shared_input[tile_x * in_channels + in_channels * out_channels_1];
  __shared__ float* shared_weight1 = shared_input[tile_x * in_channels];
  __shared__ float* shared_weight2 = shared_input[0];
  __shared__ float shared_intermedia_output[tile_x * out_channels_1];

  const int input_offset = bx * tile_x * in_channels;
  const int load_shared_input_num_iters = UPDIV(tile_x * in_channels, block_size);
  for(int i=0;i<load_shared_input_num_iters; ++i){
    if(input_offset + i*block_size + threadIdx.x < height*width*in_channels){
      shared_input[i*block_size + threadIdx.x] = input[input_offset + i*block_size + threadIdx.x];
    }
  }
  // Load shared weight
  const int load_shared_weight1_num_iters = UPDIV(in_channels * out_channels_1, block_size);
  for(int i=0;i<load_shared_weight1_num_iters; ++i){
    if(i*blockDim.x + threadIdx.x < in_channels * out_channels_1){
      shared_weight1[i*blockDim.x + threadIdx.x] = weight1[i*blockDim.x + threadIdx.x];
    }
  }
  __syncthreads();

  // Compute the intermedia output in 16x16
  const int compute_tile_size_x = 16, compute_tile_size_y = 16;
  const int num_iter_x = UPDIV(tile_x, compute_tile_size_x), num_iter_y = UPDIV(out_channels_1, compute_tile_size_y);
  float compute_local[num_iter_x][num_iter_y];
  #pragma unroll
  for(int x=0; x<num_iter_x; ++x){
    #pragma unroll
    for(int y=0; y<num_iter_y; ++y){
      compute_local[x][y]=0;
    }
  }
  #pragma unroll
  for(int x=0; x<num_iter_x; ++x){
    int row = x * compute_tile_size_x + threadIdx.x / compute_tile_size_y;
    if(row<tile_x){
      #pragma unroll
      for(int y=0; y<num_iter_y; ++y){
        int col = y * compute_tile_size_y + threadIdx.x % compute_tile_size_y;
        if(col < out_channels_1){
          // TODO(Chunwei Xia) Do tile at here
          #pragma unroll
          for(int rk=0; rk<in_channels; ++rk){
            compute_local[x][y] += shared_input[row*in_channels+rk] * shared_weight1[rk*out_channels_1 + col];
          }
        }
      }
    }
  }

  // Write local intermedia result to shared memory
  #pragma unroll
  for(int x=0; x<num_iter_x; ++x){
    int row = x * compute_tile_size_x + threadIdx.x / compute_tile_size_y;
    #pragma unroll
    for(int y=0; y<num_iter_y; ++y){
      int col = y * compute_tile_size_y + threadIdx.x % compute_tile_size_y;
      shared_input[row][col] = compute_local[x][y];
    }
  }

  // Load weight2, reuse the (input and weight1)'s shared memory
  if(tile_y<=block_size){
    const int load_shared_weight2_tile_size_x = block_size / tile_y;
    const int load_shared_weight2_num_iter_x = UPDIV(out_channels_1, load_shared_weight2_tile_size_x);
    for(int i=0; i<load_shared_weight2_num_iter_x; ++i){
      int row = i * load_shared_weight2_tile_size_x + threadIdx.x / tile_y;
      int col = threadIdx.x % tile_y;
      if(row < out_channels_1 && col < out_channels_2){
        shared_weight2[row*tile_y + col] = weight2[row*out_channels_2 + col];
      }
    }
  }else{
    assert(true);
  }
  
  __syncthreads();

  // Start compute
  const int compute2_tile_x = 8, compute2_tile_y = 32;
  const int compute2_num_iter_x = UPDIV(tile_x, compute2_tile_x), compute2_num_iter_y = UPDIV(tile_y, compute2_tile_y);
  float compute2_local[compute2_num_iter_x][compute2_num_iter_y];
  #pragma unroll
  for(int i=0; i<compute2_num_iter_x; ++i){
    #pragma unroll
    for(int j=0;j<compute2_num_iter_y; ++j){
      compute2_local[i][j] = 0;
    }
  }

  #pragma unroll
  for(int i=0; i<compute2_num_iter_x; ++i){
    int row = i*compute2_tile_x + threadIdx.x / compute2_tile_y;
    if(row < tile_x){
      #pragma unroll
      for(int j=0;j<compute2_num_iter_y; ++j){
        int col = j*compute2_tile_y + threadIdx.x % compute2_tile_y;
        if(col < tile_y){
          // TODO(Chunwei Xia) Do tile at here
          #pragma unroll
          for(int rk=0; rk<out_channels_1; ++rk){
            compute2_local[i][j] += (shared_intermedia_output[row*out_channels_1 + rk] * shared_weight2[rk * out_channels_1 + col]);
          }
          // Store output to global memory
          output[(bx*tile_x + row) * out_channels_2 + (by*tile_y + col)] = compute2_local[i][j];
        }
      }
    }
  }
}
