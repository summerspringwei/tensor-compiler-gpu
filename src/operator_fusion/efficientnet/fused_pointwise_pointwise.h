#include <assert.h>

#define UPDIV(x, y) (((x)%(y))==0? ((x)/(y)): (((x)/(y))+1))
#define MAX(x, y) (((x)>(y))? (x): (y))

// 112, 112, 32, 16, 96,
// Layout: input: NHWC, weight1: CICO, weight2: CICO, output: NHWC
template<int tile_x, int tile_y, int num_block, int block_size, int compute1_tile_x, int compute1_tile_y, int compute2_tile_x, int compute2_tile_y,
int height, int width, int in_channels, int out_channels_1, int out_channels_2>
__global__ void __launch_bounds__(block_size, 1) fused_pointwise_pointwise(float* input, float* weight1, float* weight2, float* output){
  // const int tile_x = 32, tile_y = out_channels_2;
  // Load shared input 
  // const int num_iter_bx = UPDIV(height*width, tile_x);
  const int num_iter_by = UPDIV(out_channels_2, tile_y);
  const int bx = blockIdx.x / num_iter_by, by = (threadIdx.x % num_iter_by);
  // __shared__ float shared_input[tile_x * in_channels];
  // __shared__ float shared_weight1[in_channels * out_channels_1];
  // __shared__ float shared_weight2[out_channels_1 * out_channels_2];
  float __shared__ shared_pool[MAX((out_channels_1 * tile_y), (tile_x * in_channels + in_channels * out_channels_1)) + tile_x * out_channels_1];
  // __shared__ float shared_pool[MAX((out_channels_1 * tile_y), (tile_x * in_channels + in_channels * out_channels_1)) + tile_x * out_channels_1];
  float *shared_input;
  float* shared_weight1;
  float* shared_intermedia_output;
  shared_input = (float*)&shared_pool[0];
  shared_weight1 = (float*)&shared_pool[tile_x * in_channels];
  shared_intermedia_output = (float*)&shared_weight1[in_channels * out_channels_1];
  float* shared_weight2;
  shared_weight2 = shared_pool + 0;
  // __shared__ float shared_intermedia_output[tile_x * out_channels_1];

  const int input_offset = bx * tile_x * in_channels;
  const int load_shared_input_num_iters = UPDIV(tile_x * in_channels, block_size);
  for(int i=0;i<load_shared_input_num_iters; ++i){
    if(input_offset + i*block_size + threadIdx.x < height*width*in_channels){
      shared_input[i*block_size + threadIdx.x] = input[input_offset + i*block_size + threadIdx.x];
    }
  }
  // Load shared weight
  // const int load_shared_weight1_num_iters = UPDIV(in_channels * out_channels_1, block_size);
  // for(int i=0;i<load_shared_weight1_num_iters; ++i){
  //   if(i*blockDim.x + threadIdx.x < in_channels * out_channels_1){
  //     shared_weight1[i*blockDim.x + threadIdx.x] = weight1[i*blockDim.x + threadIdx.x];
  //   }
  // }
  // Load shared weight
  const int kVectorSize = 4;
  const int load_shared_weight1_num_iters = UPDIV(in_channels * out_channels_1, block_size * kVectorSize);
  for(int i=0;i<load_shared_weight1_num_iters; ++i){
    if(kVectorSize * (i*blockDim.x + threadIdx.x) < in_channels * out_channels_1){
      reinterpret_cast<float4*>(shared_weight1)[i*blockDim.x + threadIdx.x] = reinterpret_cast<float4*>(weight1)[i*blockDim.x + threadIdx.x];
    }
  }
  __syncthreads();

  // Compute the intermedia output in 16x16
  // const int compute1_tile_x = 16, compute1_tile_y = 16;
  const int num_iter_x = UPDIV(tile_x, compute1_tile_x), num_iter_y = UPDIV(out_channels_1, compute1_tile_y);
  float compute_local[num_iter_x][num_iter_y];
  #pragma unroll
  for(int x=0; x<num_iter_x; ++x){
    #pragma unroll
    for(int y=0; y<num_iter_y; ++y){
      compute_local[x][y]=0;
    }
  }
  if((tile_x % compute1_tile_x) ==0 && (tile_y % compute2_tile_y) == 0){
    const int tx = threadIdx.x / compute1_tile_y, ty = (threadIdx.x % compute1_tile_y);
    #pragma unroll
    for(int x=0; x<num_iter_x; ++x){
      int row = x * compute1_tile_x + tx;
      #pragma unroll
      for(int y=0; y<num_iter_y; ++y){
        int col = y * compute1_tile_y + ty;
        // TODO(Chunwei Xia) Do tile at here
        #pragma unroll
        for(int rk=0; rk<in_channels; ++rk){
          compute_local[x][y] += shared_input[row*in_channels+rk] * shared_weight1[rk*out_channels_1 + col];
        }
      }
    }
  }else{// Note! Here we only assume that tile_y can not be exact division
    assert((tile_x % compute1_tile_x)==0);
    const int tx = threadIdx.x / compute1_tile_y, ty = (threadIdx.x % compute1_tile_y);
    #pragma unroll
    for(int x=0; x<num_iter_x; ++x){
      int row = x * compute1_tile_x + tx;
        #pragma unroll
        for(int y=0; y<num_iter_y-1; ++y){
          int col = y * compute1_tile_y + ty;
            #pragma unroll
            for(int rk=0; rk<in_channels; ++rk){
              compute_local[x][y] += shared_input[row*in_channels+rk] * shared_weight1[rk*out_channels_1 + col];
            }
        }
    }
    #pragma unroll
    for(int x=0; x<num_iter_x; ++x){
      int row = x * compute1_tile_x + tx;
      int col = (num_iter_y-1) * compute1_tile_y + ty;
      if(col < out_channels_1){
        #pragma unroll
        for(int rk=0; rk<in_channels; ++rk){
          compute_local[x][num_iter_y-1] += shared_input[row*in_channels+rk] * shared_weight1[rk*out_channels_1 + col];
        }
      }
    }
  }
  

  // Write local intermedia result to shared memory
  #pragma unroll
  for(int x=0; x<num_iter_x; ++x){
    int row = x * compute1_tile_x + threadIdx.x / compute1_tile_y;
    if(row < tile_x){
      #pragma unroll
      for(int y=0; y<num_iter_y; ++y){
        int col = y * compute1_tile_y + (threadIdx.x % compute1_tile_y);
        if(col < out_channels_1){
          shared_intermedia_output[row*out_channels_1 + col] = compute_local[x][y];
        }
      }
    }
  }
  __syncthreads();
  // Load weight2, reuse the (input and weight1)'s shared memory
  if(tile_y<=block_size){
    const int load_shared_weight2_tile_size_x = block_size / tile_y;
    const int load_shared_weight2_num_iter_x = UPDIV(out_channels_1, load_shared_weight2_tile_size_x);
    for(int i=0; i<load_shared_weight2_num_iter_x; ++i){
      int row = i * load_shared_weight2_tile_size_x + threadIdx.x / tile_y;
      int col = (threadIdx.x % tile_y);
      if(row < out_channels_1 && col < tile_y){
        shared_weight2[row*tile_y + col] = weight2[row*out_channels_2 + col];
      }
    }
  }else{
    const int load_shared_weight2_tile_size_x = out_channels_1;
    const int load_shared_weight2_tile_size_y = UPDIV(out_channels_2, block_size);
    for(int i=0; i<load_shared_weight2_tile_size_x; ++i){
      for(int j=0; j<load_shared_weight2_tile_size_y; ++j){
        if(j * block_size + threadIdx.x < out_channels_2){
          shared_weight2[i*out_channels_2 + j * block_size + threadIdx.x] = weight2[i*out_channels_2 + j * block_size + threadIdx.x];
        }
      }
    }
  }
  
  __syncthreads();

  // Compute the second output
  // const int compute2_tile_x = 8, compute2_tile_y = 32;
  const int compute2_num_iter_x = UPDIV(tile_x, compute2_tile_x), compute2_num_iter_y = UPDIV(tile_y, compute2_tile_y);
  float compute2_local[compute2_num_iter_x][compute2_num_iter_y];
  #pragma unroll
  for(int i=0; i<compute2_num_iter_x; ++i){
    #pragma unroll
    for(int j=0;j<compute2_num_iter_y; ++j){
      compute2_local[i][j] = 0;
    }
  }

  if((tile_x%compute2_tile_x)==0 && (tile_y%compute2_tile_y)==0){
    const int tx = threadIdx.x / compute2_tile_y, ty = (threadIdx.x % compute2_tile_y);
    #pragma unroll
    for(int i=0; i<compute2_num_iter_x; ++i){
      int row = i*compute2_tile_x + tx;
      #pragma unroll
      for(int j=0;j<compute2_num_iter_y; ++j){
        int col = j*compute2_tile_y + ty;
        // TODO(Chunwei Xia) Do tile at here
        #pragma unroll
        for(int rk=0; rk<out_channels_1; ++rk){
          compute2_local[i][j] += (shared_intermedia_output[row*out_channels_1 + rk] * shared_weight2[rk * tile_y + col]);
        }
        // Store output to global memory
        output[(bx*tile_x + row) * out_channels_2 + (by * tile_y + col)] = compute2_local[i][j];
      }
    }
  }else{ // Note! Here we only assume that tile_y can not be exact division
    assert((tile_x % compute2_tile_x)==0);
    const int tx = threadIdx.x / compute2_tile_y, ty = (threadIdx.x % compute2_tile_y);
    #pragma unroll
    for(int i=0; i<compute2_num_iter_x; ++i){
      int row = i*compute2_tile_x + tx;
      #pragma unroll
      for(int j=0;j<compute2_num_iter_y-1; ++j){
        int col = j*compute2_tile_y + ty;
        #pragma unroll
        for(int rk=0; rk<out_channels_1; ++rk){
          compute2_local[i][j] += (shared_intermedia_output[row*out_channels_1 + rk] * shared_weight2[rk * tile_y + col]);
        }
        output[(bx*tile_x + row) * out_channels_2 + (by * tile_y + col)] = compute2_local[i][j];
      }
    }
    #pragma unroll
    for(int i=0; i<compute2_num_iter_x; ++i){
      int row = i*compute2_tile_x + tx;
      int col = (compute2_num_iter_y-1)*compute2_tile_y + (threadIdx.x % compute2_tile_y);
      if(col < tile_y){
        #pragma unroll
        for(int rk=0; rk<out_channels_1; ++rk){
          compute2_local[i][(compute2_num_iter_y-1)] += (shared_intermedia_output[row*out_channels_1 + rk] * shared_weight2[rk * tile_y + col]);
        }
        output[(bx*tile_x + row) * out_channels_2 + (by * tile_y + col)] = compute2_local[i][(compute2_num_iter_y-1)];
      }
    }
  }
}
  // Store output to global memory
  // #pragma unroll
  // for(int i=0; i<compute2_num_iter_x; ++i){
  //   int row = i*compute2_tile_x + threadIdx.x / compute2_tile_y;
  //   // if(row < tile_x){
  //     #pragma unroll
  //     for(int j=0;j<compute2_num_iter_y; ++j){
  //       int col = j*compute2_tile_y + (threadIdx.x % compute2_tile_y);
  //       // if(col < tile_y){
  //         output[(bx*tile_x + row) * out_channels_2 + (by * tile_y + col)] = compute2_local[i][j];
  //       // }
  //     }
  //   // }
  // }