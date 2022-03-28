#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

const int height = 56, width = 56, in_channels = 24, out_channels = 144, kernel_width=3, kernel_height=3;
const int tile_size_height = 4, tile_size_width=4, tile_size_out_channels = 72;
const int num_tile_height = (height / tile_size_height), num_tile_width = (width / tile_size_width), num_tile_out_channels = (out_channels / tile_size_out_channels);
const int num_blocks =  num_tile_height * num_tile_width * num_tile_out_channels;
const int padded_tile_size_height = (tile_size_height+2), padded_tile_size_width = (tile_size_width+2);

const int num_threads = 256;
const int size_pointwise_weight = in_channels * tile_size_out_channels;

__device__ int updiv(int a, int b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}



#define UPDIV(x, y) (((x)%(y))==0? ((x)/(y)): (((x)/(y))+1))

// input: NHWC, pointwise_weight: CICO, depthwise_weight: KHKWCI, output: NHWC
extern "C" __global__ void __launch_bounds__(256) fused_pointwise_depthwise(
  float* __restrict__ input,  float* __restrict__ pointwise_weight, float* __restrict__ depthwise_weight, float* __restrict__ output) {
  __shared__ float shared_input[padded_tile_size_height*padded_tile_size_width*in_channels];// CHW
  __shared__ float shared_pointwise_weight[in_channels * tile_size_out_channels];// CICO
  __shared__ float shared_intermedia_result[padded_tile_size_height*padded_tile_size_width*tile_size_out_channels];//HWC
  __shared__ float* shared_depthwise_weight;//HWC

  const int bx = blockIdx.x / (num_tile_width * num_tile_out_channels);
  const int by = (blockIdx.x % (num_tile_width * num_tile_out_channels)) / num_tile_out_channels;
  const int bz = blockIdx.x % num_tile_out_channels;
  // Using Asynchronous memcpy between global memory and shared memory
  auto group = cooperative_groups::this_thread_block();
  cooperative_groups::memcpy_async(group, shared_pointwise_weight, &pointwise_weight[0], kernel_height*kernel_width*out_channels);

  // Load pointwise_weight into shared memory
  // const int load_pointwise_weight_bz = blockIdx.x % num_tile_out_channels;
  // const int load_pointwise_weight_tx = threadIdx.x / num_tile_out_channels;
  // const int load_pointwise_weight_ty = threadIdx.x % num_tile_out_channels;
  // const int load_pointwise_weight_num_iters = UPDIV(in_channels*tile_size_out_channels, blockDim.x);
  // #pragma unroll
  // for(int i=0; i<load_pointwise_weight_num_iters; ++i){
  //   int idx_shared_pointwise_weight = i * blockDim.x + threadIdx.x;
  //   int row = idx_shared_pointwise_weight / tile_size_out_channels;
  //   int col = idx_shared_pointwise_weight % tile_size_out_channels;
  //   int idx_pointwise_weight = row*out_channels + bz * tile_size_out_channels + col;
  //   // Note in tvm kernel the layout of pointwise_weight is in COCI
  //   // int idx_pointwise_weight = (bz * tile_size_out_channels + col) * in_channels + row;
  //   if(idx_shared_pointwise_weight >= in_channels * tile_size_out_channels){
  //     continue;
  //   }
  //   shared_pointwise_weight[idx_shared_pointwise_weight] = pointwise_weight[idx_pointwise_weight];
  // }


  // Load and pad input
  const int load_tz = threadIdx.x / (padded_tile_size_height*padded_tile_size_width);
  const int load_tx = (threadIdx.x % (padded_tile_size_height*padded_tile_size_width)) / padded_tile_size_width;
  const int load_ty = threadIdx.x % padded_tile_size_width;
  // TODO(Chunwei Xia) Here the `in_channels` should divide the num_tile_out_channels (2us)
  const int load_shared_tile_z = (blockDim.x / (padded_tile_size_height*padded_tile_size_width));
  if(threadIdx.x < load_shared_tile_z * padded_tile_size_height*padded_tile_size_width) {
    if(bx==0 && by==0){
      #pragma unroll
      for(int i=0; i<UPDIV(in_channels, load_shared_tile_z); ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==0 || load_ty==0) ?
          0: input[idx_input];
      }
    }else if(bx==0 && by==num_tile_width-1){
      #pragma unroll
      for(int i=0; i<UPDIV(in_channels, load_shared_tile_z); ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==0 || load_ty==padded_tile_size_width-1) ?
          0: input[idx_input];
      }
    }else if(bx==num_tile_height-1 && by==0){
      #pragma unroll
      for(int i=0; i<UPDIV(in_channels, load_shared_tile_z); ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==padded_tile_size_height-1 || load_ty==0) ?
          0: input[idx_input];
      }
    }else if(bx==num_tile_height-1 && by==num_tile_width-1){
      #pragma unroll
      for(int i=0; i<UPDIV(in_channels, load_shared_tile_z); ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==padded_tile_size_height-1 || load_ty==padded_tile_size_width-1) ?
          0: input[idx_input];
      }
    }else if(bx>0 && bx < num_tile_height-1 && by==0){// Left middle
      #pragma unroll
      for(int i=0; i<UPDIV(in_channels, load_shared_tile_z); ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_ty==0) ?
          0: input[idx_input];
      }
    }else if(bx>0 && bx < num_tile_height-1 && by==num_tile_width-1){// Right middle
      #pragma unroll
      for(int i=0; i<UPDIV(in_channels, load_shared_tile_z); ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_ty==padded_tile_size_width-1) ?
          0: input[idx_input];
      }
    }else if(bx==0 && by>0 && by<num_tile_width-1){
      #pragma unroll
      for(int i=0; i<UPDIV(in_channels, load_shared_tile_z); ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==0) ?
          0: input[idx_input];
      }
    }else if(bx==num_tile_height-1 && by>0 && by<num_tile_width-1){
      #pragma unroll
      for(int i=0; i<UPDIV(in_channels, load_shared_tile_z); ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==padded_tile_size_height-1) ?
          0: input[idx_input];
      }
    }else{
      #pragma unroll
      for(int i=0; i<UPDIV(in_channels, load_shared_tile_z); ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input]=input[idx_input];
      }
    }
  }
  group.sync();
  __syncthreads();

  // Each block computes (padded_tile_size_height*padded_tile_size*width) * (tile_size_out_channels)
  // (6*6) * 72, with 256 threads, we compute 8*32 outputs every time.
  const int compute_tile_size_x = 16, compute_tile_size_y = 16;
  // const int compute_num_tile_x = updiv(padded_tile_size_height*padded_tile_size_width, compute_tile_size_x) ;
  const int compute_num_tile_x = UPDIV(padded_tile_size_height * padded_tile_size_width, compute_tile_size_x);
  const int compute_num_tile_y = UPDIV(tile_size_out_channels, compute_tile_size_y);
  float pointwise_local[compute_num_tile_x * compute_num_tile_y];
  
  #pragma unroll
  for(int i=0; i<compute_num_tile_x; ++i){
    #pragma unroll
    for(int j=0; j<compute_num_tile_y; ++j){
      pointwise_local[i*compute_num_tile_y + j] = 0;
    }
  }
  const int pointwise_tx = threadIdx.x / compute_tile_size_y;
  const int pointwise_ty = (threadIdx.x % compute_tile_size_y);
  #pragma unroll
  for(int rk=0; rk<in_channels; ++rk){
    #pragma unroll
    for(int i=0; i<compute_num_tile_x; ++i){
      int row_idx = i*compute_tile_size_x + pointwise_tx;
      #pragma unroll
      for(int j=0; j<compute_num_tile_y; ++j){
        int col_idx = j*compute_tile_size_y + pointwise_ty;
        if(row_idx >= (padded_tile_size_height*padded_tile_size_width) || col_idx >= tile_size_out_channels){
          continue;
        }
        int idx_shared_input = rk*(padded_tile_size_height*padded_tile_size_width) + row_idx;
        int idx_shared_pointwise_weight = rk*tile_size_out_channels + col_idx;
        pointwise_local[i*compute_num_tile_y + j] += 
          shared_input[idx_shared_input] * shared_pointwise_weight[idx_shared_pointwise_weight];
      }
    }
  }
  // Save to shared memory as HWC format
  #pragma unroll
  for(int i=0; i<compute_num_tile_x; ++i){
    #pragma unroll
    for(int j=0; j<compute_num_tile_y; ++j){
      int row_idx = i*compute_tile_size_x + pointwise_tx;
      int col_idx = j*compute_tile_size_y + pointwise_ty;
      if(row_idx >= (padded_tile_size_height*padded_tile_size_width) || col_idx >= tile_size_out_channels){
        continue;
      }
      shared_intermedia_result[row_idx*tile_size_out_channels + col_idx] = pointwise_local[i*compute_num_tile_y+j];
      // shared_intermedia_result[col_idx * padded_tile_size_height*padded_tile_size_width + row_idx]= pointwise_local[i*compute_num_tile_y+j];
    }
  }

  // TODO(Chunwei Xia) Use coperative group to copy weights from global memory to shared memory 
  // Load depthwise load to shared memory (kernel_height*kernel_width) * tile_size_out_channels, 
  // let threads spread along out_channels to access to memory continuous (1us)
  const int kVecSize = 4;
  // assert(tile_size_out_channels % kVecSize == 0);
  shared_depthwise_weight = shared_pointwise_weight; // Reuse the pointwise_weight shared memory
  const int load_depthwise_weight_num_iter = UPDIV(kernel_height*kernel_width*tile_size_out_channels, blockDim.x);
  const int load_depthwise_weight_bz = blockIdx.x % num_tile_out_channels;
  #pragma unroll
  for(int i=0; i<load_depthwise_weight_num_iter; ++i) {
    int idx_shared_depthwise_weight = (i * blockDim.x + threadIdx.x);
    if(idx_shared_depthwise_weight >= tile_size_out_channels*kernel_height*kernel_width) {
      continue;
    }
    int row = idx_shared_depthwise_weight / tile_size_out_channels;
    int col = idx_shared_depthwise_weight % tile_size_out_channels;
    int idx_depthwise_weight = row * out_channels + (col + load_depthwise_weight_bz * tile_size_out_channels);
    // reinterpret_cast<float4*>(shared_depthwise_weight)[idx_shared_depthwise_weight] = reinterpret_cast<float4*>(depthwise_weight)[idx_depthwise_weight];
    shared_depthwise_weight[idx_shared_depthwise_weight] = depthwise_weight[idx_depthwise_weight];
  }
  
  __syncthreads();

  const int depth_tile_size_x = 2;
  const int depth_tile_size_z = 128;
  const int depth_tz = threadIdx.x / depth_tile_size_x;
  const int depth_tx = threadIdx.x % depth_tile_size_x;
  if(depth_tz >= tile_size_out_channels){
    return;
  }
  // Every two threads compute a channel, thus the weights are keep in registers and the features are also keep in registers
  // Load weights to register
  float weights_local[kernel_height][kernel_width];
  #pragma unroll
  for(int kh=0; kh<kernel_height; ++kh){
    #pragma unroll
    for(int kw=0; kw<kernel_width; ++kw){
      weights_local[kh][kw] = shared_depthwise_weight[(kh*kernel_width*tile_size_out_channels) + kw * tile_size_out_channels + depth_tz];
    }
  }
  // Load feature to resiter
  float features_local[4][padded_tile_size_width];
  #pragma unroll
  for(int i=0; i<4; ++i){
    #pragma unroll
    for(int j=0; j<padded_tile_size_width; ++j){
      features_local[i][j] = shared_intermedia_result[(depth_tx * 2 + i)*padded_tile_size_width*tile_size_out_channels + j*tile_size_out_channels + depth_tz];
    }
  }
  float depth_local[2][4];
  #pragma unroll
  for(int i=0; i<8; ++i){
    depth_local[i/4][i%4]=0;
  }
  #pragma unroll
  for(int i=0; i<2; ++i){
    #pragma unroll
    for(int j=0; j<tile_size_width; ++j){
      #pragma unroll
      for(int kh=0; kh<kernel_height; ++kh){
        #pragma unroll
        for(int kw=0; kw<kernel_width; ++kw){
          depth_local[i][j] += features_local[i+kh][j+kw] * weights_local[kh][kw];
        }
      }
      output[(bx*tile_size_height+depth_tx*2+i)*width*out_channels + (by*tile_size_width+j)*out_channels + bz * tile_size_out_channels + depth_tz] = depth_local[i][j];
    }
  }

  // Load feature to resiter
  // float features_local[padded_tile_size_height / 2][padded_tile_size_width];
  // #pragma unroll
  // for(int i=0; i<padded_tile_size_height / 2; ++i){
  //   #pragma unroll
  //   for(int j=0; j<padded_tile_size_width; ++j){
  //     int new_i = (threadIdx.x % 2) * 2 + ((threadIdx.x % 2) == 0? 1: -1) * i;
  //     features_local[new_i][j] = shared_intermedia_result[(depth_tx * 2 + i)*padded_tile_size_width*tile_size_out_channels + j*tile_size_out_channels + depth_tz];
  //   }
  // }

  // #pragma unroll
  // for(int i=0; i<2; ++i){
  //   #pragma unroll
  //   for(int j=0; j<4; ++j){
  //     #pragma unroll
  //     for(int kh=0; kh<kernel_height-1; ++kh){
  //       #pragma unroll
  //       for(int kw=0; kw<kernel_width; ++kw){
  //         depth_local[i][j] += features_local[i+kh][j+kw] * weights_local[kh][kw];
  //       }
  //     }
  //     output[(bx*tile_size_height+depth_tx*2+i)*width*out_channels + (by*tile_size_width+j)*out_channels + bz * tile_size_out_channels + depth_tz] = depth_local[i][j];
  //   }
  // }


}
