#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

const int height = 56, width = 56, in_channels = 24, out_channels = 144, kernel_width=3, kernel_height=3;
const int tile_size_height = 4, tile_size_width=4, tile_size_out_channels = 144;
const int num_tile_height = (height / tile_size_height), num_tile_width = (width / tile_size_width), num_tile_out_channels = (out_channels / tile_size_out_channels);
const int num_blocks =  num_tile_height * num_tile_width * num_tile_out_channels;
const int padded_tile_size_height = (tile_size_height+2), padded_tile_size_width = (tile_size_width+2);

const int num_threads = 256;
const int size_pointwise_weight = in_channels * out_channels;

__device__ int updiv(int a, int b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

// input: NHWC, pointwise_weight: CICO, depthwise_weight: KHKWCI, output: NHWC
extern "C" __global__ void __launch_bounds__(256) fused_pointwise_depthwise(
  float* __restrict__ input,  float* __restrict__ pointwise_weight, float* __restrict__ depthwise_weight, float* __restrict__ output) {
  __shared__ float shared_input[padded_tile_size_height*padded_tile_size_width*in_channels];// CHW
  __shared__ float shared_pointwise_weight[in_channels * out_channels];// CICO
  __shared__ float shared_intermedia_result[padded_tile_size_height*padded_tile_size_width*out_channels];//HWC
  __shared__ float shared_depthwise_weight[out_channels*kernel_height*kernel_width];//HWC

  // Load pointwise_weight into shared memory
  #pragma unroll
  for(int i=0; i<updiv(size_pointwise_weight, blockDim.x); ++i){
    int idx = i * blockDim.x + threadIdx.x;
    if(idx >= size_pointwise_weight){
      continue;
    }
    shared_pointwise_weight[idx] = pointwise_weight[idx];
  }

  // Load and pad input
  const int bx = blockIdx.x / (num_tile_width * num_tile_out_channels);
  const int by = (blockIdx.x % (num_tile_width * num_tile_out_channels));
  // const int bz = blockIdx.x % num_tile_out_channels;
  const int load_tz = threadIdx.x / (padded_tile_size_height*padded_tile_size_width);
  const int load_tx = (threadIdx.x % (padded_tile_size_height*padded_tile_size_width)) / padded_tile_size_width;
  const int load_ty = threadIdx.x % padded_tile_size_width;
  // TODO(Chunwei Xia) Here the `in_channels` should divide the num_tile_out_channels
  const int load_shared_tile_z = (blockDim.x / (padded_tile_size_height*padded_tile_size_width)); // 256 / (6*6) == 7
  if(threadIdx.x < load_shared_tile_z * padded_tile_size_height*padded_tile_size_width) {
    if(bx==0 && by==0){
      #pragma unroll
      for(int i=0; i<(in_channels / load_shared_tile_z)+1; ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==0 || load_ty==0) ?
          0: input[idx_input];
      }
    }else if(bx==0 && by==num_tile_width-1){
      #pragma unroll
      for(int i=0; i<(in_channels / load_shared_tile_z)+1; ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==0 || load_ty==padded_tile_size_width-1) ?
          0: input[idx_input];
      }
    }else if(bx==num_tile_height-1 && by==0){
      #pragma unroll
      for(int i=0; i<(in_channels / load_shared_tile_z)+1; ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==padded_tile_size_height-1 || load_ty==0) ?
          0: input[idx_input];
      }
    }else if(bx==num_tile_height-1 && by==num_tile_width-1){
      #pragma unroll
      for(int i=0; i<(in_channels / load_shared_tile_z)+1; ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==padded_tile_size_height-1 || load_ty==padded_tile_size_width-1) ?
          0: input[idx_input];
      }
    }else if(bx>0 && bx < num_tile_height-1 && by==0){// Left middle
      #pragma unroll
      for(int i=0; i<(in_channels / load_shared_tile_z)+1; ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_ty==0) ?
          0: input[idx_input];
      }
    }else if(bx>0 && bx < num_tile_height-1 && by==num_tile_width-1){// Right middle
      #pragma unroll
      for(int i=0; i<(in_channels / load_shared_tile_z)+1; ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_ty==padded_tile_size_width-1) ?
          0: input[idx_input];
      }
    }else if(bx==0 && by>0 && by<num_tile_width-1){
      #pragma unroll
      for(int i=0; i<(in_channels / load_shared_tile_z)+1; ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==0) ?
          0: input[idx_input];
      }
    }else if(bx==num_tile_height-1 && by>0 && by<num_tile_width-1){
      #pragma unroll
      for(int i=0; i<(in_channels / load_shared_tile_z)+1; ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input] = (load_tx==padded_tile_size_height-1) ?
          0: input[idx_input];
      }
    }else{
      #pragma unroll
      for(int i=0; i<(in_channels / load_shared_tile_z)+1; ++i) {
        if(i*load_shared_tile_z+load_tz >= in_channels) {continue;}
        int idx_shared_input = (i*load_shared_tile_z + load_tz) * padded_tile_size_height * padded_tile_size_width + load_tx * padded_tile_size_width + load_ty;
        int idx_input = (bx*tile_size_height+load_tx-1)*height*in_channels + (by*tile_size_width+load_ty-1)*in_channels + (i*load_shared_tile_z + load_tz);
        shared_input[idx_shared_input]=input[idx_input];
      }
    }
  }
  
  __syncthreads();

  // Using Asynchronous memcpy between global memory and shared memory
  // auto group = cooperative_groups::this_thread_block();
  // cooperative_groups::memcpy_async(group, shared_depthwise_weight, &depthwise_weight[0], in_channels*out_channels);
  
  // Each block computes (padded_tile_size_height*padded_tile_size*width) * (tile_size_out_channels)
  // (6*6) * 144, with 256 threads, we compute 8*32 outputs every time.
  // Thus we need compute updiv(6*6, 8) x updiv(144, 32)
  const int compute_tile_size_x = 8, compute_tile_size_y = 32;
  // const int compute_num_tile_x = updiv(padded_tile_size_height*padded_tile_size_width, compute_tile_size_x) ;
  const int compute_num_tile_x = padded_tile_size_height * padded_tile_size_width / compute_tile_size_x + 1;
  const int compute_num_tile_y = out_channels / compute_tile_size_y + 1;
  float pointwise_local[compute_num_tile_x * compute_num_tile_y];
  
  #pragma unroll
  for(int i=0; i<compute_num_tile_x; ++i){
    #pragma unroll
    for(int j=0; j<compute_num_tile_y; ++j){
      pointwise_local[i*compute_num_tile_y + j] = 0;
    }
  }
  #pragma unroll
  for(int rk=0; rk<in_channels; ++rk){
    #pragma unroll
    for(int i=0; i<compute_num_tile_x; ++i){
      #pragma unroll
      for(int j=0; j<compute_num_tile_y; ++j){
        int row_idx = i*compute_tile_size_x + threadIdx.x / compute_tile_size_y ;
        int col_idx = j*compute_tile_size_y + (threadIdx.x % compute_tile_size_y);
        if(row_idx >= (padded_tile_size_height*padded_tile_size_width) || col_idx >= out_channels){
          continue;
        }
        int idx_shared_input = rk*(padded_tile_size_height*padded_tile_size_width) + row_idx;
        int idx_shared_pointwise_weight = rk*out_channels + col_idx;
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
      int row_idx = i*compute_tile_size_x+threadIdx.x / compute_tile_size_y ;
      int col_idx = j*compute_tile_size_y + (threadIdx.x % compute_tile_size_y);
      if(row_idx >= (padded_tile_size_height*padded_tile_size_width) || col_idx >= out_channels){
        continue;
      }
      shared_intermedia_result[row_idx*out_channels + col_idx] = pointwise_local[i*compute_num_tile_y+j];
      // shared_intermedia_result[col_idx * padded_tile_size_height*padded_tile_size_width + row_idx]= pointwise_local[i*compute_num_tile_y+j];
    }
  }

  // TODO(Chunwei Xia) Use coperative group to copy weights from global memory to shared memory 
  // Load depthwise load to shared memory 3*3*144, let threads spread along out_channels to access to memory continuous (1us)
  #pragma unroll
  for(int i=0; i<out_channels*kernel_height*kernel_height / blockDim.x + 1; ++i) {
    int idx = i*blockDim.x + threadIdx.x;
    if(idx>=out_channels*kernel_height*kernel_width){
      continue;
    }
    shared_depthwise_weight[idx] = depthwise_weight[idx];
  }
  // group.sync();
  __syncthreads();
  // DepthwiseConv2d[threadIdx.x] = shared_intermedia_result[threadIdx.x];
  // we need compute tile_size_height*tile_size_width*out_channels with 256 threads
  // const int num_iters = updiv(out_channels*tile_size_height*tile_size_width, blockDim.x);
  // Note: 4*4*144 / 256 = 9
  const int num_iters = out_channels*tile_size_height*tile_size_width / 256;
  const int depth_tile_size_z = blockDim.x / (tile_size_height*tile_size_width);
  float depthwise_local[num_iters];
  #pragma unroll
  for(int i=0; i<num_iters; ++i){
    depthwise_local[i] = 0;
  }

  const int depth_tile_size_out_channels = 16; // 144=16*9
  const int depth_tz = threadIdx.x % depth_tile_size_out_channels;
  const int depth_tx = threadIdx.x / depth_tile_size_out_channels / tile_size_width;
  const int depth_ty = (threadIdx.x / depth_tile_size_out_channels) % tile_size_width;

  #pragma unroll
    for(int i=0; i<num_iters; ++i) {
    #pragma unroll
    for(int kh=0; kh<kernel_height; ++kh){
      #pragma unroll
      for(int kw=0; kw<kernel_width; ++kw){    
        int idx_intermedia = (depth_tx+kh)*padded_tile_size_width*out_channels + (depth_ty+kw) * out_channels + i * depth_tile_size_out_channels + depth_tz;
        int idx_depth_weight = i*kernel_height*kernel_width*depth_tile_size_out_channels + kh*kernel_width*depth_tile_size_out_channels + kw*kernel_width + depth_tz;
        // int idx_depth_weight = kh*kernel_width*out_channels + kw*out_channels + i * depth_tile_size_out_channels + depth_tz;
        depthwise_local[i] += shared_intermedia_result[idx_intermedia] * shared_depthwise_weight[idx_depth_weight];
      }
    }
    output[(bx*tile_size_height+depth_tx) * width * out_channels + (by*tile_size_width + depth_ty) * out_channels + (i*depth_tile_size_out_channels+depth_tz)] = depthwise_local[i];
  }

  // Every two threads compute a channel, thus the weights are keep in registers and the features are also keep in registers
  // const int depth_tx = threadIdx.x / 128;
  // const int depth_tz = threadIdx.x % 128;
  // // Load weights to register
  // float weights_local[3][3];
  // #pragma unroll
  // for(int kh=0; kh<kernel_height; ++kh){
  //   #pragma unroll
  //   for(int kw=0; kw<kernel_width; ++kw){
  //     weights_local[kh][kw] = shared_depthwise_weight[kh*kernel_width*out_channels+kw*out_channels+depth_tz];
  //   }
  // }
  // float features_local[4][6];
  // #pragma unroll
  // for(int i=0; i<4; ++i){
  //   #pragma unroll
  //   for(int j=0; j<padded_tile_size_width; ++j){
  //     features_local[i][j] = shared_intermedia_result[(depth_tx*3-1+i)*padded_tile_size_width*out_channels + j*out_channels+depth_tz];
  //   }
  // }

  // float depth_local[2][4];
  // #pragma unroll
  // for(int i=0; i<8; ++i){
  //   depth_local[i/4][i%4]=0;
  // }

  // #pragma unroll
  // for(int i=0; i<2; ++i){
  //   #pragma unroll
  //   for(int j=0; j<4; ++j){
  //     #pragma unroll
  //     for(int kh=0; kh<kernel_height; ++kh){
  //       #pragma unroll
  //       for(int kw=0; kw<kernel_width; ++kw){
  //         depth_local[i][j] += features_local[(i+kw)*padded_tile_size_width][j+kw] * weights_local[kh][kw];
  //       }
  //     }
  //     output[(bx*tile_size_height+i)*width*out_channels + (by*tile_size_width+j)*out_channels+ depth_tz] = depth_local[i][j];
  //   }
  // }
}
