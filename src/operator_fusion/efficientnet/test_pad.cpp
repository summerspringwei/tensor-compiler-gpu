
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


class BlockIdx {
  public:
  BlockIdx(): x(0), y(0), z(0){}
  BlockIdx(int a, int b, int c): x(a), y(b), z(c) {}
  int x, y, z;
};


// const int img_height = 56, img_width = 56, img_in_channels = 4;
// const int padded_tile_size_height = 6, padded_tile_size_width=16, padded_in_channels = img_in_channels;
// void simmulate_test_pad(float* input, float* pointwise_PaddedInput_shared, BlockIdx blockIdx){
//   const int block_tile_size_x = 14, block_tile_size_y = 4, block_tile_size_z = 9;
//   const int num_block_x=4, num_block_y=14, num_block_z=14;
//   const int thread_tile_size_x = 6, thread_tile_size_y=16;

//   // Simulate threads in a block
//   BlockIdx arr_threadIdx[128];
//   for(int j=0; j<128; ++j){
//     arr_threadIdx[j].x=j;
//     auto threadIdx = arr_threadIdx[j];
//     const int bx = blockIdx.x / (block_tile_size_y * block_tile_size_z); // range [0, 14)
//     const int by = (blockIdx.x % (block_tile_size_y * block_tile_size_z)) / block_tile_size_z; // range [0, 4)
//     const int bz = (blockIdx.x % block_tile_size_z);

//     const int tx = threadIdx.x / thread_tile_size_y; // range [0, 6)
//     const int ty = threadIdx.x % thread_tile_size_y; // range [0, 16)
//     for(int i=0; i<img_in_channels; ++i){
//       if(threadIdx.x >= thread_tile_size_x*thread_tile_size_y){
//         continue;
//       }
//       if(bx == 0 && by == 0){/* top left corner OK */
//         pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx == 0 || ty == 0) ? 
//           0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
//       }else if(bx == 0 && by == block_tile_size_y-1){/* top right corner OK */
//         pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==0 || ty==thread_tile_size_y-1) ?
//           0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y - 1 + ty) * img_in_channels + i];
//       }else if(bx==block_tile_size_x-1 && by==0) {/* bottom right corner OK */ 
//         pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==thread_tile_size_x-1 || ty==0) ?
//           0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
//       }else if(bx==block_tile_size_x-1 && by==block_tile_size_y-1){/* bottom right corner OK */ 
//         pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==thread_tile_size_x-1 || ty == thread_tile_size_y-1) ?
//           0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
//       }else if(bx==0 && by>0 && by<block_tile_size_y-1){ /* top middle OK*/
//         pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==0) ?
//           0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
//       }else if(bx==block_tile_size_x-1 && by>0 && by<block_tile_size_y-1){ /* bottom middle OK*/
//         pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==thread_tile_size_x-1) ?
//           0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
//       }else if(bx>0 && bx<block_tile_size_x-1 && by==0){ /* left middle */
//         pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (ty==0) ?
//           0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
//       }else if(bx>0 && bx<block_tile_size_x - 1 && by==block_tile_size_y-1){
//         pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (ty==thread_tile_size_y-1) ?
//           0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
//       }else{
//         pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = 
//           input[(bx * block_tile_size_x + tx - 1 ) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
//       }
//     }
//   }
// }

const int height = 56, width = 56, in_channels = 24, out_channels = 144, kernel_width=3, kernel_height=3;
const int tile_size_height = 4, tile_size_width=4, tile_size_out_channels = 144;
const int num_tile_height = (height / tile_size_height), num_tile_width = (width / tile_size_width), num_tile_out_channels = (out_channels / tile_size_out_channels);
const int num_blocks =  num_tile_height * num_tile_width * num_tile_out_channels;
const int padded_tile_size_height = (tile_size_height+2), padded_tile_size_width = (tile_size_width+2);

void simmulate_test_pad(float* input, float* shared_input, BlockIdx blockIdx, int num_threads){
  BlockIdx arr_threadIdx[num_threads];
  for(int j=0; j<num_threads; ++j){
    arr_threadIdx[j].x=j;
    auto threadIdx = arr_threadIdx[j];
    const int bx = blockIdx.x / (num_tile_width * num_tile_out_channels);
    const int by = (blockIdx.x % (num_tile_width * num_tile_out_channels));
    // const int bz = blockIdx.x % num_tile_out_channels;
    const int load_tz = threadIdx.x / (padded_tile_size_height*padded_tile_size_width);
    const int load_tx = (threadIdx.x % (padded_tile_size_height*padded_tile_size_width)) / padded_tile_size_width;
    const int load_ty = threadIdx.x % padded_tile_size_width;
    // TODO(Chunwei Xia) Here the `in_channels` should divide the num_tile_out_channels
    const int load_shared_tile_z = (num_threads / (padded_tile_size_height*padded_tile_size_width)); // 256 / (6*6) == 7
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
  }

}


void add(float* a, float* b, float* c){
  for(int i=0;i<10; ++i){
    c[i] = a[i] + b[i];
  }
}

void init_input(float* input){
  for(int i=0; i<height; ++i){
    for(int j=0; j<width; ++j){
      for(int k=0; k<in_channels; ++k){
        input[i*width*in_channels + j* in_channels + k] = i*width + j;
      }
    }
  }
}

void init_data(float* input, int arr_size, float value){
  for(int i=0;i<arr_size;++i){
    input[i] = value;
  }
}

void print_pad(float* input){
  for(int i=0; i<in_channels; ++i){
    printf("channel %d\n", i);
    for(int j=0; j<padded_tile_size_height; ++j){
      for(int k=0; k<padded_tile_size_width; ++k){
        printf("%.1f ", input[i*padded_tile_size_height*padded_tile_size_width + j * padded_tile_size_width +k]);
      }printf("\n");
    }printf("\n");
  }printf("\n");
}

int main(int argc, char* argv[]){
  int idx = 0;
  if(argc>1){
    idx = atoi(argv[1]);
  }
  int input_size = height*width*in_channels;
  int pad_size =padded_tile_size_height*padded_tile_size_width*in_channels;
  float * input = new float[input_size];
  float * pointwise_PaddedInput_shared = new float[pad_size];
  init_input(input);
  // init_data(input, input_size, 1);
  init_data(pointwise_PaddedInput_shared, pad_size, 0);
  BlockIdx blockIdx(idx, 0, 0);
  simmulate_test_pad(input, pointwise_PaddedInput_shared, blockIdx, 256);
  print_pad(pointwise_PaddedInput_shared);
  delete[] input;
  delete[] pointwise_PaddedInput_shared;
  return 0;
}
