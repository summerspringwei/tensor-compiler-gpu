
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


class BlockIdx {
  public:
  BlockIdx(): x(0), y(0), z(0){}
  BlockIdx(int a, int b, int c): x(a), y(b), z(c) {}
  int x, y, z;
};


const int img_height = 56, img_width = 56, img_in_channels = 4;
const int padded_height = 6, padded_width=16, padded_in_channels = img_in_channels;
void simmulate_test_pad(float* input, float* pointwise_PaddedInput_shared, BlockIdx blockIdx){
  const int block_tile_size_x = 14, block_tile_size_y = 4, block_tile_size_z = 9;
  const int num_block_x=4, num_block_y=14, num_block_z=14;
  const int thread_tile_size_x = 6, thread_tile_size_y=16;

  // Simulate threads in a block
  BlockIdx arr_threadIdx[128];
  for(int j=0; j<128; ++j){
    arr_threadIdx[j].x=j;
    auto threadIdx = arr_threadIdx[j];
    const int bx = blockIdx.x / (block_tile_size_y * block_tile_size_z); // range [0, 14)
    const int by = (blockIdx.x % (block_tile_size_y * block_tile_size_z)) / block_tile_size_z; // range [0, 4)
    const int bz = (blockIdx.x % block_tile_size_z);

    const int tx = threadIdx.x / thread_tile_size_y; // range [0, 6)
    const int ty = threadIdx.x % thread_tile_size_y; // range [0, 16)
    for(int i=0; i<img_in_channels; ++i){
      if(threadIdx.x >= thread_tile_size_x*thread_tile_size_y){
        continue;
      }
      if(bx == 0 && by == 0){/* top left corner OK */
        pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx == 0 || ty == 0) ? 
          0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }else if(bx == 0 && by == block_tile_size_y-1){/* top right corner OK */
        pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==0 || ty==thread_tile_size_y-1) ?
          0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y - 1 + ty) * img_in_channels + i];
      }else if(bx==block_tile_size_x-1 && by==0) {/* bottom right corner OK */ 
        pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==thread_tile_size_x-1 || ty==0) ?
          0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }else if(bx==block_tile_size_x-1 && by==block_tile_size_y-1){/* bottom right corner OK */ 
        pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==thread_tile_size_x-1 || ty == thread_tile_size_y-1) ?
          0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }else if(bx==0 && by>0 && by<block_tile_size_y-1){ /* top middle OK*/
        pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==0) ?
          0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }else if(bx==block_tile_size_x-1 && by>0 && by<block_tile_size_y-1){ /* bottom middle OK*/
        pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==thread_tile_size_x-1) ?
          0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }else if(bx>0 && bx<block_tile_size_x-1 && by==0){ /* left middle */
        pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (ty==0) ?
          0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }else if(bx>0 && bx<block_tile_size_x - 1 && by==block_tile_size_y-1){
        pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (ty==thread_tile_size_y-1) ?
          0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }else{
        pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = 
          input[(bx * block_tile_size_x + tx - 1 ) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }
  }
}

void init_input(float* input){
  for(int i=0; i<img_height; ++i){
    for(int j=0; j<img_width; ++j){
      for(int k=0; k<img_in_channels; ++k){
        input[i*img_width*img_in_channels + j* img_in_channels + k] = i*img_width + j;
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
  for(int i=0; i<padded_in_channels; ++i){
    for(int j=0; j<padded_height; ++j){
      for(int k=0; k<padded_width; ++k){
        printf("%.1f ", input[i*padded_height*padded_width + j * padded_width +k]);
      }printf("\n");
    }printf("\n");
  }printf("\n");
}

int main(int argc, char* argv[]){
  int idx = 0;
  if(argc>1){
    idx = atoi(argv[1]);
  }
  int input_size = img_height*img_width*img_in_channels;
  int pad_size =padded_height*padded_width*padded_in_channels;
  float * input = new float[input_size];
  float * pointwise_PaddedInput_shared = new float[input_size];
  init_input(input);
  // init_data(input, input_size, 1);
  init_data(pointwise_PaddedInput_shared, pad_size, 0);
  BlockIdx blockIdx(idx, 0, 0);
  simmulate_test_pad(input, pointwise_PaddedInput_shared, blockIdx);
  print_pad(pointwise_PaddedInput_shared);
  delete[] input;
  delete[] pointwise_PaddedInput_shared;
  return 0;
}
