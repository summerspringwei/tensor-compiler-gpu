


#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


__device__ __forceinline__ float sigmoid(float x){
    return (1.0f / (1+exp(-x)));
}

// num_layers: 10, timesteps: 100
// inputs_timestep: [1, 100, 128], outputs_timestep[1, 100, 128]
// input_wavefront: [1, 10, 128], state_wavefront: [1, 10, 128], weight_*_wavefront [40, 128, 128]
// c: [1, 10, 128], output_buffer:[1, 80, 128]
// blockDim(80, ), threadDim(128, ) 
const size_t num_layer=10, num_hidden=96, num_timestep=100, batch=1;
const int kNumGatesInLstmCell = 8;
__global__ void lstm_wavefront_magic(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    // Load weight to shared memory
    __shared__ float shared_weight[num_hidden*num_hidden];
    if(threadIdx.x >= num_hidden){
      return;
    }

    float * weight_ptr = NULL;
    float* is_ptr = NULL;
    if(blockIdx.x%2==0){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
    }else{
        weight_ptr=weight_state_wavefront;
        is_ptr = h_wavefront;
    }
    #pragma unroll
    for(int i=0; i<num_hidden; ++i){
      // if(i * num_hidden + threadIdx.x >= num_hidden * num_hidden){
      //   printf("%d\n", i * num_hidden + threadIdx.x);
      // }
      shared_weight[i * num_hidden + threadIdx.x] = weight_ptr[blockIdx.x/2 * num_hidden * num_hidden + i * num_hidden + threadIdx.x];
    }
    
    __syncthreads();
    
    for(int step=num_layer-1; step<num_timestep-(num_layer-1); ++step){
        // Compute gate, for even block compute input_gate*weight_input, for odd block compute h_gate*weight_state
        float thread_output = 0;
        for(int i=0; i<num_hidden; ++i){
          // TODO(xiachunwei) Check
            thread_output = thread_output + \
                is_ptr[blockIdx.x/kNumGatesInLstmCell * num_hidden + threadIdx.x] * \
                shared_weight[i * num_hidden + threadIdx.x];
        }
        
        // Save shared_output to global memory
        output_buffer[blockIdx.x * num_hidden + threadIdx.x] = thread_output;
        __threadfence();
        // Add input and state, output_buffer
        // Cross block, therefore we need a fence
        if(blockIdx.x%2==0){
            output_buffer[blockIdx.x * num_hidden + threadIdx.x] = output_buffer[(blockIdx.x+1) * num_hidden + threadIdx.x] + \
                output_buffer[blockIdx.x * num_hidden + threadIdx.x] + bias[threadIdx.x];
            // if(blockIdx.x==0){
            //   printf("step: %d, threadIdx.x: %d output_buffer:%f \n", step, threadIdx.x, output_buffer[blockIdx.x * num_hidden + threadIdx.x]);
            // }
        }// Now output_buffer: [1, 40, 128] with stride [0, 1, 0]
        // Cross block, therefore we need a fence
        __threadfence();
        if(blockIdx.x % kNumGatesInLstmCell==0){
            float* i = output_buffer + (blockIdx.x + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x + 2) * num_hidden;
            float* f = output_buffer + (blockIdx.x + 4) * num_hidden;
            float* o = output_buffer + (blockIdx.x + 6) * num_hidden;
            // Do state update
            // TODO(xiachunwei) check
            c_wavefront[blockIdx.x/kNumGatesInLstmCell*num_hidden + threadIdx.x] = c_wavefront[blockIdx.x/kNumGatesInLstmCell*num_hidden + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tan(j[threadIdx.x]);
            h_wavefront[blockIdx.x/kNumGatesInLstmCell*num_hidden + threadIdx.x] = tan(c_wavefront[blockIdx.x/kNumGatesInLstmCell*num_hidden + threadIdx.x]) * sigmoid(o[blockIdx.x/kNumGatesInLstmCell*num_hidden + threadIdx.x]);
            // if(blockIdx.x==0){
            //   printf("step: %d, threadIdx.x: %d c_wavefront:%f \n", step, threadIdx.x, c_wavefront[blockIdx.x * num_hidden + threadIdx.x]);
            //   printf("step: %d, threadIdx.x: %d h_wavefront:%f \n", step, threadIdx.x, h_wavefront[blockIdx.x * num_hidden + threadIdx.x]);
            // }
            // Feed inputs[step] to input_wavefront
            if(blockIdx.x==0){
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
            }// Shift h
            else if(blockIdx.x/kNumGatesInLstmCell < num_layer-1){
                input_wavefront[(blockIdx.x / kNumGatesInLstmCell + 1) * num_hidden + threadIdx.x] = h_wavefront[blockIdx.x/kNumGatesInLstmCell*num_hidden + threadIdx.x];
            }
            else if(blockIdx.x/kNumGatesInLstmCell == num_layer-1){
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[blockIdx.x/kNumGatesInLstmCell*num_hidden + threadIdx.x];
            }
        }
        __threadfence();
    }
}


// num_layers: 10, timesteps: 100
// inputs_timestep: [1, 100, 128], outputs_timestep[1, 100, 128]
// input_wavefront: [1, 10, 128], state_wavefront: [1, 10, 128], weight_*_wavefront [40, 128, 128]
// c: [1, 10, 128], output_buffer:[1, 80, 128]
// blockDim(80, ), threadDim(128, ) 
__global__ void lstm_wavefront_magic_v2(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    // Load weight to shared memory
    extern __shared__ float shared_weight[];

    float * weight_ptr = NULL;
    float* is_ptr = NULL;
    if(blockIdx.x%2==0){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
    }else{
        weight_ptr=weight_state_wavefront;
        is_ptr = h_wavefront;
    }
    for(int i=0; i<num_hidden; ++i){
      shared_weight[i * num_hidden + threadIdx.x] = weight_ptr[blockIdx.x * num_hidden * num_hidden + i * num_hidden + threadIdx.x];
    }
    float thread_output = 0;
    __syncthreads();
    
    for(int step=num_layer-1; step<num_timestep-(num_layer-1); ++step){
        // Compute gate, for even block compute input_gate*weight_input, for odd block compute h_gate*weight_state
        for(int i=0; i<num_hidden; ++i){
            thread_output = thread_output + \
                is_ptr[blockIdx.x/8 * num_hidden + threadIdx.x] * \
                shared_weight[i * num_hidden + threadIdx.x];
        }
        // Save thread_output to global memory
        output_buffer[blockIdx.x * num_hidden + threadIdx.x] = thread_output;
        __threadfence();// Cross block, therefore we need a fence
        // Add input and state, output_buffer
        if(blockIdx.x==0){
          for(int i=0; i<4*num_layer; ++i){
            output_buffer[(2 * i) * num_hidden + threadIdx.x] = output_buffer[(2*i) * num_hidden + threadIdx.x] + \
                output_buffer[(2*i+1) * num_hidden + threadIdx.x] + bias[threadIdx.x];
          }// Now output_buffer: [1, 40, 128] with stride [0, 1, 0]
          for(int k=0; k<num_layer; ++k){
            float* i = output_buffer + (2 * k + 0) * num_hidden;
            float* j = output_buffer + (2 * k + 2) * num_hidden;
            float* f = output_buffer + (2 * k + 4) * num_hidden;
            float* o = output_buffer + (2 * k + 6) * num_hidden;
            // Do state update
            c_wavefront[k*num_hidden + threadIdx.x] = c_wavefront[k*num_hidden + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tan(j[threadIdx.x]);
            h_wavefront[k*num_hidden + threadIdx.x] = tan(c_wavefront[k*num_hidden + threadIdx.x]) * sigmoid(o[k*num_hidden + threadIdx.x]);
          }
          // Feed inputs[step] to input_wavefront
          input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
          // Shift h
          for(int k=0; k<num_layer; ++k){
            input_wavefront[(k + 1) * num_hidden + threadIdx.x] = h_wavefront[k/8*num_hidden + threadIdx.x];
          }
          outputs_timestep[step*num_hidden + threadIdx.x]=h_wavefront[9*num_hidden + threadIdx.x];
        }
        __threadfence();
    }
}




// build
// export CUDA_VISIBLE_DEVICES=0 && nvcc ./lstm_wavefront_magic.cu -gencode arch=compute_80,code=compute_80
