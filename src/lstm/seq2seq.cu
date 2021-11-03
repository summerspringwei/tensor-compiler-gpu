
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "cuda_runtime.h"

__device__ __forceinline__ float sigmoid(float x){
    return (1.0f / (1+exp(-x)));
}

// num_layers: 8, timesteps: 100
// inputs_timestep: [1, 100, 128], outputs_timestep[1, 100, 128]
// input_wavefront: [1, 8, 128], state_wavefront: [1, 8, 128], weight_*_wavefront [32, 128, 128]
// c: [1, 8, 128], output_buffer:[1, 8, 128]
// two block computes one gate, therefore each block compute [1, 1, 128] * [1, 128, 64]
// gridDim(64*2, ), blockDim(64, ) 
// template<int batch, int num_layer, int num_hidden, int num_timestep>
extern "C" __global__ void seq2seq_encoder(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    const size_t batch=1, num_layer=8, num_timestep=100, num_hidden=128;
    const int kNumGatePart = 2;
    const int kNumGatesInLstmCell = 8;
    const int kHalfHidden = num_hidden / kNumGatePart;
    
    __shared__ float shared_weight[num_hidden * kHalfHidden];
    float * weight_ptr = NULL;
    float* is_ptr = NULL;
    // Load weight to shared memory
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
        // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
        for(int i=0; i<num_hidden; ++i){
            shared_weight[i * kHalfHidden + threadIdx.x] = weight_ptr[blockIdx.x / 2 * num_hidden * num_hidden + i * num_hidden + (blockIdx.x % 2) * kHalfHidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        is_ptr = h_wavefront;
        for(int i=0; i<num_hidden; ++i){
            shared_weight[i * kHalfHidden + threadIdx.x] = weight_ptr[(blockIdx.x-gridDim.x/2) / 2 * num_hidden * num_hidden + i * num_hidden + (blockIdx.x % 2) * kHalfHidden + threadIdx.x];
        }
    }
    
    // if(blockIdx.x==0 && threadIdx.x == 0){
    //     for(int i=0; i<num_hidden; ++i){
    //         printf("%f ",shared_weight[i * kHalfHidden + threadIdx.x]);
    //     }printf("\n");
    // }
    // if((blockIdx.x==0 && threadIdx.x == 0) || (blockIdx.x==64 && threadIdx.x == 0)){
    //     printf("<%d, %d>  is_ptr %f\n", blockIdx.x, threadIdx.x, is_ptr[0]);
    // }
    __syncthreads();

    for(int step=num_layer-1; step<num_timestep-(num_layer-1); ++step){
        // Compute gate, for even block compute first half gate, for odd block compute second half gate
        float thread_output = 0;
        if(blockIdx.x < gridDim.x / 2){
            for(int i=0; i<num_hidden; ++i){
                thread_output = thread_output + \
                    is_ptr[blockIdx.x / kNumGatePart * num_hidden + i] * \
                        shared_weight[i * kHalfHidden + threadIdx.x];
            }
        }else{
            for(int i=0; i<num_hidden; ++i){
                thread_output = thread_output + \
                    is_ptr[(blockIdx.x-gridDim.x/2) / kNumGatePart * num_hidden + i] * \
                        shared_weight[i * kHalfHidden + threadIdx.x];
            }
        }
        // save thread_output to global memory
        output_buffer[blockIdx.x * kHalfHidden + threadIdx.x] = thread_output;
        __threadfence();
        
        // Let first 64 block compute input_gate+hidden_gate+bias
        if(blockIdx.x < gridDim.x / 2){
            output_buffer[blockIdx.x * kHalfHidden + threadIdx.x] = output_buffer[blockIdx.x * kHalfHidden + threadIdx.x] + \
                output_buffer[(blockIdx.x + gridDim.x / 2) * kHalfHidden + threadIdx.x] + bias[(blockIdx.x % 2) * kHalfHidden + threadIdx.x];
        }
        __threadfence();
        // Each block compute a cell thus we need num_layer blocks
        if(blockIdx.x < gridDim.x/2 && blockIdx.x % kNumGatesInLstmCell==0){
            float* i = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 3) * num_hidden;
            c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x] = 
                c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tan(j[threadIdx.x]);
            c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kHalfHidden + threadIdx.x] = 
                c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kHalfHidden + threadIdx.x] * sigmoid(f[kHalfHidden + threadIdx.x] + 1.0) +
                sigmoid(i[kHalfHidden + threadIdx.x]) * tan(j[kHalfHidden + threadIdx.x]);
            h_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x] = 
                tan(c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            h_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kHalfHidden + threadIdx.x] = 
                tan(c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kHalfHidden + threadIdx.x]) * sigmoid(o[kHalfHidden + threadIdx.x]);
            // if(blockIdx.x==0 && threadIdx.x == 0){
            //     printf("step: %d c_wavefront: %f \n", step, c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x]);
            //     printf("step: %d h_wavefront: %f \n", step, h_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x]);
            // }
            // Feed inputs[step] to input_wavefront
            if(blockIdx.x==0){
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
                input_wavefront[0*num_hidden + kHalfHidden + threadIdx.x] = inputs_timestep[step*num_hidden + kHalfHidden + threadIdx.x];
            }// Shift h
            else if(blockIdx.x / kNumGatesInLstmCell < num_layer - 1){
                input_wavefront[(blockIdx.x / kNumGatesInLstmCell + 1) * num_hidden + threadIdx.x] = h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + threadIdx.x];
                input_wavefront[(blockIdx.x / kNumGatesInLstmCell + 1) * num_hidden + kHalfHidden + threadIdx.x] = h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + kHalfHidden + threadIdx.x];
            }else if(blockIdx.x / kNumGatesInLstmCell == num_layer - 1){
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + threadIdx.x];
                outputs_timestep[(step+1-num_layer)*num_hidden + kHalfHidden + threadIdx.x]=h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + kHalfHidden + threadIdx.x];
            }
        }
        __threadfence();
    }
}
