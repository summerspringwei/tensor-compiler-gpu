#include <cassert>

#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>

__device__ __forceinline__ float sigmoid(float x){
    return (1.0f / (1+exp(-x)));
}

// TODO(xiachunwei) using more threads per block to do GEMV
// num_layers: 8, timesteps: 100
// inputs_timestep: [1, 100, 128], outputs_timestep[1, 100, 128]
// input_wavefront: [1, 8, 128], state_wavefront: [1, 8, 128], weight_*_wavefront [32, 128, 128]
// c: [1, 8, 128], output_buffer:[1, 8, 128]
// two block computes one gate, therefore each block compute [1, 1, 128] * [1, 128, 64]
// gridDim(64*2, ), blockDim(64, ) 
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void lstm_reuse_shared_memory(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    const int kNumGatePart = 2;
    // const int kNumInputGatesInLstmCell = 4;
    const int kNumGatesInLstmCell = 8;
    const int kHalfHidden = num_hidden / kNumGatePart;
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kHalfHidden * sizeof(float) < 48 * 1024);
    if(blockIdx.x >= num_layer*8*kNumGatePart){
        return;
    }
    // __shared__ float shared_weight[num_hidden * kHalfHidden];
    extern float __shared__ shared_weight[];
    float * weight_ptr = NULL;
    float* is_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < 2){
        input_wavefront[blockIdx.x * kHalfHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kHalfHidden + threadIdx.x];
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<num_hidden; ++i){
            shared_weight[i * kHalfHidden + threadIdx.x] = weight_ptr[blockIdx.x / 2 * num_hidden * num_hidden + i * num_hidden + (blockIdx.x % 2) * kHalfHidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<num_hidden; ++i){
            shared_weight[i * kHalfHidden + threadIdx.x] = weight_ptr[(blockIdx.x-gridDim.x/2) / 2 * num_hidden * num_hidden + i * num_hidden + (blockIdx.x % 2) * kHalfHidden + threadIdx.x];
        }
    }
    
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        // GEMV gate, for even blocks compute first half gate, for odd blocks compute second half gate
        float thread_output = 0;
        // Blocks from 0 to kNumGatePart * kNumInputGatesInLstmCell * num_layer compute the input gate
        bool cond_input = (blockIdx.x < min(step+1, (int)num_layer) * kNumGatesInLstmCell) 
            || (step>=num_timestep && blockIdx.x > (step+1-num_timestep)* kNumGatesInLstmCell);
        // Blocks from kNumGatePart * kNumInputGatesInLstmCell * num_layer to gridDim.x compute the state gate
        bool cond_state = ((blockIdx.x >= gridDim.x / 2) && (blockIdx.x < gridDim.x / 2 + min(step+1, (int)num_layer) * kNumGatesInLstmCell))
            || (step>=num_timestep && (blockIdx.x >= (gridDim.x / 2 + (step+1-num_timestep)* kNumGatesInLstmCell)));
        // GEMV
        if(cond_input){
            #pragma unroll
            for(int i=0; i<num_hidden; ++i){
                thread_output = thread_output + \
                    is_ptr[blockIdx.x / kNumGatePart * num_hidden + i] * \
                        shared_weight[i * kHalfHidden + threadIdx.x];
            }
        }else if(cond_state){
            #pragma unroll
            for(int i=0; i<num_hidden; ++i){
                thread_output = thread_output + \
                    is_ptr[(blockIdx.x-gridDim.x/2) / kNumGatePart * num_hidden + i] * \
                        shared_weight[i * kHalfHidden + threadIdx.x];
            }
        }
        // TODO(xiachunwei) Can be replaced by atomicAdd?
        // Store thread_output to global memory
        output_buffer[blockIdx.x * kHalfHidden + threadIdx.x] = thread_output;
        __threadfence();
        grid.sync();
        
        // Each block compute a cell thus we need num_layer blocks
        if(cond_input && blockIdx.x % kNumGatesInLstmCell==0){// TODO(xiachunwei) Processing last part for wavefront
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            #pragma unroll
            for(int i=0; i<kNumGatesInLstmCell; ++i){
                output_buffer[(blockIdx.x + i) * kHalfHidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x+i) + gridDim.x / 2) * kHalfHidden + threadIdx.x] + bias[((blockIdx.x+i) % 2) * kHalfHidden + threadIdx.x];
            }

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
            
            if(blockIdx.x / kNumGatesInLstmCell < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / kNumGatesInLstmCell + 1) * num_hidden + threadIdx.x] = h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + threadIdx.x];
                input_wavefront[(blockIdx.x / kNumGatesInLstmCell + 1) * num_hidden + kHalfHidden + threadIdx.x] = h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + kHalfHidden + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
                input_wavefront[0*num_hidden + kHalfHidden + threadIdx.x] = inputs_timestep[step*num_hidden + kHalfHidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / kNumGatesInLstmCell == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + threadIdx.x];
                outputs_timestep[(step+1-num_layer)*num_hidden + kHalfHidden + threadIdx.x]=h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + kHalfHidden + threadIdx.x];
            }
        }
    }
    // __threadfence();
    grid.sync();
}
