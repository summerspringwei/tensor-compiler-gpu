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
    const int kPartHidden = num_hidden / kNumGatePart;
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kPartHidden * sizeof(float) < 48 * 1024);
    if(blockIdx.x >= num_layer*8*kNumGatePart){
        return;
    }
    // __shared__ float shared_weight[num_hidden * kPartHidden];
    extern float __shared__ shared_weight[];
    float * weight_ptr = NULL;
    float* is_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < 2){
        input_wavefront[blockIdx.x * kPartHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kPartHidden + threadIdx.x];
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<num_hidden; ++i){
            shared_weight[i * kPartHidden + threadIdx.x] = weight_ptr[blockIdx.x / 2 * num_hidden * num_hidden + i * num_hidden + (blockIdx.x % 2) * kPartHidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<num_hidden; ++i){
            shared_weight[i * kPartHidden + threadIdx.x] = weight_ptr[(blockIdx.x-gridDim.x/2) / 2 * num_hidden * num_hidden + i * num_hidden + (blockIdx.x % 2) * kPartHidden + threadIdx.x];
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
                        shared_weight[i * kPartHidden + threadIdx.x];
            }
        }else if(cond_state){
            #pragma unroll
            for(int i=0; i<num_hidden; ++i){
                thread_output = thread_output + \
                    is_ptr[(blockIdx.x-gridDim.x/2) / kNumGatePart * num_hidden + i] * \
                        shared_weight[i * kPartHidden + threadIdx.x];
            }
        }
        // TODO(xiachunwei) Can be replaced by atomicAdd?
        // Store thread_output to global memory
        output_buffer[blockIdx.x * kPartHidden + threadIdx.x] = thread_output;
        __threadfence();
        grid.sync();
        
        // Each block compute a cell thus we need num_layer blocks
        if(cond_input && blockIdx.x % kNumGatesInLstmCell==0){// TODO(xiachunwei) Processing last part for wavefront
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            #pragma unroll
            for(int i=0; i<kNumGatesInLstmCell; ++i){
                output_buffer[(blockIdx.x + i) * kPartHidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x+i) + gridDim.x / 2) * kPartHidden + threadIdx.x] + bias[((blockIdx.x+i) % 2) * kPartHidden + threadIdx.x];
            }

            float* i = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 3) * num_hidden;
            c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x] = 
                c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tan(j[threadIdx.x]);
            c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kPartHidden + threadIdx.x] = 
                c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kPartHidden + threadIdx.x] * sigmoid(f[kPartHidden + threadIdx.x] + 1.0) +
                sigmoid(i[kPartHidden + threadIdx.x]) * tan(j[kPartHidden + threadIdx.x]);
            h_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x] = 
                tan(c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            h_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kPartHidden + threadIdx.x] = 
                tan(c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kPartHidden + threadIdx.x]) * sigmoid(o[kPartHidden + threadIdx.x]);
            
            if(blockIdx.x / kNumGatesInLstmCell < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / kNumGatesInLstmCell + 1) * num_hidden + threadIdx.x] = h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + threadIdx.x];
                input_wavefront[(blockIdx.x / kNumGatesInLstmCell + 1) * num_hidden + kPartHidden + threadIdx.x] = h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + kPartHidden + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
                input_wavefront[0*num_hidden + kPartHidden + threadIdx.x] = inputs_timestep[step*num_hidden + kPartHidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / kNumGatesInLstmCell == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + threadIdx.x];
                outputs_timestep[(step+1-num_layer)*num_hidden + kPartHidden + threadIdx.x]=h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + kPartHidden + threadIdx.x];
            }
        }
    }
    // __threadfence();
    grid.sync();
}


// Each block compute num_hidden/kNumGatePart fma then all blocks sync
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void lstm_reuse_shared_memory_v2(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    
    const int kNumInputGate = 4;
    const int kNumGatesInLstmCell = 8;
    const int kNumGatePart = gridDim.x / (num_layer*kNumGatesInLstmCell);
    const int kPartHidden = num_hidden / kNumGatePart;

    assert(gridDim.x % (num_layer*kNumGatePart)==0);
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kPartHidden * sizeof(float) < 48 * 1024);
    if(blockIdx.x >= num_layer*kNumGatesInLstmCell*kNumGatePart){
        return;
    }
    
    extern float __shared__ shared_weight[];
    float * weight_ptr = NULL;
    float* is_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < kNumGatePart){
        input_wavefront[blockIdx.x * kPartHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kPartHidden + threadIdx.x];
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<kPartHidden; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[blockIdx.x / kNumGatePart * num_hidden * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<kPartHidden; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x - gridDim.x / 2) %  kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }
    output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        float thread_output = 0;
        // Blocks of first half gridDim.x compute the input gate
        bool cond_input = ((step<num_timestep && blockIdx.x < min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart) 
            || (step>=num_timestep && blockIdx.x >= (step+1-num_timestep) * kNumInputGate * kNumGatePart)) && (blockIdx.x < gridDim.x / 2);
        // Blocks of second half gridDim.x compute the input gate
        bool cond_state = (((blockIdx.x >= gridDim.x / 2) && (blockIdx.x < gridDim.x / 2 + min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart))
            || (step>=num_timestep && (blockIdx.x >= (gridDim.x / 2 + (step+1-num_timestep) * kNumInputGate * kNumGatePart)))) && (blockIdx.x >= gridDim.x / 2);
        
        // GEMV
        if(cond_input){
            #pragma unroll
            for(int i=0; i<kPartHidden; ++i){
                thread_output += is_ptr[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + threadIdx.x] * shared_weight[i*num_hidden + threadIdx.x];
            }
        }else if(cond_state){
            #pragma unroll
            for(int i=0; i<kPartHidden; ++i){
                thread_output += is_ptr[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x] * shared_weight[i*num_hidden + threadIdx.x];
            }
        }
        
        if(cond_input || cond_state){
            // Store thread_output to global memory
            atomicAdd(output_buffer + ((blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x), thread_output);
            __syncthreads();
        }
        
        grid.sync();
        
        // Each block compute a cell thus we need num_layer blocks
        //(step>=num_timestep && blockIdx.x==gridDim.x-(num_layer + num_timestep - step) * kNumInputGate))
        if(cond_input && (blockIdx.x % (kNumInputGate * kNumGatePart)==0 || false)){// TODO(xiachunwei) Processing last part for wavefront
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                output_buffer[(blockIdx.x / kNumGatePart + i) * num_hidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x + gridDim.x / 2) / kNumGatePart + i) * num_hidden + threadIdx.x] + bias[(blockIdx.x / (kNumInputGate * kNumGatePart)) + threadIdx.x];
            }

            float* i = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 3) * num_hidden;
            c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tan(j[threadIdx.x]);
            
            h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                tan(c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            
            if(blockIdx.x / (kNumInputGate * kNumGatePart) < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden + threadIdx.x] = h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / (kNumInputGate * kNumGatePart) == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x];
            }
            // if(step>= num_timestep && threadIdx.x==0){
            //     printf("step: %d\n", step);
            //     printf("blockIdx.x %d\n", blockIdx.x);
            // }
        }
        output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
        __threadfence();
        grid.sync();
    }
}


// Each block compute num_hidden/kNumGatePart fma then all blocks sync
// Using atomicAdd to build spin lock to sync between blocks
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void lstm_reuse_shared_memory_v3(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer, float* arr_sync){
    
    const int kNumInputGate = 4;
    const int kNumGatesInLstmCell = 8;
    const int kNumGatePart = gridDim.x / (num_layer*kNumGatesInLstmCell);
    const int kPartHidden = num_hidden / kNumGatePart;

    assert(gridDim.x % (num_layer*kNumGatePart)==0);
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kPartHidden * sizeof(float) < 48 * 1024);
    if(blockIdx.x >= num_layer*kNumGatesInLstmCell*kNumGatePart){
        return;
    }
    
    extern float __shared__ shared_weight[];
    float * weight_ptr = NULL;
    float* is_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < kNumGatePart){
        input_wavefront[blockIdx.x * kPartHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kPartHidden + threadIdx.x];
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<kPartHidden; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[blockIdx.x / kNumGatePart * num_hidden * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<kPartHidden; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x - gridDim.x / 2) %  kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }
    output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
    __syncthreads();
    
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        float thread_output = 0;
        // Blocks of first half gridDim.x compute the input gate
        bool cond_input = ((step<num_timestep && blockIdx.x < min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart) 
            || (step>=num_timestep && blockIdx.x >= (step+1-num_timestep) * kNumInputGate * kNumGatePart)) && (blockIdx.x < gridDim.x / 2);
        // Blocks of second half gridDim.x compute the input gate
        bool cond_state = (((blockIdx.x >= gridDim.x / 2) && (blockIdx.x < gridDim.x / 2 + min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart))
            || (step>=num_timestep && (blockIdx.x >= (gridDim.x / 2 + (step+1-num_timestep) * kNumInputGate * kNumGatePart)))) && (blockIdx.x >= gridDim.x / 2);
        
        // GEMV
        if(cond_input){
            #pragma unroll
            for(int i=0; i<kPartHidden; ++i){
                thread_output += is_ptr[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + threadIdx.x] * shared_weight[i*num_hidden + threadIdx.x];
            }
        }else if(cond_state){
            #pragma unroll
            for(int i=0; i<kPartHidden; ++i){
                thread_output += is_ptr[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x] * shared_weight[i*num_hidden + threadIdx.x];
            }
        }
        
        if(cond_input || cond_state){
            // Store thread_output to global memory
            atomicAdd(output_buffer + ((blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x), thread_output);
            __threadfence();
            atomicAdd(arr_sync+(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x, 1);
            while(arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] != kNumGatePart){
                // Spin here
            }
        }
        
        // grid.sync();
        
        // Each block compute a cell thus we need num_layer blocks
        //(step>=num_timestep && blockIdx.x==gridDim.x-(num_layer + num_timestep - step) * kNumInputGate))
        if(cond_input && (blockIdx.x % (kNumInputGate * kNumGatePart)==0 || false)){// TODO(xiachunwei) Processing last part for wavefront
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                output_buffer[(blockIdx.x / kNumGatePart + i) * num_hidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x + gridDim.x / 2) / kNumGatePart + i) * num_hidden + threadIdx.x] + bias[(blockIdx.x / (kNumInputGate * kNumGatePart)) + threadIdx.x];
            }

            float* i = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 3) * num_hidden;
            c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tan(j[threadIdx.x]);
            
            h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                tan(c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            
            if(blockIdx.x / (kNumInputGate * kNumGatePart) < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden + threadIdx.x] = h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / (kNumInputGate * kNumGatePart) == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x];
            }
            // if(step>= num_timestep && threadIdx.x==0){
            //     printf("step: %d\n", step);
            //     printf("blockIdx.x %d\n", blockIdx.x);
            // }
        }
        output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
        arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
        __threadfence();
        grid.sync();
    }
}
