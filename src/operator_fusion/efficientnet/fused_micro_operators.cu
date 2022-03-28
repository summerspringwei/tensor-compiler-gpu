

#define UPDIV(x, y) (((x)%(y))==0? ((x)/(y)): (((x)/(y))+1))

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__inline__ __device__
float blockReduceSum(float val) {
  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  val = warpReduceSum(val);     // Each warp performs partial reduction
  if (lane==0) shared[wid]=val; // Write reduced value to shared memory
  __syncthreads();              // Wait for all partial reductions
  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp
  return val;
}

__device__ float sigmoid(float x){
  float e_x = __expf(x);
  return e_x / (1+e_x);
}

const int in_channels = 480, height = 14, width=14;
const int out_channels_1 = 20, out_channels_2 = 480;

const int block_size = 256;

// Layout: input: NHWC, weight1: COCI, bias1: CO, weigh2: CICO, bias2: CO
extern "C" __global__ void __launch_bounds__(256) fused_micro_operators(
  float* input, float* weight1, float* bias1, float* weight2, float* bias2, float* output){
  __shared__ float shared_output1[out_channels_1];
  const int reduce_num_iters = UPDIV(in_channels, block_size);
  float reduce_local[reduce_num_iters];

  // ReduceMean: Reduce multiple elements per-thread
  // Estimated num cycles: height*width*reduce_num_iters*time_load_from_gm
  #pragma unroll
  for(int i=0; i<reduce_num_iters; ++i){
    int idx = i * blockDim.x + threadIdx.x;
    if(idx >= in_channels){
      continue;
    }
    reduce_local[i] = 0;
    #pragma unroll
    for(int rk = 0; rk<height*width; ++rk){
      reduce_local[i] += input[rk*in_channels + idx];
    }
    reduce_local[i] = reduce_local[i] / (height*width);
  }
  // dbg_shared_reduce_local[threadIdx.x] = reduce_local[0];
  // Fused_matmul_biasAdd_sigmoid_mul Do the first vector*matrix
  // Estimated cycles: out_channels_1 * reduce_num_iters * (256/warpSize) * 8
  #pragma unroll
  for(int i=0; i<out_channels_1; ++i){
    float sum = 0;
    #pragma unroll
    for(int j=0; j<reduce_num_iters; ++j){
      int idx = j * blockDim.x + threadIdx.x;
      if(idx < in_channels){
        sum += reduce_local[j] * weight1[i*in_channels+idx];
      }
    }
    __syncthreads(); 
    // Fused BiasAdd, Sigmoid, mul
    float reduce_one_channel = blockReduceSum(sum);
    float bias_add1_output = reduce_one_channel + bias1[i];
    float mul1_output = bias_add1_output * sigmoid(bias_add1_output);
    shared_output1[i] = mul1_output;
  }
  __syncthreads();

  // Fused_matmul_biasAdd_mul: Do the second vec*matrix, the reduce dimension is much small
  // Estimated number of cycles: out_channels_1 * output2_num_iters
  const int output2_num_iters = UPDIV(out_channels_2, block_size);
  float output2_local[output2_num_iters];
  #pragma unroll
  for(int i=0; i<output2_num_iters; ++i){
    output2_local[i] = 0;
  }
  
  #pragma unroll
  for(int i=0; i<output2_num_iters; ++i){
    int idx = i * blockDim.x + threadIdx.x;
    if(idx >= out_channels_2){
      continue;
    }
    #pragma unroll
    for(int rk=0; rk<out_channels_1; ++rk){
      output2_local[i] += shared_output1[rk]*weight2[rk*out_channels_2 + idx];
    }
    output[idx] = sigmoid(output2_local[i] + bias2[idx]);
  }
}


