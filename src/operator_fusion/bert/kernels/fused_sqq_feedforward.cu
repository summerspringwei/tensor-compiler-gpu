
#include "bert.h"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>

using namespace fuselage::experiments::networks::bert;

__global__ void fused_sqq_feedforward(half* __restrict__ attn_fc_output,
                                half* __restrict__ feed_forward_fc1_weight,
                                half* __restrict__ feed_forward_fc1_output,
                                half* __restrict__ feed_forward_fc2_weight,
                                half* __restrict__ feed_forward_fc2_output,
                                half eps, half h_gama, half h_beta,
                                float * query_key_softmax_sum,
                                float* layer_norm_variance,
                                int64_t* profile_grid_clock){

}