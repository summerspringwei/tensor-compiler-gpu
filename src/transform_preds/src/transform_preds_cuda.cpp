
#include "affine_transform_cuda.hpp"
#include <cuda_runtime.h>

void transform_preds(float* target_coords, float* coords, int batch, int n, Point2f center, Point2f scale, Point2f output_size){
    double trans[6];
    Point2f shift {0, 0};
    get_affine_transform(trans, center, scale, 0, output_size, shift);
    float transf[6];
    for(int i=0; i<6; ++i){
        transf[i] = (float)trans[i];
    }
    affine_transform(target_coords, coords, transf, batch, n, 1);
}

/**
 * @brief 
 * 
 * @param d_target_coords Affined coords that lies on GPU memory
 * @param d_coords src coords that lies on GPU memory
 * @param batch 
 * @param n 
 * @param center : On host memroy 
 * @param scale : On host memory
 * @param output_size : On host memory
 */
// void transform_preds_cuda(float* d_target_coords, float* d_coords, int batch, int n, Point2f center, Point2f scale, Point2f output_size){
//     double trans[6];
//     Point2f shift {0, 0};
//     get_affine_transform(trans, center, scale, 0, output_size, shift);
//     float transf[6];
//     for(int i=0; i<6; ++i){
//         transf[i] = (float)trans[i];
//     }
//     float* d_transf = (float*)cudaMalloc(sizeof(float) * 6);
//     cudaMemcpy(d_transf, transf, sizeof(float) * 6, cudaMemcpyDeviceToHost);
//     affine_transform_cuda(d_target_coords, d_coords, d_transf, batch, n);
// }
