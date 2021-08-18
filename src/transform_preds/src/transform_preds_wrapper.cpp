
#include "transform_preds_wrapper.hpp"
#include "affine_transform.hpp"

at::Tensor transform_preds_cuda_forward(const at::Tensor &dets,
                    const at::Tensor &center,
                    const at::Tensor &scale,
                    const at::Tensor &output_size,
                    int num_classes){
    // dets shape (batch, 100, 6)
    // Initially dets is in NVIDIA GPU, actually we do not need to copy to CPU
    auto cpu_dets = dets.to(torch::kCPU);
    int batch = dets.size(0);
    int n = dets.size(1);
    int last_dim = dets.size(2);
    auto f_dets = cpu_dets.to(torch::kFloat32);
    float* coords = (float*)malloc(sizeof(float)*n*2);
    float* target_coords = (float*)malloc(sizeof(float)*n*2);
    using scalar_t = float;
    // copy dets Slice[:, :, 0:2]
    for(int i=0; i<batch; ++i){
        for(int j=0; j<n; ++j){
            coords[2*(i*n + j)] = (f_dets.data_ptr<scalar_t>())[((i*n) + j) * last_dim];
            coords[2*(i*n + j) + 1] = (f_dets.data_ptr<scalar_t>())[((i*n) + j) * last_dim + 1];
        }
    }
    auto f_center = center.data_ptr<scalar_t>();
    auto f_scale = scale.data_ptr<scalar_t>();
    auto f_output_size = output_size.data_ptr<scalar_t>();
    transform_preds(target_coords, coords, batch, n, 
        {f_center[0], f_center[1]}, {f_scale[0], f_scale[1]}, {f_output_size[0], f_output_size[1]});
    // copy dets [:, :0:2] slice back
    for(int i=0; i<batch; ++i){
        for(int j=0; j<n; ++j){
            (f_dets.data_ptr<scalar_t>())[((i*n) + j) * last_dim] = target_coords[2*(i*n + j)];
            (f_dets.data_ptr<scalar_t>())[((i*n) + j) * last_dim + 1] = target_coords[2*(i*n + j) + 1];
        } 
   }
    // copy dets Slice[:, :, 2:4]
    for(int i=0; i<batch; ++i){
        for(int j=0; j<n; ++j){
            coords[2*(i*n + j)] = (f_dets.data_ptr<scalar_t>())[((i*n) + j) * last_dim + 2];
            coords[2*(i*n + j) + 1] = (f_dets.data_ptr<scalar_t>())[((i*n) + j) * last_dim + 3];
        }
    }
    transform_preds(target_coords, coords, batch, n, 
        {f_center[0], f_center[1]}, {f_scale[0], f_scale[1]}, {f_output_size[0], f_output_size[1]});
    // copy dets [:, :0:2] slice back
    for(int i=0; i<batch; ++i){
        for(int j=0; j<n; ++j){
            (f_dets.data_ptr<scalar_t>())[((i*n) + j) * last_dim + 2] = target_coords[2*(i*n + j)];
            (f_dets.data_ptr<scalar_t>())[((i*n) + j) * last_dim + 3] = target_coords[2*(i*n + j) + 1];
        }
   }
   free(coords);
   free(target_coords);

   return dets;
}