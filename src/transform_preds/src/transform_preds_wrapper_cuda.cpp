
#include "transform_preds_wrapper_cuda.hpp"
#include "affine_transform_cuda.hpp"
#include "print_cuda.hpp"
#include <cuda_runtime.h>


at::Tensor transform_preds_cuda_forward_v1(const at::Tensor &dets,
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
    print_cuda(dets.data_ptr<float>(), long(batch * n * last_dim));
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
    for(int i=0; i<batch; ++i){
        for(int j=0; j<n; ++j){
             printf("[%f %f] ", target_coords[2*(i*n + j)], target_coords[2*(i*n + j)+1]);
        }printf("\n");
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
    printf("f_dets device: %d", f_dets.device());
    // auto g_dets = f_dets.to(torch::kCUDA);
    // printf("g_dets device: %d", g_dets.device());
    for(int i=0; i<batch; ++i){
        for(int j=0; j<n; ++j){
             printf("[%f %f] ", target_coords[2*(i*n + j)], target_coords[2*(i*n + j)+1]);
        }printf("\n");
    }

    free(coords);
    free(target_coords);
    print_cuda(dets.data_ptr<float>(), long(batch * n * last_dim));
    dets.print();
    f_dets.print();
    // auto g_dets = f_dets.to(torch::kCUDA);
    // torch::Tensor gpu_two_tensor = f_dets.to(torch::kCUDA);
    // gpu_two_tensor.print();
    // return f_dets.to(torch::kCUDA);
    return f_dets;
}

at::Tensor transform_preds_cuda_forward(const at::Tensor dets,
                    const at::Tensor &center,
                    const at::Tensor &scale,
                    const at::Tensor &output_size,
                    int num_classes){
    // dets shape (batch, 100, 6)
    // int batch = dets.size(0);
    // int n = dets.size(1);
    // int last_dim = dets.size(2);
    int batch = 1;
    int n = 1;
    int last_dim = 6;
    auto target_dets = torch::ones({last_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    target_dets.print();
    // print_cuda(target_dets.data_ptr<float>(), long(batch * n * last_dim));
    print_cuda(target_dets.packed_accessor64<float,1>(), long(batch * n * last_dim));
    print_cuda(target_dets.packed_accessor64<float,1>(), long(batch * n * last_dim));
    dets.print();
    printf("addr: %p\n", (void*)(target_dets.data_ptr<float>()));
    // auto f_dets = dets.to(torch::kFloat32);
    // float* ptr_f_dets = f_dets.data_ptr<float>();
    // print_cuda(dets.data_ptr<float>(), long(batch * n * last_dim));
    print_cuda(dets.packed_accessor64<float,1>(), long(batch * n * last_dim));
    using scalar_t = float;
    auto f_center = center.data_ptr<scalar_t>();
    auto f_scale = scale.data_ptr<scalar_t>();
    auto f_output_size = output_size.data_ptr<scalar_t>();

    // Compute affine transform
    double trans[6];
    get_affine_transform(trans, {f_center[0], f_center[1]}, {f_scale[0], f_scale[1]}, 0, {f_output_size[0], f_output_size[1]}, {0., 0.});
    float f_trans[6];
    for(int i=0; i<6;++i){
        f_trans[i] = (float)trans[i];
    }
    // Apply affine transform to dets
    float* d_trans = nullptr;
    cudaMalloc((void**)&d_trans, sizeof(float) * 6);
    cudaMemcpy(d_trans, f_trans, sizeof(float) * 6, cudaMemcpyHostToDevice);
    // affine_transform_dets_cuda(target_dets.packed_accessor64<float,1>(), dets.packed_accessor64<float,1>(), d_trans, batch, n);
    printf("addr: %p\n", (void*)(target_dets.data_ptr<float>()));
    // print_cuda(target_dets.data_ptr<float>(), long(batch * n * last_dim));
    print_cuda(target_dets.packed_accessor64<float,1>(), long(batch * n * last_dim));
    
    cudaDeviceSynchronize();
    cudaFree(d_trans);
    return target_dets;
}
