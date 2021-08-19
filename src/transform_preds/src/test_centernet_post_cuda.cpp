
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <vector>
#include <memory>

#include <cuda_runtime.h>

#include "affine_transform_cuda.hpp"

int test_get_affine_transform_cv(int argc, char*argv[])
{
    // Expect {}
    // Point2f src[] = {Point2f{240.f, 320.f}, Point2f{240.f, 64.f}, Point2f{-16.f, 64.f}};
    // Point2f dst[] = {Point2f{64.f, 84.f}, Point2f{64.f, 20.f}, Point2f{0.f, 20.f}};
    Point2f src[] = {Point2f{320.,  213.}, Point2f{320., -123.}, Point2f{-16., -123.}};
    Point2f dst[] = {Point2f{ 336.,  224.}, Point2f{336., -112.}, Point2f{ 0., -112.}};
    // Expect trasn = {1.00000000e+00 -0.00000000e+00  1.60000000e+01 -3.33066907e-17  1.00000000e+00  1.10000000e+01}
    double trans[6];
    auto t1 = std::chrono::steady_clock::now();
    int loop_count = 1;
    for(int i=0; i<loop_count;++i){
        get_affine_transform_cv(trans, dst, src);
    }
    auto t2 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::micro>(t2-t1).count();
    printf("Avg latency %f\n", latency / loop_count);
    return 0;
}

int test_affine_transform(int argc, char** argv)
{
    if(argc < 2){
        printf("Usage: transform_preds n loop_count\n");
        return 0;
    }
    int batch = 1;
    int n = atoi(argv[1]);
    int loop_count = atoi(argv[2]);
    assert((loop_count>0) && (n>0));
    
    std::vector<float> trans_vec = {4, 0, -16, 0, 4, -11};
    std::vector<float> coords_vec(n*2);
    for(int i=0; i<n; ++i){
        coords_vec[2*i] = 39.385944;
        coords_vec[2*i+1] = 69.22718;
    }
    float* coords = (float*)malloc(sizeof(float) * n * batch * 2);
    float* target_coords = (float*)malloc(sizeof(float) * n * batch * 2);

    affine_transform(target_coords, coords_vec.data(), trans_vec.data(), batch, n, loop_count);
    
    printf("target_coords:\n");
    auto num_out = n * batch;
    for(int i=0; i<(num_out < 16 ? num_out: 16);++i){
        printf("[%f %f] ", target_coords[2*i], target_coords[2*i+1]);
    }printf("\n");
    free(coords);
    free(target_coords);
    return 0;
}

void test_transform_preds(){
    Point2f center{240., 320.}, scale{512., 672.}, output_size{128, 168};
    float coords[] = { 78.93826f, 163.65175f};
    float target_coords[2]; // Expects to be 299.75305176 638.60699463
    int batch = 1, n = 1;
    transform_preds(target_coords, coords, batch, n, center, scale, output_size);
    printf("target coords: ");
    for(int i=0 ;i<batch*n; ++i){
        printf("%f %f ", target_coords[2*i], target_coords[2*i+1]);
    }printf("\n");
}

void test_affine_transform_dets_cuda(){
    float target_dets[6], dets[6] = {78.93826f, 163.65175f, 78.93826f, 163.65175f, 78.93826f, 163.65175f};
    float trans[6] = {4.000000, -0.000000, -16.000000, -0.000000, 4.000000, -16.000000};
    int batch=1, n=1;
    float* d_target_dets = nullptr;
    float* d_dets = nullptr;
    float* d_trans = nullptr;
    cudaMalloc((void**)&d_target_dets, sizeof(float) * 6);
    cudaMalloc((void**)&d_dets, sizeof(float) * 6);
    cudaMalloc((void**)&d_trans, sizeof(float) * 6);
    cudaMemcpy(d_dets, dets, sizeof(float) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_trans, trans, sizeof(float) * 6, cudaMemcpyHostToDevice);
    affine_transform_dets_cuda(d_target_dets, d_dets, d_trans, batch, n);
    cudaDeviceSynchronize();
}

void test_affine_transform_dets_torch(){
    float data_target_dets[6], data_dets[6] = {78.93826f, 163.65175f, 78.93826f, 163.65175f, 78.93826f, 163.65175f};
    float data_trans[6] = {4.000000, -0.000000, -16.000000, -0.000000, 4.000000, -16.000000};
    torch::Tensor dets = torch::from_blob(data_dets, {6}).to(torch::kCUDA);;
    torch::Tensor trans = torch::from_blob(data_trans, {6}).to(torch::kCUDA);;
    torch::Tensor target_dets = torch::from_blob(data_dets, {6}).to(torch::kCUDA);;
    int batch=1, n=1;

    affine_transform_dets_cuda(target_dets.data_ptr<float>(), dets.data_ptr<float>(), trans.data_ptr<float>(), batch, n);
    target_dets.print();
    cudaDeviceSynchronize();
}


int main(int argc, char** argv){
    // test_get_affine_transform_cv(argc, argv);
    // test_affine_transform(argc, argv);
    // test_transform_preds();
    // test_affine_transform_dets_cuda();
    test_affine_transform_dets_torch();
    return 0;
}
