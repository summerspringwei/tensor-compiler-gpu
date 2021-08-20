
#include "get_affine_transform_cpu_torch_wrapper.hpp"

#include "get_affine_transform_cpu.hpp"

at::Tensor get_affine_transform_cpu_forward(const at::Tensor center, 
                    const at::Tensor scale, float rot, const at::Tensor output_size, 
                    const at::Tensor shift, int inv){
    auto src = torch::zeros({6, 6}, torch::dtype(torch::kFloat64).device(torch::kCPU));
    auto dst = torch::zeros({6, 1}, torch::dtype(torch::kFloat64).device(torch::kCPU));
    format_affine_transform_cpu(src.data_ptr<double>(), dst.data_ptr<double>(), 
                {center.data_ptr<float>()[0], center.data_ptr<float>()[1]},
                {scale.data_ptr<float>()[0], scale.data_ptr<float>()[0]}, rot, 
                {output_size.data_ptr<float>()[0], output_size.data_ptr<float>()[1]}, 
                {shift.data_ptr<float>()[0], shift.data_ptr<float>()[1]}, inv);
    #ifdef DEBUG
        printf("src:");
        for(int i=0;i<6; ++i){
            printf("[");
            for(int j=0; j<6; ++j){
                printf("%f, ", src.data_ptr<double>()[i*6+j]);
            }printf("],");
            printf("\n");
        }printf("dst: ");
        for(int i=0;i<6; ++i){
            printf("%f, ", dst.data_ptr<double>()[i]);
        }printf("\n");
    #endif
    auto inv_src = src.inverse();
    auto trans = inv_src.matmul(dst);
    return trans;
}
