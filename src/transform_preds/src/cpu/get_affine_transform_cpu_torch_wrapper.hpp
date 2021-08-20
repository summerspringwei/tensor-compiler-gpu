
#include <torch/extension.h>

at::Tensor get_affine_transform_cpu_forward(const at::Tensor center, 
                    const at::Tensor scale, float rot, const at::Tensor output_size, 
                    const at::Tensor shift, int inv);