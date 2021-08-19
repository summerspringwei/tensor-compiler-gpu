
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc")
# project(tensor-compiler-gpu LANGUAGES C CXX CUDA)
# cmake_minimum_required(VERSION 3.5)
# # include(CheckLanguage)
# # check_language(CUDA)
# # set(CMAKE_C_COMPILER "/usr/local/cuda-10.0/bin//nvcc")
# # set(CMAKE_CXX_COMPILER "/usr/local/cuda-10.0/bin//nvcc")
# set(CMAKE_CUDA_ARCHITECTURES 75)

# include_directories("/usr/local/cuda-10.1/include" "/home/xiachunwei/Software/anaconda3/envs/openmmlab/lib/python3.7/site-packages/torch/include" "/home/xiachunwei/Software/anaconda3/envs/openmmlab/lib/python3.7/site-packages/torch/include/torch/csrc/api/include/" "/home/xiachunwei/Software/anaconda3/envs/openmmlab/include/python3.7m")
# link_directories("/usr/local/cuda-10.1/lib64" "/home/xiachunwei/Software/anaconda3/envs/openmmlab/lib/python3.7/site-packages/torch/lib")
# add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)

# add_library(affine_transform src/affine_transform_cuda.cu src/get_affine_transform_cuda.cu src/transform_preds_cuda.cpp src/print_cuda.cu)

# add_executable(test_centernet_post_cuda src/test_centernet_post_cuda.cpp)

# target_link_libraries(test_centernet_post_cuda affine_transform cublas cublasLt cudart cusolver c10 c10_cuda torch torch_cpu torch_cuda torch_global_deps torch_python)
