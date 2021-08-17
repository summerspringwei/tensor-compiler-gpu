
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc")
project(tensor-compiler-gpu LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.5)
# include(CheckLanguage)
# check_language(CUDA)
# set(CMAKE_C_COMPILER "/usr/local/cuda-10.0/bin//nvcc")
# set(CMAKE_CXX_COMPILER "/usr/local/cuda-10.0/bin//nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories("/usr/local/cuda-11.0/include")
link_directories("/usr/local/cuda-11.0/lib64")
# add_compile_options(-fno-stack-protector)

add_library(affine_transform src/affine_transform.cu src/get_affine_transform.cu src/transform_preds.cpp)

add_executable(reduce_using_reg src/reduce_using_reg.cu)
add_executable(add src/Add.cu)
add_executable(reduce src/Reduce.cu)
add_executable(normalize src/normalize.cu)
add_executable(sobel src/sobel.cu)
add_executable(test_centernet_post src/test_centernet_post.cpp)
target_link_libraries(test_centernet_post affine_transform cublas cublasLt cudart cusolver)
# add_executable(transform_preds src/transform_preds.cu)
# add_executable(cuda_solve src/cuda_solve.cpp)

# target_link_libraries(transform_preds cusolver cublas cublasLt)
# target_link_libraries(cuda_solve cublas cublasLt cudart cusolver)