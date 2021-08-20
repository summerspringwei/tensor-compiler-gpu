
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc")
project(transform_preds LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.5)
# include(CheckLanguage)
# check_language(CUDA)
# set(CMAKE_C_COMPILER "/usr/local/cuda-10.0/bin//nvcc")
# set(CMAKE_CXX_COMPILER "/usr/local/cuda-10.0/bin//nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)

if(DEFINED ENV{CUDA_HOME})
    include_directories("$ENV{CUDA_HOME}/include")
    link_directories("$ENV{CUDA_HOME}/lib64")
else()
    include_directories("/usr/local/cuda/include")
    link_directories("/usr/local/cuda/lib64")
endif()

add_library(affine_transform src/cuda/affine_transform_cuda.cu src/cuda/print_cuda.cu)
add_library(get_affine_transform src/cpu/get_affine_transform_cpu.cpp)

add_executable(test_affine_transform_dets_cuda src/cuda/test_affine_transform_dets_cuda.cpp.cpp)

target_link_libraries(test_affine_transform_dets_cuda affine_transform cudart)
