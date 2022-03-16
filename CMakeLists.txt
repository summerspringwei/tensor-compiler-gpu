
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc")
project(tensor-compiler-gpu LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.5)
include(CheckLanguage)
check_language(CUDA)
# set(CMAKE_C_COMPILER "/usr/local/cuda-10.0/bin//nvcc")
# set(CMAKE_CXX_COMPILER "/usr/local/cuda-10.0/bin//nvcc")
set(CMAKE_CUDA_ARCHITECTURES "70;75;80")
SET(CUDA_SEPARABLE_COMPILATION ON)
include_directories("/usr/local/cuda-11.0/include" "/usr/local/cudnn-11.2/include" "./inc")
link_directories("/usr/local/cuda-11.0/lib64" "/usr/local/cudnn-11.2/lib64")


include_directories("third_party/libnpy/include")
# add_compile_options(-fno-stack-protector)

add_executable(reduce_using_reg src/learn_cuda/reduce_using_reg.cu)
add_executable(add src/learn_cuda/Add.cu)
add_executable(reduce src/learn_cuda/Reduce.cu)
add_executable(normalize src/learn_cuda/normalize.cu)
add_executable(sobel src/learn_cuda/sobel.cu)
add_executable(cudnn_conv src/learn_cuda/cudnn_conv.cu)
add_executable(pointwise_conv src/learn_cuda/pointwise_conv.cu)
file(GLOB FUSE_SRC_FILES src/operator_fusion/efficientnet/*.cu)

# add_executable(fused_pointwise_pointwise_conv src/operator_fusion/efficientnet/fused_pointwise_pointwise_main.cu src/operator_fusion/efficientnet/pointwise_112_112_144_6.cu)
add_executable(fused_pointwise_pointwise_conv ${FUSE_SRC_FILES})
# add_executable(tvmgen src/tvm_codegen/fused_matmul_main.cu src/tvm_codegen/block_fuse_d2l_matmul_kernel0 /tmp/mmult_kernel0.cu)

add_executable(lstm_main src/lstm/lstm_main.cu)
# add_executable(lstm_wavefront_timesteps src/lstm/lstm_wavefront_timesteps.cu) 
add_executable(seq2seq src/lstm/seq2seq.cu)
add_executable(lstm_cpu src/lstm/lstm_cpu.cpp)
add_executable(test_atomic_add src/block_sync/test_atomic_add.cu)
add_executable(cu_fused_pw_dw_conv src/operator_fusion/fused_depthwise_pointwise_conv.cu)
# set_target_properties(lstm_wavefront_magic PROPERTIES CUDA_ARCHITECTURES "70;75;80")
target_link_libraries(cudnn_conv cudnn)

