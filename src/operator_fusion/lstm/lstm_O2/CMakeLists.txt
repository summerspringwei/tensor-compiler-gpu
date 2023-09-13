cmake_minimum_required(VERSION 3.14)

set(HUNTER_PACKAGES abseil GTest)

include(FetchContent)
FetchContent_Declare(SetupHunter GIT_REPOSITORY https://github.com/cpp-pm/gate)
FetchContent_MakeAvailable(SetupHunter)

set(CMAKE_BUILD_TYPE_INIT Release)

# project(mica_experiments LANGUAGES CUDA CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(absl CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80 86)
set(CUDA_ARCHITECTURES 80 86)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -cudart shared")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_AddUW_Float4_lib ictlstm100_10_256_naivefuse_fusedsolve_fusedcompute_adduw_float4_lib/*.cu) 
cuda_add_library(IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_AddUW_Float4_cudalib
    ${IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_AddUW_Float4_lib})

file(GLOB IctLstm100_10_256_Fusedcompute_lib ictlstm100_10_256_fusedcompute_lib/*.cu)
cuda_add_library(IctLstm100_10_256_Fusedcompute_cudalib
    ${IctLstm100_10_256_Fusedcompute_lib})

add_subdirectory(utils)
add_subdirectory(lstm_cell_net_100_10)

add_executable(BenchmarkFuseLstm100_10_1_256_Naivefuse_Fusedsolve_Fusedcompute_AddUW_Float4 BenchmarkFuseLstm100_10_1_256.cc)
target_link_libraries(BenchmarkFuseLstm100_10_1_256_Naivefuse_Fusedsolve_Fusedcompute_AddUW_Float4 GTest::gtest_main LSTM100_10 IctLstm100_10_256_Naivefuse_Fusedsolve_Fusedcompute_AddUW_Float4_cudalib IctLstm100_10_256_Fusedcompute_cudalib utilslib)