#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"


// The configuration for gemm
using ElementAccumulator = double;                  
using ElementComputeEpilogue = double; 
using ElementInputA = double;              
using ElementInputB = double;                  
using ElementOutput = double;                   

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm80;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 16>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 16>;  
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>; 

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     
    128 / cutlass::sizeof_bits<ElementOutput>::value,  
    ElementAccumulator,                                
    ElementComputeEpilogue>;  

constexpr int NumStages = 4;

using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                                                  ElementOutput, LayoutOutput, ElementAccumulator,
                                                  MMAOp, 
                                                  SmArch, 
                                                  ShapeMMAThreadBlock,
                                                  ShapeMMAWarp,
                                                  ShapeMMAOp,
                                                  EpilogueOp,
                                                  SwizzleThreadBlock,
                                                  NumStages>;

int run(){
  // problem size
  const int length_m = 1024;
  const int length_n = 128;
  const int length_k = 1024;
  cutlass::gemm::GemmCoord problem_size = {length_m, length_n, length_k};
  
  // modify the problem size based on the alignment
  int alignment_a = 128 / sizeof(ElementInputA) / 8;
  int alignment_b = 128 / sizeof(ElementInputB) / 8;
  int alignment_c = 128 / sizeof(ElementOutput) / 8;

  int lda = ((problem_size.m() + alignment_a - 1) / alignment_a) * alignment_a;
  int ldb = ((problem_size.n() + alignment_b - 1) / alignment_b) * alignment_b;
  int ldc = ((problem_size.m() + alignment_c - 1) / alignment_c) * alignment_c;
  int ldd = ldc;
  
  problem_size = {lda, ldb, problem_size.k()};

  // Create operand matrix
  // cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk()); 
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.km()); 
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
  // cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.nm());
  // cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.nm());

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1, ElementInputA(4), ElementInputA(-4), 0);  
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1, ElementInputB(4), ElementInputB(-4), 0);  
  cutlass::reference::host::TensorFillRandomUniform(tensor_c.host_view(), 1, ElementOutput(4), ElementOutput(-4), 0);  
  cutlass::reference::host::TensorFill(tensor_d.host_view());  
  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Set the arguments for GemmUniversal
  cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;

  int batch_count = 1;

  int batch_stride_A = tensor_a.size();
  int batch_stride_B = tensor_b.size();
  int batch_stride_C = tensor_c.size();
  int batch_stride_D = tensor_d.size();

  typename Gemm::Arguments arguments{
    mode,
    problem_size,
    batch_count,
    {alpha, beta},
    tensor_a.device_ref().data(),              // <- reference to matrix A on device
    tensor_b.device_ref().data(),              // <- reference to matrix B on device
    tensor_c.device_ref().data(),              // <- reference to matrix C on device
    tensor_d.device_ref().data(),              // <- reference to matrix D on device
    batch_stride_A,
    batch_stride_B,
    batch_stride_C,
    batch_stride_D,
    lda,
    ldb,
    ldc,
    ldd
  };

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // total computation time
  float total_time = 0;
  // repetition times
  const int profile_times = 100;
  const int warmup_times = 100;
  
  // First warm up separately
  for(int i = 0; i < warmup_times; i++){
    status = gemm_op();
  }
  CUTLASS_CHECK(status);

  // Begin profiling
  cudaEvent_t start1;
  cudaEventCreate(&start1);
  cudaEvent_t stop1;
  cudaEventCreate(&stop1);
  cudaEventRecord(start1, NULL);
  for(int i = 0; i < profile_times; i++){
    status = gemm_op();
  }
  cudaEventRecord(stop1, NULL);
  cudaEventSynchronize(stop1); 
  cudaEventElapsedTime(&total_time, start1, stop1); 

  float average_time = total_time / profile_times;
  double peak_performance = (double) length_m * length_n * length_k * 2.0 / (average_time / 1000) / 1e9; // GFLOPS

  std::cout << "Time / ms : " << average_time << std::endl;
  std::cout << "Performance / GFLOPS : " << peak_performance << std::endl;

  return 0;
}

int main(int argc, char const *argv[])
{
  
  return run();
}