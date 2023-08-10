# /usr/local/cuda//bin/nvcc \
#     -forward-unknown-to-host-compiler \
#     -I/usr/local/cuda/include \
#     -I/usr/local/cuda/targets/x86_64-linux/include \
#     -I/home/xiachunwei/anaconda3/lib/python3.8/site-packages/torch/include \
#     -I/home/xiachunwei/anaconda3/lib/python3.8/site-packages/torch/include/torch/csrc/api/include\
#     -I/home/xiachunwei/Projects/tensor-compiler-gpu/third_party/libnpy/include \
#     -I/include -I/tools/util/include \
#     -I/examples/common \
#     -g -G -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 \
#     --generate-code=arch=compute_80,code=[compute_80,sm_80] --generate-code=arch=compute_86,code=[compute_86,sm_86] \
#     "-D_GLIBCXX_USE_CXX11_ABI=0 -g -G -O0" -MD -MT -x cu -dc vector_matrix_mul_main.cu


/usr/local/cuda//bin/nvcc \
    -forward-unknown-to-host-compiler \
    -I/usr/local/cuda/include \
    -I/usr/local/cuda/targets/x86_64-linux/include \
    -I/home/xiachunwei/anaconda3/lib/python3.8/site-packages/torch/include \
    -I/home/xiachunwei/anaconda3/lib/python3.8/site-packages/torch/include/torch/csrc/api/include\
    -I/home/xiachunwei/Projects/tensor-compiler-gpu/third_party/libnpy/include \
    -I/include -I/tools/util/include \
    -I/examples/common \
    -L/build/tools/library  -L/home/xiachunwei/anaconda3/lib/python3.8/site-packages/torch/lib \
    -L/home/xiachunwei/Projects/tensor-compiler-gpu/dbg_build \
     -Wl,-rpath,/build/tools/library:/home/xiachunwei/anaconda3/lib/python3.8/site-packages/torch/lib:/home/xiachunwei/Projects/tensor-compiler-gpu/dbg_build \
      -lcublasLt -lcublas -ltorch -ltorch_cuda -lc10 -lc10_cuda -ltorch_python -ltorch_cpu -ltorch_utils -lcudadevrt -lcudart_static -lrt -lpthread -ldl  \
      -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" -L"/usr/local/cuda/targets/x86_64-linux/lib" \
    -g -G -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 \
    --generate-code=arch=compute_80,code=[compute_80,sm_80] --generate-code=arch=compute_86,code=[compute_86,sm_86] \
    -D_GLIBCXX_USE_CXX11_ABI=0 -g -G -O0 vector_matrix_mul_main.cu
