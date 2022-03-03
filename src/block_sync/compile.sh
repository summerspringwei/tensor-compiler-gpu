# /usr/local/cuda/bin/nvcc -ccbin g++ -m64 -O2 -Xptxas -dlcm=cg --std=c++11 -gencode arch=compute_80,code=sm_80 -o test_atomic_add test_atomic_add.cu

# Compile binary
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -O2 --std=c++11 -gencode arch=compute_80,code=\"sm_80,compute_80\" -o test_atomic_add test_atomic_add.cu
# Get ptx and sass file
cuobjdump -ptx test_atomic_add > test_atomic_add.ptx
cuobjdump -sass test_atomic_add > test_atomic_add.sass
