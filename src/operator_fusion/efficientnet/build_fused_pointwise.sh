/usr/local/cuda/bin/nvcc -ccbin g++   -m64 -g -G  -O2 \
 -maxrregcount=255 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 \
 -o pointwise_conv.o -c pointwise_conv.cu 
/usr/local/cuda/bin/nvcc -ccbin g++   -m64 -g -G   -O2  \
 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 \
 -o pointwise_conv pointwise_conv.o
