from glob import glob
from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))


extra_compile_args = {"cxx": []}
extra_compile_args["nvcc"] = [
            "-g",
            "-DCUDA_HAS_FP16=1",
            "-DUSE_FP16=ON",
            "-DCUDA_ARCH_BIN=80",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            ]
setup(name='bert_binding',
      ext_modules=[
        cpp_extension.CUDAExtension(
          'bert_binding', 
          ['bert_binding.cu'],
          include_dirs=["/usr/local/cuda/include", this_dir+"../../"],
          library_dirs=["/usr/local/cuda/lib64"],
          extra_compile_args=extra_compile_args
          )
        ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})