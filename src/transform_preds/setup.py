#!/usr/bin/env python

import glob
import os
import sys

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    # main_file = ["transform_preds_wrapper_.cpp"]
    # source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    # source_cuda = ["transform_preds.cpp", "affine_transform.cu", "get_affine_transform.cu"]
    

    os.environ["CC"] = "g++"
    sources = glob.glob(os.path.join(extensions_dir, "*.c*"))
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        # sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            # "CUDA_LAUNCH_BLOCKING=1",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-lcublas -lcublasLt -lcudart -lcusolver",
        ]
    else:
        raise NotImplementedError('Cuda is not available')
        
    
    # extra_compile_args['cxx'].append('-fopenmp')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir, "/usr/local/cuda-11.0/include"]
    ext_modules = [
        extension(
            "_transform_preds",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="transform_preds",
    version="0.1",
    author="ChunweiXia",
    url="gitee.com:xiachunwei/tensor-compiler-gpu.git",
    description="transform_preds for CenterNet",
    packages=find_packages(
        exclude=(
            "configs",
            "tests",
        )
    ),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
