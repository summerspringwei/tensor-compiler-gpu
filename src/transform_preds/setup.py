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

    main_file = [os.path.join(extensions_dir, "transform_preds.cpp")]
    sources_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    
    os.environ["CC"] = "g++"
    extension = CppExtension
    define_macros = []
    extra_compile_args = {"cxx": []}
    extra_compile_args['cxx'].append('-fopenmp')
    extra_compile_args['cxx'].append('--std=c++11')
    # extra_compile_args['cxx'].append('-DDEBUG=1')
    
    sources = []
    sources += main_file
    sources += sources_cpu

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.c*"))
        sources += sources_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            # "CUDA_LAUNCH_BLOCKING=1",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-I/usr/local/cuda-10.2/lib64",
            "-lcublas -lcublasLt -lcudart -lcusolver",
            # "-DDEBUG=1"
        ]
    else:
        raise NotImplementedError('Cuda is not available')


    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir, "/usr/local/cuda-10.2/include"]
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
