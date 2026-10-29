import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch

# Determine if CUDA is available for compilation
cuda_available = torch.cuda.is_available()

sources = [
    "cuda/bindings.cpp",
    "cpp/scheduler_utils.cpp",
]

extra_compile_args = {
    "cxx": ["/std:c++17" if sys.platform == "win32" else "-std=c++17"],
}

if cuda_available:
    sources.extend([
        "cuda/custom_ops.cpp",
        "cuda/custom_kernels.cu",
    ])
    extra_compile_args["nvcc"] = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_89,code=sm_89",  # RTX 4060 Ada Lovelace
        "-gencode=arch=compute_80,code=sm_80",  # Ampere fallback
    ]
    ext_modules = [
        CUDAExtension(
            name="llm_engine_native",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    # CPU-only fallback extension
    ext_modules = [
        CppExtension(
            name="llm_engine_native",
            sources=["cuda/bindings.cpp", "cpp/scheduler_utils.cpp"],
            extra_compile_args=extra_compile_args,
        )
    ]

setup(
    name="llm_engine_native",
    version="0.1.0",
    description="Native C++ and CUDA acceleration operators for LLM inference",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
