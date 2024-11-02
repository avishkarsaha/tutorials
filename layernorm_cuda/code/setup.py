from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='layernorm_cuda',
    ext_modules=[
        CUDAExtension(
            name='layernorm_cuda',
            sources=['layernorm_cuda.cpp', 'layernorm_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_87'  # for NVIDIA L4
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
