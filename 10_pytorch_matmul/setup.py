from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_matmul',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='custom_matmul',
            sources=[
                'csrc/matmul.cpp',
                'csrc/matmul_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
