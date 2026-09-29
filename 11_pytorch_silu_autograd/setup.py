from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_silu',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='custom_silu',
            sources=['csrc/silu.cpp', 'csrc/silu_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
