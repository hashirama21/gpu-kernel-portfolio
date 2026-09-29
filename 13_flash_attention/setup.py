from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_flash_attention',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='custom_flash_attention',
            sources=['csrc/flash_attention.cpp', 'csrc/flash_attention_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
