from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_attention',
    ext_modules=[
        CUDAExtension(
            name='my_attention',
            sources=['bindings.cpp', 'naive_attention.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)