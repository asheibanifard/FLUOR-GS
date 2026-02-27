from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="splat_mip_tiled_cuda",
    ext_modules=[
        CUDAExtension(
            "splat_mip_tiled_cuda",
            ["splat_mip_tiled_cuda.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
