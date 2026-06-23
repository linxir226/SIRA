from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package metadata
NAME = "sira"
VERSION = "0.1.0"
DESCRIPTION = "SIRA: reasoning-aware surgical instrument segmentation"
AUTHOR = "Zhibo Zhang, Qijie Wang, Zengqiang Yan"
URL = "https://github.com/linxir226/SIRA"

# Read the contents of README file
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "torch>=2.3.1",
    "torchvision>=0.18.1",
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
]

EXTRA_PACKAGES = {
    "demo": ["matplotlib>=3.9.1", "jupyter>=1.0.0", "opencv-python>=4.7.0"],
    "dev": ["black==24.2.0", "usort==1.0.2", "ufmt==2.0.0b2"],
}


def get_extensions():
    srcs = ["sam2/csrc/connected_components.cu"]
    compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    }
    ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    return ext_modules


# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    packages=find_packages(exclude="notebooks"),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    python_requires=">=3.10.0",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
