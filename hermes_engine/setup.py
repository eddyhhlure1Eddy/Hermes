import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST", "12.0")

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--threads", "4",
]

for arch in ARCH_LIST.split(";"):
    arch = arch.strip()
    if not arch:
        continue
    sm = arch.replace(".", "")
    suffix = "a" if sm in ("90", "100", "120") else ""
    nvcc_flags += [
        "-gencode", f"arch=compute_{sm}{suffix},code=sm_{sm}{suffix}",
    ]

cxx_flags = ["-O3", "-std=c++17"]
if os.name == "nt":
    cxx_flags = ["/O2", "/std:c++17"]

ext = CUDAExtension(
    name="hermes_engine._C",
    sources=[
        "csrc/hermes_engine.cu",
    ],
    extra_compile_args={
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    },
    include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")],
)

setup(
    name="hermes_engine",
    version="0.1.0",
    description="Hermes CUDA inference engine for ConditionalFinancialModel (sm_120a Blackwell)",
    packages=["hermes_engine"],
    package_dir={"hermes_engine": "python"},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
