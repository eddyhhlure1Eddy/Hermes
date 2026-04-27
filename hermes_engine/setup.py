"""Build hermes_fast._C — the custom CUDA extension.

Build:
    cd src/
    TORCH_CUDA_ARCH_LIST=12.0 pip install -e . -v

On 201:
    export CUDA_HOME=/usr/local/cuda-13.1
    export PATH="$CUDA_HOME/bin:$PATH"

If the build fails, all Python imports still work — fused_ops.py falls back
to torch.layer_norm + torch.add (slower but correct).
"""

import os
from setuptools import setup
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

ext = CUDAExtension(
    name="hermes_fast._C",
    sources=[
        "csrc/fused_ln_residual.cu",
        "csrc/bindings.cpp",
    ],
    extra_compile_args={
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    },
    include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")],
)

setup(
    name="hermes_fast",
    version="0.1.0",
    description="Custom training kernels for Spatiotemporal Hermes v1 (sm_120)",
    packages=["hermes_fast", "hermes_fast.python"],
    package_dir={
        "hermes_fast": ".",
        "hermes_fast.python": "python",
    },
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
