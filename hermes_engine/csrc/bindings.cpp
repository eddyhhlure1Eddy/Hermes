// PyTorch bindings for hermes_fast custom kernels.

#include <torch/extension.h>

namespace hermes_fast {
torch::Tensor fused_ln_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ln_residual", &hermes_fast::fused_ln_residual,
          "out = LayerNorm(x + residual) (fused, bf16/fp16, sm_120)");
}
