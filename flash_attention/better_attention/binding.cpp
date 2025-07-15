#include <torch/extension.h>

torch::Tensor multi_head_attention_better(
    torch::Tensor x, // the input batch (B, T, C)
    torch::Tensor q_weight, // concatenate query matrices (C, C)
    torch::Tensor k_weight, // concatenate of key matrices (C, C)
    torch::Tensor v_weight, // concatenate of value matrices (C, C)
    torch::Tensor proj, // redundancy matrix
    int num_heads
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_attention", &multi_head_attention_better, "faster way to compute multi-head attention");
}