#include <torch/extension.h>

torch::Tensor mat_mul(torch::Tensor a, torch::Tensor b);
torch::Tensor mask(torch::Tensor a, torch::Tensor b, float value);
torch::Tensor softmax(torch::Tensor a);
torch::Tensor tensor_mul(torch::Tensor a, torch::Tensor b);
torch::Tensor multi_head_attention_naive(
    torch::Tensor x, // the input batch (B, T, C)
    torch::Tensor q_weight, // concatenate query matrices (C, C)
    torch::Tensor k_weight, // concatenate of key matrices (C, C)
    torch::Tensor v_weight, // concatenate of value matrices (C, C)
    torch::Tensor proj, // redundancy matrix
    int num_heads
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mat_mul", &mat_mul, "Batched matrix multiplication");
    m.def("mask", &mask, "Masking");
    m.def("softmax", &softmax, "do softmax on a tensor");
    m.def("tensor_mul", &tensor_mul, "do tensor multiplication");
    m.def("naive_attention", &multi_head_attention_naive, "naive computation of multihead attention layer");
}