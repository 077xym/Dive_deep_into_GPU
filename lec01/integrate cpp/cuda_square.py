import torch
from torch.utils.cpp_extension import load_inline

# define the cuda kernel
cuda_source = """
__global__ void square_matrix_kernel(const float *matrix, float *result, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int idx = row * m + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}


torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto m = matrix.size(0);
    const auto n = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);
    square_matrix_kernel<<<grid_size, block_size>>>(matrix.data_ptr<float>(), result.data_ptr<float>(), m, n);

    return result;
}

"""

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

squared_matrix_extension = load_inline(
    name = 'sqaure_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cflags=["-O2"],
    build_directory='./load_inline_cuda'
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(squared_matrix_extension.square_matrix(a))
