#include <cuda_runtime.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <float.h>
#include <vector>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)


#define TILE_SIZE 32
#define BLOCK_SIZE 256


__global__ void matmul(float *a, float *b, float *c, int batch, int m, int k, int n) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int layer = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (layer < batch && row < m && tile * TILE_SIZE + threadIdx.x < k) {
            sharedA[threadIdx.y][threadIdx.x] = a[layer * m * k + row * k + tile * TILE_SIZE + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (tile * TILE_SIZE + threadIdx.y < k && col < n) {
            sharedB[threadIdx.y][threadIdx.x] = b[(tile * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int l = 0; l < TILE_SIZE; l++) {
            sum += sharedA[threadIdx.y][l] * sharedB[l][threadIdx.x];
        }
        __syncthreads();
    }

    if (layer < batch && row < m && col < n) {
        c[layer * m * n + row * n + col] = sum;
    }
}


__global__ void matmul_tensor(float *a, float *b, float *c, int batch, int m, int k, int n) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int layer = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (layer < batch && row < m && tile * TILE_SIZE + threadIdx.x < k) {
            sharedA[threadIdx.y][threadIdx.x] = a[layer * m * k + row * k + tile * TILE_SIZE + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (layer < batch && tile * TILE_SIZE + threadIdx.y < k && col < n) {
            sharedB[threadIdx.y][threadIdx.x] = b[layer * k * n + (tile * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int l = 0; l < TILE_SIZE; l++) {
            sum += sharedA[threadIdx.y][l] * sharedB[l][threadIdx.x];
        }
        __syncthreads();
    }

    if (layer < batch && row < m && col < n) {
        c[layer * m * n + row * n + col] = sum;
    }
}


__global__ void ele_prod(float *a, float *b, int batch, int m, int n) {
    int layer = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (layer < batch && row < m && col < n) {
        a[layer*m*n + row * n + col] *= b[row * n + col];
    }
}


__global__ void softmax_kernel(float *a, int batch, int m, int n, int num_elements_per_thread) {
    // we create a block for each row of the tensor (B, T, C)
    // the block size depends on how many elements we decided to operate on each thread
    // let's say n = 2048, and each thread is responsible to operate on 32 elements, we then get a block size of 64, 1, 1
    // So the grid size will be (batch, m, 1), block size will be (1, 64), in our cases
    // For fused softmax, what we need to do is
    // each thread compute local max from num_elements it is responsible for
    // then we will get C/num_elements local max
    // use reduction to get global max of each row
    // then each thread is responsible to do element - global max
    // then each thread is responsible to raise exponentially of each element it is responsible
    // then each thread is responsible for sum up all its elements to have C/num_elements local sum
    // do reduction to get global sum of the row
    // then each thread is responsible for normalize the elements
    extern __shared__ float shared[];
    // load row onto shared memory
    float *r = shared;
    // store the local max into shared memory
    float *shared_max = shared + n;

    int layer = blockIdx.z;
    int row = blockIdx.y;
    int tid = threadIdx.x;

    // load the row onto shared memory
    for (int i = 0; i < num_elements_per_thread; i++) {
        int col = tid * num_elements_per_thread + i;
        if (col < n) {
            r[col] = a[layer * m * n + row * n + col];
        }
    }

    __syncthreads();

    // each thread computes local max, resulting in n / num_elements_per_thread local max per block
    float local_max = -FLT_MAX;
    for (int i = 0; i < num_elements_per_thread; i++) {
        int col = tid * num_elements_per_thread + i;
        if (col < n) {
            local_max = fmaxf(local_max, r[col]);
        }
    }
    shared_max[tid] = local_max;

    __syncthreads();

    // use reduction to compare the local max
    int active_threads = blockDim.x;
    while (active_threads > 1) {
        int half_point = (active_threads + 1) / 2;
        if (tid < half_point && tid + half_point < active_threads) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid+half_point]);
        }
        active_threads = half_point;
        __syncthreads();
    }

    float global_max = shared_max[0];

    // raise each element, and reuse shared_max to compute local sum
    // since we already have the global max.
    float local_sum = 0.0f;
    for (int i = 0; i < num_elements_per_thread; i++) {
        int col = tid * num_elements_per_thread + i;
        if (col < n) {
            r[col] = expf(r[col]-global_max);
            local_sum += r[col];
        }
    }
    shared_max[tid] = local_sum;

    __syncthreads();

    // use reduction to get the sum of the entire row
    int active_threads_for_sum = blockDim.x;
    while (active_threads_for_sum > 1) {
        int half_point = (active_threads_for_sum + 1) / 2;
        if (tid < half_point && tid + half_point < active_threads_for_sum) {
            shared_max[tid] += shared_max[tid+half_point];
        }
        active_threads_for_sum = half_point;
        __syncthreads();
    }
    float global_sum = shared_max[0];

    // now we have each element set in r, and global_sum, we can do normalization
    for (int i = 0; i < num_elements_per_thread; i++) {
        int col = tid * num_elements_per_thread + i;
        if (col < n) {
            a[layer * m * n + row * n + col] = r[col] / global_sum;
        }
    }
}


torch::Tensor mat_mul(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    const auto batch = a.size(0);
    const auto m = a.size(1);
    const auto k = a.size(2);
    const auto n = b.size(1);

    torch::Tensor c = torch::zeros({batch, m, n}, a.options());

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE, batch);
    matmul<<<grid_size, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), batch, m, k, n);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    return c;
}


torch::Tensor tensor_mul(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    const auto batch = a.size(0);
    const auto batch_b = b.size(0);
    const auto m = a.size(1);
    const auto k = a.size(2);
    const auto n = b.size(2);

    torch::Tensor c = torch::zeros({batch, m, n}, a.options());

    TORCH_CHECK(batch == batch_b, "tensor must have same number of batches");
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE, batch);
    matmul_tensor<<<grid_size, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), batch, m, k, n);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    return c;
}


torch::Tensor mask(torch::Tensor a, torch::Tensor b, float value) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    const auto batch = a.size(0);
    const auto m = a.size(1);
    const auto n = a.size(2);
    const auto m_b = b.size(0);
    const auto n_b = b.size(1);

    TORCH_CHECK(m == m_b && n == n_b, "Tensor b must have shape (", m, ", ", n, "), but got (", m_b, ", ", n_b, ")");

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE, batch);
    ele_prod<<<grid_size, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), batch, m, n);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    a.masked_fill_(a == 0.0f, value);

    return a;
}

torch::Tensor softmax(torch::Tensor a) {
    CHECK_INPUT(a);
    const auto batch = a.size(0);
    const auto m = a.size(1);
    const auto n = a.size(2);

    int num_elements_per_thread = 8;
    int block_size = (n + num_elements_per_thread - 1) / num_elements_per_thread;
    dim3 grid_size(1, m, batch);
    int shared_mem_size = sizeof(float) * (n + block_size);
    softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(a.data_ptr<float>(), batch, m, n, num_elements_per_thread);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    return a;
}

/*
    Setup:  we have num_heads set of (q_weight, k_weight, v_weight) with dim (C, C / num_heads)
            we concatenate them to form 3 matrices of dim (C, C)
            we have a redundancy matrix proj with dim (C, C) to integrate the output from the multi head

    procedure:
        1. we have 3 matrix multiplications to get Q, K and V. (B, T, C) * (T, C) -> (B, T, C)
        2. we view Q, K, V as (B, T, num_heads, C / num_heads)
        3. we permute Q, K, V as (0, 2, 1, 3) -> (B, num_heads, T, C / num_heads)
        4. we then view Q, K, V as (B * num_heads, T, C / num_heads)
        5. we do a tensor multiplication of Q * K^T to get attention scores -> (B * num_heads, T, T)
        6. then we do fused softmax on attention scores -> (B * num_heads, T, T)
        7. we do a matrix multiplication on attention scores and V -> (B * num_heads, T, C / num_heads)
        8. we view the output as (B, num_heads, T, C / num_heads)
        9. We permute the outpus as (0, 2, 1, 3) -> (B, T, num_heads, C / num_heads)
        10. we view the output as (B, T, C)
        11. Another matrix mul of output and proj -> (B, T, C) * (C, C) = (B, T, C)
*/
torch::Tensor multi_head_attention_naive(
    torch::Tensor x, // the input batch (B, T, C)
    torch::Tensor q_weight, // concatenate query matrices (C, C)
    torch::Tensor k_weight, // concatenate of key matrices (C, C)
    torch::Tensor v_weight, // concatenate of value matrices (C, C)
    torch::Tensor proj, // redundancy matrix
    int num_heads
) {
    CHECK_INPUT(q_weight);
    CHECK_INPUT(k_weight);
    CHECK_INPUT(v_weight);
    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);
    // step 1
    auto Q = matmul(x, q_weight); // (B, T, C)
    auto K = matmul(x, k_weight); // (B, T, C)
    auto V = matmul(x, v_weight); // (B, T, C)

    // reshape Q, K, V
    Q = Q.reshape({B, T, num_heads, C / num_heads});
    K = K.reshape({B, T, num_heads, C / num_heads});
    V = V.reshape({B, T, num_heads, C / num_heads});

    // permute Q, K, V
    Q = Q.permute({0, 2, 1, 3}).contiguous(); // (B, num_heads, T, C / num_heads)
    K = K.permute({0, 2, 1, 3}).contiguous(); // (B, num_heads, T, C / num_heads)
    V = V.permute({0, 2, 1, 3}).contiguous(); // (B, num_heads, T, C / num_heads)

    // reshape Q, K, V
    Q = Q.reshape({B * num_heads, T, C / num_heads});
    K = K.reshape({B * num_heads, T, C / num_heads});
    V = V.reshape({B * num_heads, T, C / num_heads});

    // transpose K and do QK^T
    K = K.permute({0, 2, 1}).contiguous();
    auto attn = tensor_mul(Q, K); // (B * num_heads, T, T)

    // masking
    auto tril = torch::tril(torch::ones({T, T}, x.options()));
    attn = mask(attn, tril, -FLT_MAX);

    // fused softmax
    auto attn_norm = softmax(attn);

    // multiply with V
    auto out = tensor_mul(attn_norm, V); // (B * num_heads, T, C / num_heads)

    // reshape out
    out = out.reshape({B, num_heads, T, C / num_heads});

    // permute out
    out = out.permute({0, 2, 1, 3}).contiguous(); // (B, T, num_heads, C / num_heads);

    // reshape out
    out = out.reshape({B, T, C});

    // multiply with proj
    return matmul(out, proj);
}
