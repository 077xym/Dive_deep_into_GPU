#include <cuda_runtime.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <float.h>
#include <vector>
#include <cublas_v2.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// In naive attention, we created our own matmul kernel. Although we applied tiling to improve the efficiency, the matmul kernel is
// far from being optimized compared to using cublas. So, in this better_attention, I plan to use cublas for all matmul tasks, and keep
// the fused softmax kernel, and optimize the fused softmax kernel for better performance.
__global__ void softmax_kernel(float *a, int batch, int m, int n, int num_elements_per_thread) {
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

__global__ void ele_prod(float *a, float *b, int batch, int m, int n) {
    int layer = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (layer < batch && row < m && col < n) {
        a[layer*m*n + row * n + col] *= b[row * n + col];
    }
}

torch::Tensor faster_tensor_mat_mul(
    torch::Tensor x,
    torch::Tensor W
) {
    CHECK_INPUT(x);
    CHECK_INPUT(W);

    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);
    int K = W.size(1);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    torch::Tensor out = torch::zeros({B, T, C}, x.options());

    /*
        Setup: x: (B, T, C), W: (C, K), out: (B, T, K)

        CUBLAS follow col-major mem layout, meaning in BLAS,
        x is viewed as (B, C, T)
        W is viewed as (K, C)
        To get a row major out, we want BLAS to return (B, K, T)

        (B, K, T) = (K, C) * (B, C, T)
    */
    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle, // session for one cublas call
        CUBLAS_OP_N, CUBLAS_OP_N, // operation of strided
        K, // row of matrix A, which is W in our case
        T, // col of matrix B, which is one stirde of x in our case
        C, // inner dimension
        &alpha,
        W.data_ptr<float>(), K, // leading dimension of W, which is K
        0, // offset(stride) of A, as q_weights is shared, no offset
        x.data_ptr<float>(), C, // leading dimension of x, which is C
        C * T, // offset of x, adjacent A[i] has C * T elements in between
        &beta,
        out.data_ptr<float>(), K, // leading dimension of one stride of out, which is C
        K * T, // offset of out, adjacent out[i] has K * T elements in between
        B // how many Batches
    ));

    CHECK_CUBLAS(cublasDestroy(handle));

    return out;
}

/*
    x: (B, T, C), W: (B, C, K), out -> (B, T, K)

    on BLAS
    x: (B, C, T), W: (B, K, C)

    we want our out to be (B, K, T), which is
    (B, K, C) * (B, C, T)
*/
torch::Tensor faster_tensor_mul(
    torch::Tensor x,
    torch::Tensor W
) {
    CHECK_INPUT(x);
    CHECK_INPUT(W);

    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);
    int K = W.size(2);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    torch::Tensor out = torch::zeros({B, T, K}, x.options());

    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle, // session for one cublas call
        CUBLAS_OP_N, CUBLAS_OP_N, // operation of strided
        K, // row of matrix A, which is W in our case
        T, // col of matrix B, which is one stirde of x in our case
        C, // inner dimension
        &alpha,
        W.data_ptr<float>(), K, // leading dimension of W, which is K
        K * C, // offset(stride) of A, which is K * C as adjacent W[i] has C * T in between
        x.data_ptr<float>(), C, // leading dimension of x, which is C
        C * T, // offset of x, adjacent x[i] has C * T elements in between
        &beta,
        out.data_ptr<float>(), K, // leading dimension of one stride of out, which is K
        K * T, // offset of out, adjacent out[i] has T * T elements in between
        B // how many Batches
    ));

    CHECK_CUBLAS(cublasDestroy(handle));

    return out;
}

torch::Tensor mask(torch::Tensor a, float value) {
    CHECK_INPUT(a);
    const auto batch = a.size(0);
    const auto m = a.size(1);
    const auto n = a.size(2);

    torch::Tensor b = torch::tril(torch::ones({m, n}, a.options()));
    b = b.masked_fill_(b == 0.0f, value);

    dim3 block_size(32, 32);
    dim3 grid_size((n + 32 - 1) / 32, (m + 32 - 1) / 32, batch);
    ele_prod<<<grid_size, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), batch, m, n);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    return a;
}

torch::Tensor softmax(torch::Tensor a) {
    CHECK_INPUT(a);
    const auto batch = a.size(0);
    const auto m = a.size(1);
    const auto n = a.size(2);

    int num_elements_per_thread = 16;
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
torch::Tensor multi_head_attention_better(
    torch::Tensor x, // the input batch (B, T, C)
    torch::Tensor q_weight, // concatenate query matrices (C, C)
    torch::Tensor k_weight, // concatenate of key matrices (C, C)
    torch::Tensor v_weight, // concatenate of value matrices (C, C)
    torch::Tensor proj, // redundancy matrix
    int num_heads
) {
    CHECK_INPUT(x);
    CHECK_INPUT(q_weight);
    CHECK_INPUT(k_weight);
    CHECK_INPUT(v_weight);
    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);
    // step 1, 3 matmul
    auto Q = faster_tensor_mat_mul(x, q_weight); // (B, T, C)
    auto K = faster_tensor_mat_mul(x, k_weight); // (B, T, C)
    auto V = faster_tensor_mat_mul(x, v_weight); // (B, T, C)

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
    auto attn = faster_tensor_mul(Q, K); // (B * num_heads, T, T)

    attn = mask(attn, -FLT_MAX);

    // fused softmax
    auto attn_norm = softmax(attn);

    // multiply with V
    auto out = faster_tensor_mul(attn_norm, V); // (B * num_heads, T, C / num_heads)

    // reshape out
    out = out.reshape({B, num_heads, T, C / num_heads});

    // permute out
    out = out.permute({0, 2, 1, 3}).contiguous(); // (B, T, num_heads, C / num_heads);

    // reshape out
    out = out.reshape({B, T, C});

    // multiply with proj
    return faster_tensor_mat_mul(out, proj);
}

