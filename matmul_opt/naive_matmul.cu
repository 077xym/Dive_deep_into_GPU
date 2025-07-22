#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define M 4096
#define K 4096
#define N 4096

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const uint BLOCK_SIZE>
__global__ void matmul(float *a, float *b, float *c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
}

void init_mat(float *a, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        a[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, M * K * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b, K * N * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&c, M * N * sizeof(float)));

    init_mat(a, M, K);
    init_mat(b, K, N);

    dim3 block_size(32, 32);
    dim3 grid_size(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    matmul<32><<<grid_size, block_size>>>(a, b, c, M, K, N);

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
}