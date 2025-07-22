#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define M 4096
#define K 4096
#define N 4096

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
    uint block_size = 1024;
    dim3 grid_size(CEIL_DIV(N, 32), CEIL_DIV(M, 32))
*/

template <const int TILE_SIZE>
__global__ void matmul_tile(float *a, float *b, float *c, int m, int k, int n) {
    const uint by = blockIdx.y;
    const uint bx = blockIdx.x;

    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    const uint ty = threadIdx.y;
    const uint tx = threadIdx.x;

    // advance A, B, C to corresponding tiles, such that we then only need to consider about the local index of threads
    a += by * TILE_SIZE * k;
    b += bx * TILE_SIZE;
    c += by * TILE_SIZE * n + bx * TILE_SIZE;

    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        sharedA[ty][tx] = a[ty * K + tx];
        sharedB[ty][tx] = b[ty * N + tx];

        __syncthreads();

        // advance A, B again
        a += TILE_SIZE;
        b += TILE_SIZE * N;

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += sharedA[ty][i] * sharedB[i][tx];
        }

        __syncthreads();
    }

    c[ty * N + tx] = sum;
}

template <const uint BLOCK_SIZE>
__global__ void matmul(float *a, float *b, float *c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.x / BLOCK_SIZE;
    int col = blockIdx.x * blockDim.x + threadIdx.x % BLOCK_SIZE;

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
    matmul_tile<32><<<grid_size, block_size>>>(a, b, c, M, K, N);

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
}