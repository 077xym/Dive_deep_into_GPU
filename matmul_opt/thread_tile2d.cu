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

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void matmul_bl2d(float *a, float *b, float *c, int m, int k, int n) {
    const uint by = blockIdx.y;
    const uint bx = blockIdx.x;

    // indices of the thread in the block of the output c
    const uint c_row = threadIdx.x / (BN / TM);
    const uint c_col = threadIdx.x % (BN / TM);

    const uint num_threads = BM * BN / (TM * TM);

    // allocate shared memory
    __shared__ float shared_A[BM * BK];
    __shared__ float shared_B[BK * BN];

    // move block ptr
    a += by * BM * k;
    b += bx * BN;
    c += by * BM * k + bx * BN;

    // get local indices of the threads within tile a and tile b
    const uint a_row = threadIdx.x / BK;
    const uint a_col = threadIdx.x % BK;
    const uint a_stride = num_threads / BK;
    const uint b_row = threadIdx.x / BN;
    const uint b_col = threadIdx.x % BN;
    const uint b_stride = num_threads / BN;

    // use registers to store row and col
    float thread_res[TM * TM] = {0.0f};
    float thread_r[TM] = {0.0f};
    float thread_c[TM] = {0.0f};

    // we load BM / a_stride elements from a, and BK / b_stride elements from b per thread
    for (int blkId = 0; blkId < k; blkId += BK) {
        for (uint offset = 0; offset < BM; offset += a_stride) {
            shared_A[(a_row + offset) * BK + a_col] = a[(a_row + offset) * k + a_col];
        }
        for (uint offset = 0; offset < BK; offset += b_stride) {
            shared_B[(b_row + offset) * BN + b_col] = b[(b_col + offset) * n + b_col];
        }

        __syncthreads();

        a += BK;
        b += BK * n;

        // we load one col of length TM from shared_A, and one row of length TM from shared_B
        for (int dotId = 0; dotId < BK; dotId++) {
            for (int i = 0; i < TM; i++) {
                thread_r[i] = shared_A[(c_row * TM + i) * BK + dotId];
            }
            for (int i = 0; i < TM; i++) {
                thread_c[i] = shared_B[dotId * BN + c_col * TM + i];
            }
            for (uint res_r = 0; res_r < TM; res_r++) {
                for (uint res_c = 0; res_c < TM; res_c++) {
                    thread_res[res_r * TM + res_c] += thread_r[res_r] * thread_c[res_c];
                }
            }
            __syncthreads();
        }
    }

    for (uint res_r = 0; res_r < TM; res_r++) {
        for (uint res_c = 0; res_c < TM; res_c++) {
            c[(c_row * TM + res_r) * N + c_col * TM + res_c] = thread_res[res_r * TM + res_c];
        }
    }
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

    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 2;
    const uint TM = 8;

    dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block_size(BM * BN / (TM * TM));

    matmul_bl2d<BM, BN, BK, TM><<<grid_size, block_size>>>(a, b, c, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
}
