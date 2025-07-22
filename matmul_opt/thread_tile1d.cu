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

template <const int BM, const int BN, const int BK, const int TM>
__global__ void matmul_bl(float *a, float *b, float *c, int m, int k, int n) {
    const uint by = blockIdx.y;
    const uint bx = blockIdx.x;

    // the thread idx in the current block of output c
    const uint c_row = threadIdx.x / BN;
    const uint c_col = threadIdx.x % BN;

    __shared__ float sharedA[BM * BK];
    __shared__ float sharedB[BK * BN];

    // move block ptr such that the rest logic can be done on local indices
    a += by * BM * k;
    b += bx * BN;
    c += by * BM * k + bx * BN;

    // check the requirement
    assert(BM == BN);
    assert(BM * BK == blockDim.x);

    // compute the local indices of each thread in tile A and tile B
    const uint a_row = threadIdx.x / BK;
    const uint a_col = threadIdx.x % BK;
    const uint b_row = threadIdx.x / BN;
    const uint b_col = threadIdx.x % BN;

    // allocate cache for TM results in registerfile
    float thread_res[TM] = {0.0f};

    // outer loop is the tile movement along inner dimension
    for (int blkId = 0; blkId < k; blkId += BK) {
        // load element from a and b
        sharedA[a_row * BK + a_col] = a[a_row * k + a_col];
        sharedB[b_row * BN + b_col] = b[b_row * n + b_col];

        __syncthreads();

        // move a and b pointer
        a += BK;
        b += BK * n;

        /*
            here, we begin to calculate the partial sum of each tile a and b pair
            the usually is to put TM as outer loop, and BK as inner loop, i,e
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < BK; j++) {
                    thread_res[i] += element from shared_a and from shared_b
                }
            }
            this will cause extra read from shared_b. One clever way is to make BK as the outer row, and for each
            outer loop, we only need to read 1 element from shared_b for the inner loop

            But the current compiler is smart enough such that for both ways, it will optimize the access times on shared b
        */
        for (int i = 0; i < BK; i++) {
            float tmp = sharedB[i * BN + c_col];
            for (int j = 0; j < TM; j++) {
                thread_res[j] += sharedA[(c_row * TM + j) * BK + i] * tmp;
            }
        }
        __syncthreads();
    }
    // put the results into c
    for (int i = 0; i < TM; i++) {
        c[(c_row * TM + i) * n + c_col] = thread_res[i];
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

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 4;
    const uint TM = 16;

    dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block_size(BM * BN / TM);

    matmul_bl<BM, BN, BK, TM><<<grid_size, block_size>>>(a, b, c, M, K, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
}