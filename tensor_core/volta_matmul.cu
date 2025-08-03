// matmul_turing_wmma_half_flat.cu
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>

using namespace nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Each warp (32 threads) computes one 16×16 tile of C = A·B in half precision
__global__ void wmmaMatMulHalfFlat(const half* A,
                                   const half* B,
                                   half* C,
                                   int M, int N, int K)
{
    // compute the flattened warp id
    int global_warp_id = blockIdx.x * 32 + threadIdx.x / 32;

    int warps_per_row = N / WMMA_N;

    // organize warp into 2D layout
    int warp_row = global_warp_id / warps_per_row;
    int warp_col = global_warp_id % warps_per_row;

    // advance pointer
    // const half* tile_A = A + warp_row * WMMA_M * K;
    // const half* tile_B = B + warp_col * WMMA_N;
    // half*       tile_C = C + warp_row * WMMA_M * N + warp_col * WMMA_N;
    A += warp_row * WMMA_M * K;
    B += warp_col * WMMA_N;
    C += warp_row * WMMA_M * N + warp_col * WMMA_N;

    // WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Initialize to zero
    fill_fragment(c_frag, __float2half(0.0f));

    // Loop over K and compute
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        load_matrix_sync(a_frag, A, K);
        load_matrix_sync(b_frag, B, N);

        // adcance A and B to next tile
        A += WMMA_K;
        B += WMMA_K * N;

        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Write out
    store_matrix_sync(C, c_frag, N, mem_row_major);
}

// Host-side launcher
void wmmaMatMulHalfLauncher(const half* A, const half* B, half* C,
                            int M, int N, int K) {
    int tiles = (M/WMMA_M) * (N/WMMA_N);
    int warps_per_block = 32;
    int blocks = (tiles + warps_per_block - 1) / warps_per_block;

    dim3 blockDim(1024);
    dim3 gridDim(blocks);

    wmmaMatMulHalfFlat<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void init_mat(half *a, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        a[i] = __float2half((float) rand() / RAND_MAX);
    }
}

int main() {
    const int M = 4096;
    const int K = 8192;
    const int N = 2048;

    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(half);

    // Allocate host
    half *h_A = (half*)malloc(sizeA);
    half *h_B = (half*)malloc(sizeB);
    half *h_C = (half*)malloc(sizeC);
    // TODO: convert float data to half and store in h_A, h_B

    init_mat(h_A, M, K);
    init_mat(h_B, K, N);
    printf("%f\n", __half2float(h_A[0]));

    // Allocate device
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    wmmaMatMulHalfLauncher(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // TODO: convert h_C back to float or use directly
    printf("%f\n", __half2float(h_C[0]));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
