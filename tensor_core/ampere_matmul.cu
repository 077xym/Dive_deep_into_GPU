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
    // Claim shared memory for double buffering
    extern __shared__ half smem[];

    // Calculate shared memory layout
    auto int smem_a_size = WMMA_M * WMMA_K;
    auto int smem_b_size = WMMA_K * WMMA_N;

    // record the starting pointer address of current warp tile within the smem
    // we have 4 warp tiles from two consecutive iterations This is stored in register files
    half *smem_a[2] = {
        smem,
        smem + smem_a_size
    };

    half *smem_b[2] = {
        smem + 2 * smem_a_size,
        smem + 2 * smem_a_size + smem_b_size
    };

    // Compute the flattened warp id
    int global_warp_id = blockIdx.x * 32 + threadIdx.x / 32;
    int num_of_warps_per_row = N / WMMA_N;

    // Organize the warp into 2D layout
    int warp_row = global_warp_id / num_of_warps_per_row;
    int warp_col = global_warp_id % num_of_warps_per_row;

    // Advance pointer
    A += warp_row * WMMA_M * K;
    B += warp_col * WMMA_N;
    C += warp_row * WMMA_M * K + warp_col * WMMA_N;

    // initialize fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // initialize c_frag
    fill_fragment(c_frag, __float2half(0.0f));

    // Thread indexing. We want the local offset of a thread within a warp, and the trick is it is simply the modular over 32
    int lane_id = threadIdx.x % 32;

    // transform the flattend local offset into a 2D layout
    int lane_a_row = lane_id / WMMA_K;
    int lane_a_col = lane_id % WMMA_K;
    int lane_b_row = lane_id / WMMA_N;
    int lane_b_col = lane_id % WMMA_N;

    // elements per thread needs to load
    int elements_per_thread = WMMA_M * WMMA_K / 32;

    // double buffer index, for locating which location the computation loop will touch
    int buffer_idx = 0;

    // prefetch first tile
    for (int i = 0; i < elements_per_thread; i++) {
        int tile_row = lane_a_row * elements_per_thread + i;
        cp_async_cg(smem_a[buffer_idx] + tile_row * WMMA_K + lane_a_col.
                    A + tile_row * K + lane_a_col.
                    sizeof(half));
    }

    for (int i = 0; i < elements_per_thread; i++) {
        int tile_row = lane_b_row * elements_per_thread + i;
        cp_async_cg(smem_b[buffer_idx] + tile_row * WMMA_N + lane_a_col.
                    B + tile_row * N + lane_a_col.
                    sizeof(half));
    }

    // we need to commit the async processes for ordering bookkeeping
    cp_async_commit_group();

    // main loop
    for (int k = 0; k < K; k+=WMMA_K) {
        // load the next tile onto smem
        int next_buffer_idx = 1 - buffer_idx;
        // in case the last iteration
        if (k + WMMA_K < K) {
            for (int i = 0; i < elements_per_thread; i++) {
                int tile_row = lane_a_row * elements_per_thread + i
                cp_async_cg(smem_a[buffer_idx] + tile_row * WMMA_K + lane_a_col.
                            A + tile_row * K + lane_a_col.
                            sizeof(half));
            }

            for (int i = 0; i < elements_per_thread; i++) {
                int tile_row = lane_b_row * elements_per_thread + i
                cp_async_cg(smem_b[buffer_idx] + tile_row * WMMA_N + lane_a_col.
                            B + tile_row * N + lane_a_col.
                            sizeof(half));
            }
            cp_async_commit_group();
        }

        // make sure the current loads has finished
        cp_async_wait_group(1);
        __syncthreads();

        // Load fragments from shared memory
        load_matrix_sync(a_frag, smem_a[buffer_idx], WMMA_K);
        load_matrix_sync(b_frag, smem_b[buffer_idx], WMMA_N);

        // Perform matrix multiplication
        mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Advance pointers for next iteration
        A += WMMA_K;
        B += WMMA_K * N;

        // Swap buffers
        buffer_idx = next_buffer_idx;

        __syncthreads();
    }

    // wait for all async operations to complete
    cp_async_wait_all();

    // Write result back to global memory
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
