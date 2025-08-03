/*
  hopper_matmul.cu  — single-warp-group GEMM kernel for H100 (SM90a)
  =================================================================
  Build:
    nvcc -arch=sm_90a -O3 -lineinfo hopper_matmul.cu -o hopper_gemm

  Features:
    • Row-major A (M×K), B (K×N), C (M×N)
    • BLOCK = 128×128×32 tile per CTA, 128 threads (4 warps)
    • Double-buffered TMA loads for overlap
    • mbarrier for async handshake, __syncthreads for intra-CTA safety
    • wgmma.mma_async m64n64k16 atoms ×4 to build 128×128
    • tma.store async at epilogue
    • All inline-PTX uses smem_addr() to pass 32-bit SMEM addresses
*/

#include <cuda_fp16.h>
#include <stdint.h>

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;
constexpr int WG_SIZE  = 128;   // # threads issuing TMA loads

// Shared-memory slab
extern __shared__ uint8_t s_mem[];  // ← semicolon added
#define S_A(buf)  reinterpret_cast<half*>(s_mem + (buf)*(BLOCK_M*BLOCK_K*sizeof(half)))
#define S_B(buf)  reinterpret_cast<half*>(s_mem + 2*(BLOCK_M*BLOCK_K*sizeof(half)) + (buf)*(BLOCK_K*BLOCK_N*sizeof(half)))
#define S_C       reinterpret_cast<float*>(s_mem + 2*(BLOCK_M*BLOCK_K*sizeof(half)) + 2*(BLOCK_K*BLOCK_N*sizeof(half)))
#define S_BAR(i)  (*reinterpret_cast<uint64_t*>(s_mem \
                    + 2*(BLOCK_M*BLOCK_K*sizeof(half)) \
                    + 2*(BLOCK_K*BLOCK_N*sizeof(half)) \
                    + BLOCK_M*BLOCK_N*sizeof(float) \
                    + (i)*8))

#define MBARRIER_INIT(bar,c)   asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(&(bar)), "n"(c))
#define MBARRIER_WAIT(bar,c)   asm volatile("mbarrier.wait.shared.b64 [%0], %1;" :: "r"(&(bar)), "r"(c))

// Convert generic pointer to 32-bit shared-memory address
__device__ inline uint32_t smem_addr(const void* p) {
    return __cvta_generic_to_shared(p);
}

// Build a TMA descriptor for a 2D tile
__device__ void init_tma_desc(uint64_t &desc,
                              const void* base, int stride_bytes,
                              int rows, int cols) {
    asm volatile("tma.init.const.desc_2d.b16 %0, %1, %2, %3, %4;"
                 : "=l"(desc)
                 : "r"(base), "r"(stride_bytes), "n"(rows), "n"(cols));
}

// GEMM kernel
__global__ void hopper_gemm_tma(const half* __restrict__ A,
                                const half* __restrict__ B,
                                float*      __restrict__ C,
                                int M, int N, int K) {
    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    // allocate barriers in shared mem
    uint64_t &bar0 = S_BAR(0);
    uint64_t &bar1 = S_BAR(1);
    if (tid == 0) {
        MBARRIER_INIT(bar0, 0);
        MBARRIER_INIT(bar1, 0);
    }
    __syncthreads();

    // compute tile origin
    int bm = blockIdx.y;
    int bn = blockIdx.x;

    half* gA = const_cast<half*>(A) + bm*BLOCK_M*K;
    half* gB = const_cast<half*>(B) + bn*BLOCK_N;
    float* gC = C + bm*BLOCK_M*N + bn*BLOCK_N;

    // shared TMA descriptors (ping-pong)
    __shared__ __align__(16) uint64_t tma_a_desc[2];
    __shared__ __align__(16) uint64_t tma_b_desc[2];
    __shared__ __align__(16) uint64_t tma_c_desc;
    if (tid == 0) {
        init_tma_desc(tma_a_desc[0], gA,           K*sizeof(half), BLOCK_M, BLOCK_K);
        init_tma_desc(tma_b_desc[0], gB,           N*sizeof(half), BLOCK_K, BLOCK_N);
        init_tma_desc(tma_a_desc[1], gA + BLOCK_K, K*sizeof(half), BLOCK_M, BLOCK_K);
        init_tma_desc(tma_b_desc[1], gB + BLOCK_K*N, N*sizeof(half), BLOCK_K, BLOCK_N);
        init_tma_desc(tma_c_desc,    gC,           N*sizeof(float), BLOCK_M, BLOCK_N);
    }
    __syncthreads();

    // accumulator fragment per-thread
    float c_frag[8] = {0.f};
    int read_buf = 0, write_buf = 1;

    // initial prefetch (slice 0)
    if (tid < WG_SIZE) {
        asm volatile(
            "tma.ld.global.mbarrier::async.aligned.shared::cluster [%0], [%1], [%2];"
            :: "r"(smem_addr(S_A(read_buf))),
               "r"(tma_a_desc[read_buf]),
               "r"(bar0));
        asm volatile(
            "tma.ld.global.mbarrier::async.aligned.shared::cluster [%0], [%1], [%2];"
            :: "r"(smem_addr(S_B(read_buf))),
               "r"(tma_b_desc[read_buf]),
               "r"(bar0));
    }
    MBARRIER_WAIT(bar0, 0);
    __syncthreads();

    // main K-loop
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        bool has_next = (k0 + BLOCK_K < K);

        // update descriptors for next slice
        if (has_next && tid == 0) {
            gA += BLOCK_K;
            gB += BLOCK_K*N;
            init_tma_desc(tma_a_desc[write_buf], gA,           K*sizeof(half), BLOCK_M, BLOCK_K);
            init_tma_desc(tma_b_desc[write_buf], gB,           N*sizeof(half), BLOCK_K, BLOCK_N);
        }
        if (has_next) __syncthreads();

        // launch async load next
        if (has_next && tid < WG_SIZE) {
            asm volatile(
                "tma.ld.global.mbarrier::async.aligned.shared::cluster [%0], [%1], [%2];"
                :: "r"(smem_addr(S_A(write_buf))),
                   "r"(tma_a_desc[write_buf]),
                   "r"(bar1));
            asm volatile(
                "tma.ld.global.mbarrier::async.aligned.shared::cluster [%0], [%1], [%2];"
                :: "r"(smem_addr(S_B(write_buf))),
                   "r"(tma_b_desc[write_buf]),
                   "r"(bar1));
        }

        // compute 4×(64×64×16) atoms
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += 16)
#pragma unroll
            for (int mi = 0; mi < BLOCK_M; mi += 64)
#pragma unroll
                for (int nj = 0; nj < BLOCK_N; nj += 64) {
                    half const* a_tile = S_A(read_buf) + mi*BLOCK_K + kk;
                    half const* b_tile = S_B(read_buf) + kk*BLOCK_N + nj;
                    asm volatile(
                        "wgmma.mma_async.sync.aligned.m64n64k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, [%4], [%5], {%0,%1,%2,%3};"
                        : "+f"(c_frag[0]), "+f"(c_frag[1]),
                          "+f"(c_frag[2]), "+f"(c_frag[3])
                        : "r"(a_tile), "r"(b_tile));
                }

        // swap ping-pong
        read_buf ^= 1;
        write_buf ^= 1;

        if (has_next) {
            MBARRIER_WAIT(bar1, 0);
            __syncthreads();
        }
    }

    // epilogue: stmatrix + async store
    if (tid == 0) MBARRIER_INIT(bar0, 0);
    __syncthreads();
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        asm volatile(
            "stmatrix.sync.aligned.m16n16.x4.shared.f32 [%0], {%1,%2,%3,%4};"
            :: "r"(smem_addr(S_C
                   + ((warp*32 + (t>>1)*16)*BLOCK_N)
                   + ((t&1)*16 + ((lane&7)*2)))),
               "f"(c_frag[0]), "f"(c_frag[1]),
               "f"(c_frag[2]), "f"(c_frag[3]));
    }
    __syncthreads();
    if (tid < WG_SIZE) {
        asm volatile(
            "tma.store.mbarrier::async.shared::cluster [%0], [%1], [%2];"
            :: "l"(gC), "r"(tma_c_desc), "r"(smem_addr(&bar0)));
    }
    MBARRIER_WAIT(bar0, 0);
}
