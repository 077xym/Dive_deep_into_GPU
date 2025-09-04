#include <cuda.h>              // Driver API for CUtensorMap
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>             // rand, srand
#include <ctime>

extern "C" __global__
void mbarrier_sanity() {
    // create shared mem containing mbar object and mbar token
    extern __shared__ __align__(8) uint64_t smem[];
    // mbarrier objects have 8 bytes
    auto *mbar_obj = smem;
    auto *shared_token = smem+8;
    // convert the address of mbar_obj to a .shared address
    const unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(mbar_obj);

    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;\n\t"
            : : "r"(mbar_addr), "r"(1) : "memory"
        );
    }
    __syncthreads();

    // phase token
    if (threadIdx.x == 0) {
        unsigned long long token = 0u;
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n\t"
            : "=l"(token) : "r"(mbar_addr), "r"(16) : "memory"
        );
        *shared_token = token;
        asm volatile(
            "mbarrier.complete_tx.shared::cta.b64 [%0], %1;\n\t"
            : : "r"(mbar_addr), "r"(16) : "memory"
        );
    }

    unsigned long long token = *shared_token;
    unsigned done = 0U;
    do {
        asm volatile(
        "{ .reg .pred p;                              \n\t"
            "  mbarrier.test_wait.shared::cta.b64 p, [%1], %2; \n\t"
            "  selp.u32 %0, 1, 0, p;                     \n\t"
        "}                                           \n\t"
        : "=r"(done) : "r"(mbar_addr), "l"(token) : "memory");
    } while (!done);

    __syncthreads();
}

int main() {
    dim3 block_size(128, 1, 1);
    dim3 grid_size(3, 1, 1);
    size_t shmem = 16;
    mbarrier_sanity<<<grid_size, block_size, shmem>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error %s:%d: %s (%d)\n",
            __FILE__, __LINE__, cudaGetErrorString(e), (int)e);
        exit(EXIT_FAILURE);
    }
}