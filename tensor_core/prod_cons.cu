#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>

#define NUM_OF_ITERS 10

extern "C" __global__
void producer_consumer_sanity() {
    // create shared mem containing arrive and finish mbarrier
    __shared__ __align__(8) uint64_t arrive;
    __shared__ __align__(8) uint64_t finish;
    __shared__ __align__(8) uint64_t token_arrive;
    __shared__ __align__(8) uint64_t token_finish;
    const unsigned arrive_addr = (unsigned)__cvta_generic_to_shared(&arrive);
    const unsigned finish_addr = (unsigned)__cvta_generic_to_shared(&finish);

    int warp_id = threadIdx.x / 32;

    // initialize the mbarrier
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;\n\t"
            : : "r"(arrive_addr), "r"(1) : "memory"
        );
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;\n\t"
            : : "r"(finish_addr), "r"(1) : "memory"
        );

        uint64_t token = 0u;
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n\t"
            : "=l"(token) : "r"(arrive_addr), "r"(16) : "memory"
        );
        token_arrive = token;
        asm volatile(
            "mbarrier.complete_tx.shared::cta.b64 [%0], %1;\n\t"
            : : "r"(arrive_addr), "r"(16) : "memory"
        );
        printf("done 0 load\n");
        token_finish = 0xFFFFFFFFFFFFFFFF;
        token_arrive = 0;

    }

    __syncthreads();

    // 2 warps within a block, warp 0: producer; warp 1: consumer
    if (warp_id == 0) {
        for (int i = 1; i < NUM_OF_ITERS; i++) {
            if (threadIdx.x == 0) {
                uint64_t state_token_finish = token_finish;
                unsigned done = 0U;
                // wait on finish, but skip the first iter

                do {
                    asm volatile(
                        "{ .reg .pred p;            \n\t"
                        "   mbarrier.test_wait.shared::cta.b64 p, [%1], %2; \n\t"
                        "   selp.u32 %0, 1, 0, p;   \n\t"
                        "}              \n\t"
                        : "=r"(done) : "r"(finish_addr), "l"(state_token_finish) : "memory"
                    );
                } while (!done);

                printf("begin %d load\n", i);

                uint64_t token = 0u;
                asm volatile(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n\t"
                    : "=l"(token) : "r"(arrive_addr), "r"(16) : "memory"
                );
                token_arrive = token;

                asm volatile(
                    "mbarrier.complete_tx.shared::cta.b64 [%0], %1;\n\t"
                    : : "r"(arrive_addr), "r"(16) : "memory"
                );

                printf("done %d load\n", i);

            }
        }
        __syncthreads();

    } else {
        if (threadIdx.x == 32) {
            for (int i = 0; i < NUM_OF_ITERS; i++) {
                uint64_t state_token_arrive = token_arrive;
                unsigned done = 0U;
                // wait on arrive
                do {
                    asm volatile(
                        "{ .reg .pred p;            \n\t"
                        "   mbarrier.test_wait.shared::cta.b64 p, [%1], %2; \n\t"
                        "   selp.u32 %0, 1, 0, p;   \n\t"
                        "}              \n\t"
                        : "=r"(done) : "r"(arrive_addr), "l"(state_token_arrive) : "memory"
                    );
                } while (!done);
                printf("Consumer iter %d: starting, will set token_finish\n", i);

                // finish
                uint64_t token = 0u;
                asm volatile(
                    "mbarrier.arrive.shared.b64 %0, [%1];\n\t"
                    : "=l"(token) : "r"(finish_addr) : "memory"
                );
                token_finish = token;
                printf("Consumer iter %d: finished, set token_finish = 0x%lx\n", i, token);
            }
        }

        __syncthreads();
    }
}

int main() {
    dim3 block_size(64, 1, 1);
    dim3 grid_size(1, 1, 1);
    producer_consumer_sanity<<<grid_size, block_size>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error %s:%d: %s (%d)\n",
            __FILE__, __LINE__, cudaGetErrorString(e), (int)e);
        exit(EXIT_FAILURE);
    }
}
