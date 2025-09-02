#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>

#define NUM_OF_ITERS 5

extern "C" __global__
void producer_consumer_parity() {
    // Shared memory for barriers only - no need for tokens!
    __shared__ __align__(8) uint64_t arrive;
    __shared__ __align__(8) uint64_t finish;

    const unsigned arrive_addr = (unsigned)__cvta_generic_to_shared(&arrive);
    const unsigned finish_addr = (unsigned)__cvta_generic_to_shared(&finish);

    int warp_id = threadIdx.x / 32;

    // Initialize barriers - both start at phase 0
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;\n\t"
            : : "r"(arrive_addr), "r"(1) : "memory"
        );
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;\n\t"
            : : "r"(finish_addr), "r"(1) : "memory"
        );
    }

    __syncthreads();

    // Producer and Consumer loops
    for (int i = 0; i < NUM_OF_ITERS; i++) {

        // Producer warp
        if (warp_id == 0 && threadIdx.x == 0) {
            // Producer waits for consumer to finish previous iteration
            if (i > 0) {
                // Calculate expected parity for consumer's finish from iteration i-1
                unsigned expected_finish_parity = (i - 1) & 1;

                printf("Producer iter %d: waiting\n", i);

                // Wait for consumer to finish previous iteration
                asm volatile(
                    "{\n"
                    ".reg .pred                P1;\n"
                    "LAB_WAIT:\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                    "@P1                       bra.uni DONE;\n"
                    "bra.uni                   LAB_WAIT;\n"
                    "DONE:\n"
                    "}\n"
                    :: "r"(finish_addr),
                    "r"(expected_finish_parity)
                );
            }

            printf("Producer iter %d: start\n", i);

            __nanosleep(50000000); // 50ms work simulation

            // Producer signals arrival (this will flip arrive barrier's phase)
            asm volatile(
                "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
                : : "r"(arrive_addr) : "memory"
            );
            printf("Producer iter %d: finished\n", i);
        }

        // Consumer warp
        if (warp_id == 1 && threadIdx.x == 32) {
            // Calculate expected parity for producer's arrival from current iteration
            unsigned expected_arrive_parity = i & 1;

            printf("Consumer iter %d: waiting\n", i);

            // Wait for producer arrival from current iteration
            asm volatile(
                    "{\n"
                    ".reg .pred                P1;\n"
                    "LAB_WAIT:\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                    "@P1                       bra.uni DONE;\n"
                    "bra.uni                   LAB_WAIT;\n"
                    "DONE:\n"
                    "}\n"
                    :: "r"(arrive_addr),
                    "r"(expected_arrive_parity)
            );

            printf("Consumer iter %d: started\n", i);

            __nanosleep(50000000); // 50ms work simulation

            asm volatile(
                "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
                : : "r"(finish_addr) : "memory"
            );
            printf("Consumer iter %d: finished\n", i);
        }
    }
}

int main() {
    producer_consumer_parity<<<1, 64>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }
}
