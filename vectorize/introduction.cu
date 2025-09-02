#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// to avoid writing reinterpret_cast again and again, we can make it as a macro
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

/**
 * vectorized load/store is a common technique for optimization.
 * In this document, I am going to briefly introduce the semantics
 */

void init_mat(float *a, int M) {
    for (int i = 0; i < M; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

bool validate(float *a, float *b, int M) {
    for (int i = 0; i < M; i++) {
        if (fabs(a[i]-b[i]) > 1e-2) {
            printf("different values at (%d), benchmark: %f, gpu: %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

// each thread is responsible for transfering 4 consecutive elements from a to b
// this is GMEM -> reg, reg -> GMEM
__global__ void vec4_io(float *a, float *b, int M) {
    // local thread index within a block
    int tid = threadIdx.x;
    // global index
    int idx = 4 * (blockIdx.x * blockDim.x + tid);
    if (idx < M) {
        // the semantics here is
        // a[idx] is the starting value
        // we use &a[idx] to extract the starting address of vec4 load
        // we shall also use reinterpret_cast to cast the type to float4 *, such that 4 elements will be loaded
        // then we index into the 0th element to get the actual data that we loaded
        float4 reg_a = reinterpret_cast<float4 *>(&a[idx])[0];
        float4 reg_b;
        reg_b.x = reg_a.x;
        reg_b.y = reg_a.y;
        reg_b.z = reg_a.z;
        reg_b.w = reg_a.w;
        // same idea but a store process
        reinterpret_cast<float4 *>(&b[idx])[0] = reg_b;
    }
}

// we can also load from GMEM to smem, and store from smem to GMEM
template <const int NUM_THREADS>
__global__ void vec4_io_smem(float *a, float *b, int M) {
    __shared__ float s[NUM_THREADS * 4];
    int tid = threadIdx.x;
    int idx = 4 * (blockIdx.x * blockDim.x + tid);
    int smem_idx = 4 * tid;
    if (idx < M) {
        reinterpret_cast<float4 *>(&s[smem_idx])[0] = reinterpret_cast<float4 *>(&a[idx])[0];
        __syncthreads();
        reinterpret_cast<float4 *>(&b[idx])[0] = reinterpret_cast<float4 *>(&s[smem_idx])[0];
    }
}

int main() {
    int M = 16384;
    size_t size = M * sizeof(float);

    float *h_a, *h_b, *d_a, *d_b;
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    init_mat(h_a, M);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    const uint num_threads = 128;
    dim3 block_size(num_threads, 1);
    dim3 grid_size((M + num_threads * 4 - 1) / (num_threads * 4));
    vec4_io_smem<num_threads><<<grid_size, block_size>>>(d_a, d_b, M);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    bool correct = validate(h_a, h_b, M);
    printf("validate result: %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
}
