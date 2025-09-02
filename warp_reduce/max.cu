#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WARP_SIZE 32

float cpu_benchmark(float *a, int M) {
    float max = -INFINITY;
    for (int i = 0; i < M; i++) {
        max = fmaxf(max, a[i]);
    }
    return max;
}

/**
 * this code snippet shows how reduction max can be implemented
 * the basic logic is rather similar with reduction sum
 */

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset, WARP_SIZE));
    }
    return val;
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// block all reduce to get max
// 1. do warp reduce max on each warp
// 2. write the warp max to smem
// 3. use one warp of threads to do another warp reduce max on all the warp maxes
// 4. tid 0 of each block contains the block max, we need to do an atomic fmaxf (atomicMax)
template <const uint NUM_THREADS>
__global__ void max(float *a, float *res, int M) {
    // get total warps within a block
    constexpr int WARP_NUM = (NUM_THREADS + 31) / 32;
    // initialize smem for each block
    __shared__ float s[WARP_NUM];
    // get the local thread index within a block
    int tid = threadIdx.x;
    // get the local thread index within a warp
    int laneid = tid % 32;
    // get the warp this thread is located
    int warpid = tid / 32;
    // global index this thread will access to a
    int idx = blockIdx.x * blockDim.x + tid;

    // each thread accesses the data (thread max)
    float max = (idx < M) ? a[idx] : 0.0f;

    // do the warp reduce max (warp max)
    max = warp_reduce_max(max);
    __syncthreads();

    // now each laneid = 0 gets the warp max, set to smem
    if (laneid == 0) {
        s[warpid] = max;
    }
    __syncthreads();

    // now the smem contains the warp max of each warp, do another warp reduce on these data
    if (warpid == 0) {
        max = (laneid < WARP_NUM) ? s[laneid] : 0.0f;
        max = warp_reduce_max(max);
    }
    __syncthreads();

    // now tid = 0 of each block contains the block max, we need to do another atomicMax to get the global max
    if (tid == 0) {
        atomicMax(res, max);
    }
}

void init_mat(float *a, int M) {
    for (int i = 0; i < M; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

int main() {
    int M = 16384;
    size_t size = M * sizeof(float);
    float *h_a, *d_a, *d_res;


    // allocate memory
    h_a = (float *)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_res, sizeof(float));

    // init mat
    init_mat(h_a, M);

    // copy to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // launch kernel
    const int num_threads = 1024;
    dim3 block_size(num_threads, 1);
    dim3 grid_size((M + num_threads - 1) / num_threads);
    max<num_threads><<<grid_size, block_size>>>(d_a, d_res, M);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    float h_res = 0.0f;
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    float benchmark = cpu_benchmark(h_a, M);

    printf("benchmark: %f, gpu: %f\n", benchmark, h_res);

    free(h_a);
    cudaFree(d_a);
    cudaFree(d_res);
}
