#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define WARP_SIZE 32

/**
 * One really common kernel that applies warp reduce is dot product kernel
 */
float dot_cpu(float *a, float *b, int M) {
    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}


template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset, WARP_SIZE);
    }
    return val;
}

template <typename T, const int NUM_THREADS>
__global__ void dot_kernel(T *a, T *b, T *res, int M) {
    // how many warps within a block
    constexpr int NUM_WARPS = (NUM_THREADS + 31) / 32;
    // initiates smem
    __shared__ T s[NUM_WARPS];
    // thread local index within the block
    int tid = threadIdx.x;
    // thread local index within the warp
    int laneid = tid % 32;
    // warp id this thread locates
    int warpid = tid / 32;
    // global idx of the thread denoting the index it will access to a and b
    int idx = blockIdx.x * blockDim.x + tid;

    // get the value from a and b and multiply
    T val = (idx < M) ? a[idx] * b[idx] : T(0);

    // do reduce sum to get the warp sum (thread sum)
    val = warp_reduce_sum<T>(val);
    __syncthreads();

    // set the warp reduced sum to smem (warp sum)
    if (laneid == 0) {
        s[warpid] = val;
    }
    __syncthreads();

    // use warp 0 to do another warp reduce on the warp sums (block sum)
    val = (laneid < NUM_WARPS) ? s[laneid] : T(0);
    if (warpid == 0) {
        val = warp_reduce_sum<T>(val);
    }
    __syncthreads();

    // now tid = 0 within each block has the block reduced sum, do attomic add to get the global result
    if (tid == 0) {
        atomicAdd(res, val);
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

    float *h_a, *h_b, *d_a, *d_b, *d_res;
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_res, sizeof(float));

    init_mat(h_a, M);
    init_mat(h_b, M);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // launch kernel
    const int num_threads = 1024;
    dim3 block_size(num_threads, 1);
    dim3 grid_size((M + num_threads - 1) / num_threads);
    dot_kernel<float, num_threads><<<grid_size, block_size>>>(d_a, d_b, d_res, M);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    float h_res = 0.0f;
    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    float benchmark = dot_cpu(h_a, h_b, M);

    printf("benchmark: %f\t gpu: %f\n", benchmark, h_res);

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
}
