#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define WARP_SIZE 32

float cpu_benchmark(float *a, int M) {
    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        sum += a[i];
    }
    return sum;
}

/**
 * one good practice is to write a __device__ function for warp reduce, here I will write an industrial-level code for reduction
 */

// this is the reduce_sum function.
// You can draw the graph to see why it works
// one thing to note is, __shfl function has internal barrier such that, tid 1 cannot start iteration n while other tid haven't finished iter n-1 yet.
template <const uint width, typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = (width >> 1); offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset, width);
    }
    return val;
}

// block all reduce. Based on warp reduce, one can extend this idea to block reduce, the procedure is
// 1. do warp level reduce for all warps within the block
// 2. write the result of each warp into smem
// 3. 1 warp do the warp reduce on the result sum of each warp
// 4. now, each block's tid 0 has the block reduced sum, do atomic add to get the global sum.
template <typename T, const uint NUM_THREADS>
__global__ void sum(T *a, T* sum, int M) {
    // how many warps a block contains, which is also the size of the shared memory.
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ T s[NUM_WARPS];
    // local threadId to get the local laneid within a warp, and the warpid
    int tid = threadIdx.x;
    int laneid = tid % 32;
    int warpid = tid / 32;
    int idx = blockIdx.x * blockDim.x + tid;
    T val = (idx < M) ? a[idx] : T(0);

    // compute the reduced sum
    val = warp_reduce_sum<WARP_SIZE, T>(val);
    __syncthreads();

    // only the thread with laneid = 0 of this warp contains the warp level reduced sum
    if (laneid == 0) {
        s[warpid] = val;
    }
    __syncthreads();

    // let the 0^th warp do the reduction,
    // each lane read one local sum from smem
    // note such case that, there is less than 32 warps within a block can happen, i.e, laneid range may not be equal to the warpid range
    // therefore, it is important to have a branch for boundary check
    val = (laneid < NUM_WARPS) ? s[laneid] : T(0);
    if (warpid == 0) {
        val = warp_reduce_sum<WARP_SIZE, T>(val);
    }
    __syncthreads();

    // now, tid 0 of each block has the block sum, we can do atomic add to get the global sum
    if (threadIdx.x == 0) {
        atomicAdd(sum, val);
    }
}

void init_vec(float *a, int M) {
    for (int i = 0; i < M; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

int main() {
    int M = 16384;
    size_t size = M * sizeof(float);

    float *h_a, *h_a_res, *d_a;
    float *res;
    h_a = (float *)malloc(size);
    h_a_res = (float *)malloc(sizeof(float));
    cudaMalloc(&d_a, size);
    cudaMalloc(&res, sizeof(float));

    init_vec(h_a, M);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    const int thread_num = 128;
    dim3 block_size(thread_num, 1);
    dim3 grid_size((M + thread_num) / thread_num, 1);
    sum<float, thread_num><<<grid_size, block_size>>>(d_a, res, M);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(h_a_res, res, sizeof(float), cudaMemcpyDeviceToHost);

    float benchmark = cpu_benchmark(h_a, M);

    printf("benchmark: %f\t gpu: %f\n", benchmark, h_a_res[0]);
}

