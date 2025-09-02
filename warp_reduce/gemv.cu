#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define WARP_SIZE 32

/**
 * Matrix vector multiplication is another important application related to warp reduce.
 * This is more complicated than expected, especially when we want to think about the optimization
 */

void gemv_cpu(float *a, float *b, float *c, int M, int K) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            sum += a[i * K + j] * b[j];
        }
        c[i] = sum;
    }
}

bool validate(float *a, float *b, int M) {
    for (int i = 0; i < M; i++) {
        if (fabs(a[i] - b[i]) > 1e-2) {
            printf("different value detected at (%d): benchmark: %f\t gpu: %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

void init_mat(float *a, int M, int K) {
    for (int i = 0; i < M * K; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

void print_mat(float *a, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f\t", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset, WARP_SIZE);
    }
    return val;
}

// my first idea is to let each block computes one element
// when load from matrix and vector, we want each thread to load with a stride num_threads, such that within one warp,
// all threads will load a consecutive chunk of data -> mem coalescing.
// The procedure is
// 1. each thread iteratively load the data with a stride
// 2. record the thread sum
// 3. do the warp reduce
// 4. set the warp sum into smem
// 5. use one warp to do the block reduce sum
// 6. set the element to corresponding index on c
template <typename T, const uint NUM_THREADS>
__global__ void gemv_block_row(T *a, T *b, T *c, int M, int K) {
    // how many warps within a block
    constexpr int NUM_WARPS = (NUM_THREADS + 31) / 32;
    // initiates the smem
    __shared__ T s[NUM_WARPS];
    // local thread idx within a block
    int tid = threadIdx.x;
    // local thread idx within a warp
    int laneid = tid % 32;
    // warp id
    int warpid = tid / 32;
    // block id, which row this block is responsible for
    int bid = blockIdx.y;

    // iteratively load data from GMEM (thread sum)
    T sum = T(0);
    #pragma unroll
    for (int i = threadIdx.x; i < K; i += NUM_THREADS) {
        // the thread will load from matrix with index (bid, i)
        // and the thread will load from vector with index (i, 1)
        sum += a[bid * K + i] * b[i];
    }

    // we get the sum of each thread, we can then do the warp reduce (warp sum)
    sum = warp_reduce_sum<T>(sum);
    __syncthreads();

    // after we get the warp sum of each warp, set the sum to smem
    if (laneid == 0) {
        s[warpid] = sum;
    }
    __syncthreads();

    // do the block reduce sum (block sum)
    sum = (laneid < NUM_WARPS) ? s[laneid] : T(0);
    if (warpid == 0) {
        sum = warp_reduce_sum<T>(sum);
    }
    __syncthreads();

    // now the tid = 0 thread contains the block reduced sum of each block, set to c
    if (tid == 0) {
        c[bid] = sum;
    }
}

// let's make one warp to do more things, i.e, one warp is responsible for one row
// in such a case, we don't have to use smem for block reduce
// the procedure is:
// 1. iteratively load from a and b with a stride WARP_SIZE
// 2. record the thread sum
// 3. do warp reduce to get the sum
// 4. set to c
template <typename T>
__global__ void gemv_warp_row(T *a, T *b, T *c, int M, int K) {
    // local thread index within the block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // local thread index within a warp
    int laneid = tid % 32;
    // warpid within a block
    int warpid = tid / 32;
    // actual row this warp is responsible
    int idx = blockIdx.y * blockDim.y + warpid;

    if (idx < M) {
        // load and compute thread sum
        T sum = T(0);
        #pragma unroll
        for (int i = laneid; i < K; i += 32) {
            sum += a[idx * K + i] * b[i];
        }

        // do warp reduce
        sum = warp_reduce_sum(sum);
        __syncthreads();

        // now the thread with laneid = 0 get the warp reduced sum
        if (laneid == 0) {
            c[idx] = sum;
        }
    }
}

int main() {
    int M = 4096;
    int K = 8192;
    size_t size_mat = M * K * sizeof(float);
    size_t size_vec = K * sizeof(float);
    size_t size_res = M * sizeof(float);

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu, *d_a, *d_b, *d_c;
    h_a = (float *)malloc(size_mat);
    h_b = (float *)malloc(size_vec);
    h_c_cpu = (float *)malloc(size_res);
    h_c_gpu = (float *)malloc(size_res);
    cudaMalloc(&d_a, size_mat);
    cudaMalloc(&d_b, size_vec);
    cudaMalloc(&d_c, size_res);

    init_mat(h_a, M, K);
    init_mat(h_b, K, 1);

    cudaMemcpy(d_a, h_a, size_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_vec, cudaMemcpyHostToDevice);

    // const int num_threads = 128;
    // dim3 block_size(num_threads, 1);
    // dim3 grid_size(1, M);
    // gemv_block_row<float, num_threads><<<grid_size, block_size>>>(d_a, d_b, d_c, M, K);
    // cudaError_t err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     printf("CUDA error: %s\n", cudaGetErrorString(err));
    //     exit(1);
    // }

    dim3 block_size(32, 4);
    dim3 grid_size(1, (M + 3) / 4);
    gemv_warp_row<float><<<grid_size, block_size>>>(d_a, d_b, d_c, M, K);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(h_c_gpu, d_c, size_res, cudaMemcpyDeviceToHost);
    gemv_cpu(h_a, h_b, h_c_cpu, M, K);
    printf("\n");
    //print_mat(h_a, M, K);
    //print_mat(h_b, K, 1);
    //print_mat(h_c_cpu, M, 1);
    //print_mat(h_c_gpu, M, 1);



    bool correct = validate(h_c_cpu, h_c_gpu, M);
    printf("validation result: %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
