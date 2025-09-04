#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

void cpu_matmul(float *a, float *b, float *c, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

void init_mat(float *a, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

bool validate(float *a, float *b, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(a[i] - b[i]) > 1e-3) {
            printf("difference detected at (%d, %d), benchmark: %f, gpu: %f\n", i / N, i % N, a[i], b[i]);
            return false;
        }
    }
    return true;
}

void print_mat(float *a, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f\t", a[i * N + j]);
        }
        printf("\n");
    }
}

// naive kernel, parallel across M and N (422 ms)
__global__ void matmul_naive(float *a, float *b, float *c, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

// tiling, basic idea is to make the element loaded used as many times as possible to avoid reloading
// tiling enables: coalesced mem accessing and data reusing through smem
// this is a naive tiling with block tile 32 by 32. (329 ms)
__global__ void matmul_tiling(float *a, float *b, float *c, int M, int K, int N) {
    __shared__ float smem_a[32][32];
    __shared__ float smem_b[32][32];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    a += by * blockDim.y * K;
    b += bx * blockDim.x;
    c += by * blockDim.y * N + bx * blockDim.x;

    float sum = 0.0f;
    for (int i = 0; i < K; i += 32) {
        smem_a[ty][tx] = a[ty * K + tx];
        smem_b[ty][tx] = b[ty * N + tx];
        __syncthreads();

        for (int k = 0; k < 32; k++) {
            sum += smem_a[ty][k] * smem_b[k][tx];
        }

        a += blockDim.x;
        b += blockDim.y * N;

        // detail here, we need to make sure that all the threads are reading from the smem data of current iteration, i.e, avoid racing.
        // so we must have a __syncthreads() here or it is possible that one thread begins next iteration and update the smem, while other threads are
        // still at current iteration accessing from smem.
        __syncthreads();
    }

    c[ty * N + tx] = sum;
}

// tiling can be more efficient with larger block tile size.
// we can make our tiles on a and b to be rectangular instead of square, for example, 128 x 8 and 8 x 128, to make the result
// block tile as 128 by 128.
// when doing such technique, it is usually useful to flatten the thread and use divide/modular for finding corresponding indexing.
// 90ms
template <const uint BM, const uint BN, const uint TM, const uint NUM_THREADS>
__global__ void tiling_advanced(float *a, float *b, float *c, int M, int K, int N) {
    constexpr int BK = NUM_THREADS / BM;
    __shared__ float smem_a[BM * BK];
    __shared__ float smem_b[BK * BN];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    int ty_a = tid / BK;
    int tx_a = tid % BK;
    int ty_b = tid / BN;
    int tx_b = tid % BN;
    int ty_c = TM * (tid / (BN / TM));
    int tx_c = TM * (tid % (BN / TM));

    a += by * BM * K;
    b += bx * BN;
    c += by * BM * N + bx * BN;

    // create register for threads
    // each thread is responsible for TM by TM elements.
    float thread_res[TM * TM] = {0.0f};
    float thread_a[TM] = {0.0f};
    float thread_b[TM] = {0.0f};
    #pragma unroll
    for (int tile = 0; tile < K; tile += BK) {
        smem_a[ty_a * BK + tx_a] = a[ty_a * K + tx_a];
        smem_b[ty_b * BN + tx_b] = b[ty_b * N + tx_b];
        __syncthreads();

        a += BK;
        b += BK * N;

        // load one col of smem_a (TM * BK) and one row of smem_b (BK * TM) and do dot product to get partial
        // sum of TM by TM of this thread
        #pragma unroll
        for (int dotId = 0; dotId < BK; dotId++) {
            // load one col of smem_a
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                thread_a[i] = smem_a[(ty_c + i) * BK + dotId];
                thread_b[i] = smem_b[dotId * BN + tx_c + i];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TM; j++) {
                    thread_res[i * TM + j] += thread_a[i] * thread_b[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TM; j++) {
            c[(ty_c + i) * N + tx_c + j] = thread_res[i * TM + j];
        }
    }
}

// vec4 can be used here for a more efficient kernel. (69 ms)
template <const uint BM, const uint BN, const uint TM, const uint NUM_THREADS>
__global__ void tiling_advanced_vec4(float *a, float *b, float *c, int M, int K, int N) {
    constexpr int BK = NUM_THREADS / BM;
    // initiates shared memory, using vec4 -> 4 more space
    __shared__ float smem_a[BM * (BK * 4)];
    __shared__ float smem_b[(BK * 4) * BN];
    // get the indexing mapping
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    int ty_a = tid / BK;
    int tx_a = (tid % BK) * 4;
    int ty_b = tid / (BN / 4);
    int tx_b = (tid % (BN / 4)) * 4;
    int ty_c = (tid / (BN / TM)) * TM;
    int tx_c = (tid % (BN / TM)) * TM;

    a += by * BM * K;
    b += bx * BN;
    c += by * BM * N + bx * BN;
    float thread_a[TM] = {0.0f};
    float thread_b[TM] = {0.0f};
    float thread_res[TM * TM] = {0.0f};

    #pragma unroll
    for (int tile = 0; tile < K; tile += BK * 4) {
        FLOAT4(smem_a[ty_a * (BK * 4) + tx_a]) = FLOAT4(a[ty_a * K + tx_a]);
        FLOAT4(smem_b[ty_b * BN + tx_b]) = FLOAT4(b[ty_b * N + tx_b]);
        __syncthreads();

        a += BK * 4;
        b += (BK * 4) * N;

        #pragma unroll
        for (int dotId = 0; dotId < BK * 4; dotId++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                thread_a[i] = smem_a[(ty_c + i) * (BK * 4) + dotId];
                thread_b[i] = smem_b[dotId * BN + tx_c + i];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TM; j++) {
                    thread_res[i * TM + j] += thread_a[i] * thread_b[j];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TM; j += 4) {
            FLOAT4(c[(ty_c + i) * N + tx_c + j]) = FLOAT4(thread_res[i * TM + j]);
        }
    }
}

// before going to even more advanced optimization, one other thing can be done is cta swizzeling, making the L2 cache efficiency higher.
__global__ void matmul_advanced_final() {

}

// advanced optimizations:
// ping-ping or n-stage pipelining, where load process becomes async, enabling
// computing and loading overlap. (in Hopper and Blackwell, computation part is async, which enables even better overlapping)

void test(std::string kernel, bool valid, int M, int K, int N) {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu, *d_a, *d_b, *d_c;
    h_a = (float *)malloc(M * K * sizeof(float));
    h_b = (float *)malloc(K * N * sizeof(float));
    h_c_cpu = (float *)malloc(M * N * sizeof(float));
    h_c_gpu = (float *)malloc(M * N * sizeof(float));
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    init_mat(h_a, M, K);
    init_mat(h_b, K, N);
    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    if (kernel == "naive") {
        dim3 block_size(32, 32);
        dim3 grid_size((N + 31) / 32, (M + 31) / 32);
        matmul_naive<<<grid_size, block_size>>>(d_a, d_b, d_c, M, K, N);
    } else if (kernel == "tiling") {
        dim3 block_size(32, 32);
        dim3 grid_size((N + 31) / 32, (M + 31) / 32);
        matmul_tiling<<<grid_size, block_size>>>(d_a, d_b, d_c, M, K, N);
    } else if (kernel == "tiling_advanced") {
        const uint BM = 128;
        const uint BN = 128;
        const uint TM = 8;
        const uint num_threads = 256;
        dim3 block_size(num_threads, 1);
        dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);
        tiling_advanced<BM, BN, TM, num_threads><<<grid_size, block_size>>>(d_a, d_b, d_c, M, K, N);
    } else if (kernel == "tiling_vec4") {
        const uint BM = 128;
        const uint BN = 128;
        const uint TM = 8;
        const uint num_threads = 256;
        dim3 block_size(num_threads, 1);
        dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);
        tiling_advanced_vec4<BM, BN, TM, num_threads><<<grid_size, block_size>>>(d_a, d_b, d_c, M, K, N);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cuda error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    if (valid) {
        cudaMemcpy(h_c_gpu, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cpu_matmul(h_a, h_b, h_c_cpu, M, K, N);
        bool correct = validate(h_c_cpu, h_c_gpu, M, N);
        printf("validation result: %s\n", correct ? "correct" : "incorrect");
    }

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    //test("naive", false, 2048, 8192, 4096);
    test("tiling_vec4", false, 2048, 8192, 4096);
}
