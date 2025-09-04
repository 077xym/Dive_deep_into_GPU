#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
/**
 * layer norm is yet another kernel that heavily relies on warp reduce
 */

void cpu_lm(float *in, float *out, int M, int N, float alpha, float beta) {
    float epislon = 1e-5;
    for (int i = 0; i < M; i++) {
        // compute the row-wise sum
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += in[i * N + j];
        }

        // compute the average
        float average = sum / N;

        // compute the variance
        float sum_var = 0.0f;
        for (int j = 0; j < N; j++) {
            sum_var += (in[i * N + j] - average) * (in[i * N + j] - average);
        }

        float var = sum_var / N;

        // compute the normalized feature
        for (int j = 0; j < N; j++) {
            out[i * N + j] = alpha * ((in[i * N + j] - average) / sqrt(var + epislon)) + beta;
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
        if (fabs(a[i] - b[i]) > 1e-2) {
            printf("difference detected at (%d, %d), benchmark: %f, gpu: %f\n", i / N, i % N, a[i], b[i]);
            return false;
        }
    }
    return true;
}

void print_mat(float *a, int T, int C) {
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < C; j++) {
            printf("%f\t", a[i * C + j]);
        }
        printf("\n");
    }
}

// naive way, each thread is responsible for one row (3.73ms)
__global__ void lm_naive(float *in, float *out, int M, int N, float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += in[row * N + j];
    }

    float avg = sum / N;

    float sum_var = 0.0f;
    for (int j = 0; j < N; j++) {
        sum_var += (in[row * N + j] - avg) * (in[row * N + j] -avg);
    }

    float var = sum_var / N;

    for (int j = 0; j < N; j++) {
        out[row * N + j] = alpha * ((in[row * N + j] - avg) * rsqrtf(var + 1e-5)) + beta;
    }
}

// better way, each warp is responsible for one row (628 us)
__device__ __inline__ float warp_reduce_sum(float val) {
    for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void lm_warp(float *in, float *out, int M, int N, float alpha, float beta) {
    // local thread index within a block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // which lane it is within a warp
    int laneid = tid % 32;
    // which local row this thread is responsible
    int warpid = tid / 32;
    // global row
    int row = blockIdx.x * blockDim.y + warpid;

    // compute the thread sum
    float sum = 0.0f;
    for (int j = laneid; j < N; j += 32) {
        sum += in[row * N + j];
    }
    __syncthreads();

    // compute the warp reduced sum
    sum = warp_reduce_sum(sum);

    // broadcast the warp reduced sum from laneid=0 to all threads
    sum = __shfl_sync(0xffffffff, sum, 0);
    float avg = sum / N;
    __syncthreads();

    // compute the variance
    float sum_var = 0.0f;
    for (int j = laneid; j < N; j += 32) {
        sum_var += (in[row * N + j] - avg) * (in[row * N + j] - avg);
    }
    __syncthreads();

    // compute the warp reduced sum
    sum_var = warp_reduce_sum(sum_var);
    sum_var = __shfl_sync(0xffffffff, sum_var, 0);
    float var = sum_var / N;
    __syncthreads();

    for (int j = laneid; j < N; j+=32) {
        out[row * N + j] = alpha * ((in[row * N + j]-avg) * rsqrtf(var + 1e-5)) + beta;
    }
}

// 536 us
__global__ void lm_vec4(float *in, float *out, int M, int N, float alpha, float beta) {
    // local thread index within a block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // idx within a warp
    int laneid = tid % 32;
    // warp id (local row id)
    int warpid = tid / 32;
    // start and stride
    int start = 4 * laneid;
    int stride = 128;
    // global row
    int row = blockIdx.x * blockDim.y + warpid;

    float sum = 0.0f;
    for (int j = start; j < N; j += stride) {
        float4 reg = FLOAT4(in[row * N + j]);
        sum += reg.x;
        sum += reg.y;
        sum += reg.z;
        sum += reg.w;
    }
    __syncthreads();

    sum = warp_reduce_sum(sum);
    sum = __shfl_sync(0xffffffff, sum, 0);
    float avg = sum / N;
    __syncthreads();

    float sum_var = 0.0f;
    for (int j = start; j < N; j += stride) {
        float4 reg = FLOAT4(in[row * N + j]);
        sum_var += (reg.x - avg) * (reg.x - avg);
        sum_var += (reg.y - avg) * (reg.y - avg);
        sum_var += (reg.z - avg) * (reg.z - avg);
        sum_var += (reg.w - avg) * (reg.w - avg);
    }
    __syncthreads();

    sum_var = warp_reduce_sum(sum_var);
    sum_var = __shfl_sync(0xffffffff, sum_var, 0);
    float var = sum_var / N;
    __syncthreads();

    for (int j = start; j < N; j += stride) {
        float4 reg = FLOAT4(in[row * N + j]);
        float4 reg_s;
        reg_s.x = alpha * ((reg.x - avg) * rsqrtf(var + 1e-5)) + beta;
        reg_s.y = alpha * ((reg.y - avg) * rsqrtf(var + 1e-5)) + beta;
        reg_s.z = alpha * ((reg.z - avg) * rsqrtf(var + 1e-5)) + beta;
        reg_s.w = alpha * ((reg.w - avg) * rsqrtf(var + 1e-5)) + beta;
        FLOAT4(out[row * N + j]) = reg_s;
    }
}

void test(std::string kernel) {
    int M = 2048;
    int N = 4096;
    float alpha = 0.3f;
    float beta = 0.3f;
    size_t size = M * N * sizeof(float);
    float *h_a, *h_b_cpu, *h_b_gpu, *d_a, *d_b;
    h_a = (float *)malloc(size);
    h_b_cpu = (float *)malloc(size);
    h_b_gpu = (float *)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    init_mat(h_a, M, N);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    if (kernel == "naive") {
        dim3 block_size(32, 1);
        dim3 grid_size((M + 31) / 32);
        lm_naive<<<grid_size, block_size>>>(d_a, d_b, M, N, alpha, beta);
    } else if (kernel == "warp") {
        dim3 block_size(32, 4);
        dim3 grid_size((M + 3) / 4, 1);
        lm_warp<<<grid_size, block_size>>>(d_a, d_b, M, N, alpha, beta);
    } else if (kernel == "vec4") {
        dim3 block_size(32, 4);
        dim3 grid_size((M + 3) / 4, 1);
        lm_vec4<<<grid_size, block_size>>>(d_a, d_b, M, N, alpha, beta);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cuda Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(h_b_gpu, d_b, size, cudaMemcpyDeviceToHost);

    cpu_lm(h_a, h_b_cpu, M, N, alpha, beta);

    bool correct = validate(h_b_cpu, h_b_gpu, M, N);
    printf("validation result: %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b_cpu);
    free(h_b_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
}

int main() {
    test("vec4");
}
