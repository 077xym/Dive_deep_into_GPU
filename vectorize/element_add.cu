#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

void cpu_add(float *a, float *b, float *c, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = a[i * N + j] + b[i * N + j];
        }
    }
}

bool validate(float *a, float *b, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(a[i * N + j]-b[i * N + j]) > 1e-2) {
                printf("difference at (%d, %d), benchmark: %f, gpu: %f\n", i, j, a[i * N + j], b[i * N + j]);
                return false;
            }
        }
    }
    return true;
}

void init_mat(float *a, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

// vec4 can be used nearly every place. Though with new loading technique such as tma, vec io
// is not that widely used, but it still keep its position
// the naive way (418 us)
template <typename T>
__global__ void elementwise_add_naive(T *a, T *b, T *c, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row <= M && col <= N) {
        c[row * N + col] = a[row * N + col] + b[row * N + col];
    }
}

// when using float 4, row stays the same, but col will change (398 us on T4)
__global__ void elementwise_add_vec4(float *a, float *b, float *c, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (row <= M && col <= N) {
        float4 reg_a = FLOAT4(a[row * N + col]);
        float4 reg_b = FLOAT4(b[row * N + col]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[row * N + col]) = reg_c;
    }
}

int main() {
    int M = 2048;
    int N = 4096;
    size_t size = M * N * sizeof(float);

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu, *d_a, *d_b, *d_c;
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c_cpu = (float *)malloc(size);
    h_c_gpu = (float *)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    init_mat(h_a, M, N);
    init_mat(h_b, M, N);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 block_size(32, 32);
    dim3 grid_size((N + 31) / 32, (M + 31) / 32);
    elementwise_add_naive<float><<<grid_size, block_size>>>(d_a, d_b, d_c, M, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    cpu_add(h_a, h_b, h_c_cpu, M, N);

    bool correct = validate(h_c_cpu, h_c_gpu, M, N);
    printf("validate result: %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

