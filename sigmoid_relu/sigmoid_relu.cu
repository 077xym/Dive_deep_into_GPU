#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <string>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

/**
 * sigmoid is a classic element-wise operation, where each element is exponentiated to 1 / (1 + e ^ -x)
 * so as relu, both of them are really similar and I will put them into one code snippet here.
 */
void cpu_sigmoid(float *in, float *out, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i * N + j] = 1.0f / (1.0f + expf(-in[i * N + j]));
        }
    }
}

void cpu_relu(float *in, float *out, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i * N + j] = (in[i * N + j] >= 0.0f) ? in[i * N + j] : 0.0f;
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

// 407.7ms
__global__ void sigmoid_foward_naive(float *in, float *out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        out[row * N + col] = 1.0f / (1.0f + expf(-in[row * N + col]));
    }
}

// 267ms
__global__ void sigmoid_foward_vec4(float *in, float *out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (row < M && col < N) {
        float4 reg_in = FLOAT4(in[row * N + col]);
        float4 reg_out;
        reg_out.x = 1.0f / (1.0f + expf(-reg_in.x));
        reg_out.y = 1.0f / (1.0f + expf(-reg_in.y));
        reg_out.z = 1.0f / (1.0f + expf(-reg_in.z));
        reg_out.w = 1.0f / (1.0f + expf(-reg_in.w));
        FLOAT4(out[row * N + col]) = reg_out;
    }
}

// 351.8ms
__global__ void relu_foward_naive(float *in, float *out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        out[row * N + col] = (in[row * N + col] >= 0.0f) ? in[row * N + col] : 0.0f;
    }
}

// 266.7ms
__global__ void relu_foward_vec4(float *in, float *out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    if (row < M && col < N) {
        float4 reg_in = FLOAT4(in[row * N + col]);
        float4 reg_out;
        reg_out.x = (reg_in.x >= 0.0f) ? reg_in.x : 0.0f;
        reg_out.y = (reg_in.y >= 0.0f) ? reg_in.y : 0.0f;
        reg_out.z = (reg_in.z >= 0.0f) ? reg_in.z : 0.0f;
        reg_out.w = (reg_in.w >= 0.0f) ? reg_in.w : 0.0f;
        FLOAT4(out[row * N + col]) = reg_out;
    }
}

void test(std::string kernel) {
    int M = 2048;
    int N = 4096;
    size_t size = M * N * sizeof(float);
    float *h_a, *h_b_cpu, *h_b_gpu, *d_a, *d_b;
    h_a = (float *)malloc(size);
    h_b_cpu = (float *)malloc(size);
    h_b_gpu = (float *)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    init_mat(h_a, M, N);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    if (kernel == "sigmoid naive") {
        dim3 block_size(32, 32);
        dim3 grid_size((N + 31) / 32, (M + 31) / 32);
        sigmoid_foward_naive<<<grid_size, block_size>>>(d_a, d_b, M, N);
    } else if (kernel == "sigmoid vec4") {
        dim3 block_size(32, 32);
        dim3 grid_size((N + 32 * 4 - 1) / (32 * 4 - 1), (M + 31) / 32);
        sigmoid_foward_vec4<<<grid_size, block_size>>>(d_a, d_b, M, N);
    } else if (kernel == "relu naive") {
        dim3 block_size(32, 32);
        dim3 grid_size((N + 31) / 32, (M + 31) / 32);
        relu_foward_naive<<<grid_size, block_size>>>(d_a, d_b, M, N);
    } else if (kernel == "relu vec4") {
        dim3 block_size(32, 32);
        dim3 grid_size((N + 32 * 4 - 1) / (32 * 4 - 1), (M + 31) / 32);
        relu_foward_vec4<<<grid_size, block_size>>>(d_a, d_b, M, N);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cuda Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(h_b_gpu, d_b, size, cudaMemcpyDeviceToHost);
    if (kernel == "sigmoid naive" || kernel == "sigmoid vec4") {
        cpu_sigmoid(h_a, h_b_cpu, M, N);
    } else if (kernel == "relu vec4" || kernel == "relu naive") {
        cpu_relu(h_a, h_b_cpu, M, N);
    }

    bool correct = validate(h_b_cpu, h_b_gpu, M, N);
    printf("validation result: %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b_cpu);
    free(h_b_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
}

int main() {
    test("relu vec4");
}
