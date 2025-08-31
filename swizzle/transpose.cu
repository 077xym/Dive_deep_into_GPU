#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <string>

// cpu benchmark for checking correctness of the kernel
void transpose_cpu(int *a, int *b, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            b[j * M + i] = a[i * N + j];
        }
    }
}

//
__global__ void transpose_naive(int *a, int *b, int M, int N) {
    __shared__ int s[32][32];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        s[threadIdx.x][threadIdx.y] = a[row * N + col];
        __syncthreads();
        b[col * M + row] = s[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_padding(int *a, int *b, int M, int N) {
    __shared__ int s[32][33];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        s[threadIdx.x][threadIdx.y] = a[row * N + col];
        __syncthreads();
        b[col * M + row] = s[threadIdx.x][threadIdx.y];
    }
}


__global__ void transpose_smem_swizzle(int *a, int *b, int M, int N) {
    __shared__ int s[32][32];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        s[threadIdx.x][threadIdx.y ^ threadIdx.x] = a[row * N + col];
        __syncthreads();
        b[col * M + row] = s[threadIdx.x][threadIdx.y ^ threadIdx.x];
    }
}

__global__ void transpose_smem_cta_swizzle(int *a, int *b, int M, int N,
uint num_SMs, uint num_iters, uint super_M, uint RBlocks, uint CBlocks) {
    __shared__ int s[32][32];
    #pragma unroll
    for (int i = 0; i < num_iters; i++) {
        const int blockId = i * num_SMs + blockIdx.x;
        const int super_repeat = super_M * CBlocks;
        const int super_rows = (RBlocks / super_M) * super_M;
        const int remaining_rows = RBlocks - super_rows;
        int global_row = 0;
        int global_col = 0;
        if (blockId < super_rows * CBlocks) {
            global_row = (blockId / super_repeat) * super_M + blockId % super_M;
            global_col = (blockId % super_repeat) / super_M;
        } else if (blockId < RBlocks * CBlocks) {
            const int remaining_id = blockId - super_rows * CBlocks;
            global_row = super_rows + (remaining_id % remaining_rows);
            global_col = remaining_id / remaining_rows;
        } else {
            break;
        }
        s[threadIdx.x][threadIdx.y ^ threadIdx.x] = a[(global_row * blockDim.y + threadIdx.y) * N + global_col * blockDim.x + threadIdx.x];
        __syncthreads();
        b[(global_col * blockDim.x + threadIdx.x) * M + global_row * blockDim.y + threadIdx.y] = s[threadIdx.x][threadIdx.y ^ threadIdx.x];
    }
}

void init_mat(int *a, int M, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 100);

    for (int i = 0; i < M * N; i++) {
        a[i] = dis(gen);
    }
}

void print_mat(int *a, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

bool validate(int *a, int *b, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        if (a[i] != b[i]) {
            printf("kernel: %d, benchmark: %d\n at (%d, %d)", a[i], b[i], i / 64, i % 64);
            return false;
        }
    }
    return true;
}

void test(int M, int N, std::string kernel) {
    size_t size = M * N * sizeof(int);
    int *h_a, *h_b, *h_b_cpu, *d_a, *d_b;
    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_b_cpu = (int *)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    init_mat(h_a, M, N);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    if (kernel == "naive") {
        dim3 block_size(32, 32);
        dim3 grid_size((N+31)/32, (M+31)/32);
        transpose_naive<<<grid_size, block_size>>>(d_a, d_b, M, N);
        cudaDeviceSynchronize();
    } else if (kernel == "padding") {
        dim3 block_size(32, 32);
        dim3 grid_size((N+31)/32, (M+31)/32);
        transpose_padding<<<grid_size, block_size>>>(d_a, d_b, M, N);
        cudaDeviceSynchronize();
    } else if (kernel == "smem_swizzle") {
        dim3 block_size(32, 32);
        dim3 grid_size((N+31)/32, (M+31)/32);
        transpose_smem_swizzle<<<grid_size, block_size>>>(d_a, d_b, M, N);
        cudaDeviceSynchronize();
    } else if (kernel == "smem_cta_swizzle") {
        uint num_SMs = 40;
        uint super_M = 6;
        uint RBlocks = (M + 31) / 32;
        uint CBlocks = (N + 31) / 32;
        uint num_iters = (RBlocks * CBlocks + num_SMs - 1) / num_SMs;
        dim3 block_size(32, 32);
        dim3 grid_size(num_SMs, 1);
        transpose_smem_cta_swizzle<<<grid_size, block_size>>>(d_a, d_b, M, N, num_SMs, num_iters, super_M, RBlocks, CBlocks);
        cudaDeviceSynchronize();
    } else {
        return;
    }

    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    transpose_cpu(h_a, h_b_cpu, M, N);

    //print_mat(h_b, N, M);
    //print_mat(h_b_cpu, N, M);

    bool correct = validate(h_b, h_b_cpu, M, N);
    printf("validation check: %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b);
    free(h_b_cpu);
    cudaFree(d_a);
    cudaFree(d_b);
}

int main() {
    int M = 4096;
    int N = 8192;

    test(M, N, "smem_cta_swizzle");
}

