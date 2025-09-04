#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

void init_mat(float *a, int T, int C) {
    for (int i = 0; i < T*C; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

bool validate(float *a, float *b, int T, int C) {
    for (int i = 0; i < T * C; i++) {
        if (fabs(a[i]-b[i]) > 1e-3) {
            printf("difference detected at (%d, %d), benchmark: %f, kernel: %f\n", i / C, i % C, a[i], b[i]);
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

/**
 * softmax is one of the most widely used technique for prediction
 * the optimization of softmax kernel requires an intense amount of efforts
 * in this code snippet, I will show a complete process of optimizing softmax
 */

// in is an T by C matrix, and we want out to contain softmax value of each element row-wisely
void softmax_forward_cpu(float *in, float *out, int T, int C) {
    for (int i = 0; i < T; i++) {

        // first pass, get maxval of i^th row
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            maxval = fmaxf(maxval, in[i * C + j]);
        }

        // second pass, get denominator, and set the nominator to out to avoid repeated calculation
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            out[i * C + j] = expf(in[i * C + j]-maxval);
            sum += out[i * C + j];
        }

        // third pass, get the softmax value
        for (int j = 0; j < C; j++) {
            out[i * C + j] /= sum;
        }
    }
}

// now use online softmax
void softmax_forward_online_cpu(float *in, float *out, int T, int C) {
    for (int i = 0; i < T; i++) {

        float maxval = -INFINITY;
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float maxval_prev = maxval;
            // meaning maxval != maxval_prev
            if (in[i * C + j] > maxval) {
                maxval = in[i * C + j];
                sum = sum * expf(maxval_prev - maxval) + expf(in[i*C+j]-maxval);
            // meaning maxval == maxval_prev
            } else {
                sum += expf(in[i*C+j]-maxval);
            }
        }

        for (int j = 0; j < C; j++) {
            out[i * C + j] = expf(in[i * C + j]-maxval) / sum;
        }
    }
}

// most naive softmax kernel, one thread is responsible for one row of softmax values
// block_size (N, 1), grid_size (T / N, 1) (3.83ms)
__global__ void softmax_kernel1(float *in, float *out, int T, int C) {
    // global idx of the thread, denoting which row it is responsible for
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T) {

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            maxval = fmaxf(maxval, in[idx * C + j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            sum += expf(in[idx * C + j]-maxval);
        }

        for (int j = 0; j < C; j++) {
            out[idx * C + j] = in[idx * C + j] / sum;
        }
    }
}

// the first naive method has a serious problem in occupany, as each thread is reponsible for one row of elements
// notice that we can have warp reduce primitive for getting max and sum, so, we can have one warp or one block to
// be responsible for one row. We still need to check which layout is better, block for one row or warp for one row,
// I will implement both for testing.

__device__ __inline__ float warp_reduce_sum(float val) {
    for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __inline__ float warp_reduce_max(float val) {
    for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// each block is responsible for one row of elements (537 us with num_threads = 128)
template <const uint NUM_THREADS>
__global__ void softmax_kernel_block(float *in, float *out, int T, int C) {
    // compute the total number of warps
    constexpr int NUM_WARPS = (NUM_THREADS + 31) / 32;
    // initiates smem for block reduce
    __shared__ float smem[NUM_WARPS];
    // smem stores maxval and denominator
    __shared__ float smem_m_d[2];
    // thread index within a block
    int tid = threadIdx.x;
    // thread index within a warp
    int laneid = tid % 32;
    // warp id
    int warpid = tid / 32;
    // row this block is responsible for
    int row = blockIdx.x;

    // iteratively load elements to get thread max
    float max = -INFINITY;
    for (int j = tid; j < C; j += NUM_THREADS) {
        max = fmaxf(max, in[row * C + j]);
    }

    // do warp reduce to get warp max, and set to smem
    max = warp_reduce_max(max);
    if (laneid == 0) {
        smem[warpid] = max;
    }
    __syncthreads();

    // do block reduce to get global max
    max = (laneid < NUM_WARPS) ? smem[laneid] : -INFINITY;
    if (warpid == 0) {
        max = warp_reduce_max(max);
    }
    __syncthreads();

    // thread tid = 0 of the block gets the global max, set to smem
    if (tid == 0) {
        smem_m_d[0] = max;
    }
    __syncthreads();

    // do same procedure for sum
    float sum = 0.0f;
    for (int j = tid; j < C; j += NUM_THREADS) {
        sum += expf(in[row * C + j] - smem_m_d[0]);
    }

    // then we can do the warp reduce sum
    sum = warp_reduce_sum(sum);
    if (laneid == 0) {
        smem[warpid] = sum;
    }
    __syncthreads();

    // do the block reduce to get global sum
    sum = (laneid < NUM_WARPS) ? smem[laneid] : 0.0f;
    if (warpid == 0) {
        sum = warp_reduce_sum(sum);
    }
    __syncthreads();

    if (tid == 0) {
        smem_m_d[1] = sum;
    }
    __syncthreads();

    // then, for each thread, we do the softmax based on global max and sum
    float global_max = smem_m_d[0];
    float global_sum = smem_m_d[1];

    for (int j = tid; j < C; j += NUM_THREADS) {
        out[row * C + j] = expf(in[row * C + j]-global_max) / global_sum;
    }
}

// we can also make one warp responsible for one row, which makes code (hopefully) easier
// and get rid of smem accessing by broadcasting using shfl_sync. Let's try it (637 us)
__global__ void softmax_forward_warp(float *in, float *out, int T, int C) {
    // thread index within the block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // thread index within a warp
    int laneid = tid % 32;
    // warp id
    int warpid = tid / 32;
    // actual row this warp is responsible for
    int row = blockIdx.x * blockDim.y + warpid;

    // compute the thread max
    float max = -INFINITY;
    for (int j = laneid; j < C; j += 32) {
        max = fmaxf(max, in[row * C + j]);
    }
    __syncthreads();

    // compute the warp reduced max, which is also the global max
    max = warp_reduce_max(max);
    __syncthreads();

    // broadcasting the global max to each thread within the warp
    float maxval = __shfl_sync(0xffffffff, max, 0);
    __syncthreads();

    // do same to get the sum
    float sum = 0.0f;
    for (int j = laneid; j < C; j += 32) {
        sum += expf(in[row * C + j] - maxval);
    }
    __syncthreads();

    // compute the warp reduced sum, which is the global sum
    sum = warp_reduce_sum(sum);
    __syncthreads();

    // broadcasting the global sum to each thread within the warp
    float global_sum = __shfl_sync(0xffffffff, sum, 0);
    __syncthreads();

    for (int j = laneid; j < C; j += 32) {
        out[row * C + j] = expf(in[row * C + j]-maxval) / global_sum;
    }
}

// it turns out that each block responsible for one row is faster
// here, let's apply vec4 loading and storing
// now each thread in one iteration will be responsible for loading 4 elements, making the starting
// position as tid * 4, and stride as NUM_THREADS * 4 (445 ms)
template <const uint NUM_THREADS>
__global__ void softmax_kernel_block_vec4(float *in, float *out, int T, int C) {
    // compute the total number of warps
    constexpr int NUM_WARPS = (NUM_THREADS + 31) / 32;
    // initiates smem for block reduce
    __shared__ float smem[NUM_WARPS];
    // smem stores maxval and denominator
    __shared__ float smem_m_d[2];
    // thread index within a block
    int tid = threadIdx.x;
    // thread index within a warp
    int laneid = tid % 32;
    // warp id
    int warpid = tid / 32;
    // row this block is responsible for
    int row = blockIdx.x;
    // starting position
    int start = 4 * tid;
    // stride
    int stride = 4 * NUM_THREADS;

    // iteratively load elements to get thread max
    float max = -INFINITY;
    for (int j = start; j < C; j += stride) {
        float4 reg = FLOAT4(in[row * C + j]);
        max = fmaxf(max, reg.x);
        max = fmaxf(max, reg.y);
        max = fmaxf(max, reg.z);
        max = fmaxf(max, reg.w);
    }

    // do warp reduce to get warp max, and set to smem
    max = warp_reduce_max(max);
    if (laneid == 0) {
        smem[warpid] = max;
    }
    __syncthreads();

    // do block reduce to get global max
    max = (laneid < NUM_WARPS) ? smem[laneid] : -INFINITY;
    if (warpid == 0) {
        max = warp_reduce_max(max);
    }
    __syncthreads();

    // thread tid = 0 of the block gets the global max, set to smem
    if (tid == 0) {
        smem_m_d[0] = max;
    }
    __syncthreads();

    float maxval = smem_m_d[0];
    // do same procedure for sum
    float sum = 0.0f;
    for (int j = start; j < C; j += stride) {
        float4 reg_sum = FLOAT4(in[row * C + j]);
        sum += expf(reg_sum.x - maxval);
        sum += expf(reg_sum.y - maxval);
        sum += expf(reg_sum.z - maxval);
        sum += expf(reg_sum.w - maxval);
    }

    // then we can do the warp reduce sum
    sum = warp_reduce_sum(sum);
    if (laneid == 0) {
        smem[warpid] = sum;
    }
    __syncthreads();

    // do the block reduce to get global sum
    sum = (laneid < NUM_WARPS) ? smem[laneid] : 0.0f;
    if (warpid == 0) {
        sum = warp_reduce_sum(sum);
    }
    __syncthreads();

    if (tid == 0) {
        smem_m_d[1] = sum;
    }
    __syncthreads();

    // then, for each thread, we do the softmax based on global max and sum
    float global_sum = smem_m_d[1];

    for (int j = tid; j < C; j += NUM_THREADS) {
        out[row * C + j] = expf(in[row * C + j]-maxval) / global_sum;
    }
}

// we know online max can reduce the accessing time, let's try the naive version
// each thread is responsible for one row (3.03 ms, better than naive)
__global__ void softmax_forward_online_naive(float *in, float *out, int T, int C) {
    // get the row it responsible for
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float maxval = -INFINITY;
    float sum = 0.0f;
    for (int j = 0; j < C; j++) {
        float maxval_prev = maxval;
        if (in[row * C + j] > maxval) {
            maxval = in[row * C + j];
            sum = sum * expf(maxval_prev - maxval) + expf(in[row * C + j]-maxval);
        } else {
            sum += expf(in[row * C + j] - maxval);
        }
    }

    for (int j = 0; j < C; j++) {
        out[row * C + j] = expf(in[row * C + j] - maxval) / sum;
    }
}

// similarly, we can apply warp reduce and vec4 for a better performance!
// first, let's make one warp responsible for one row
// here we also need to define a union process, of uniting two disjoint set of running max and running denominator into one
// (m1, s1) unite (m2, s2) = (max(m1, m2), s1*e^(m1-m) + s2 * e^(m2-m)) (348 us)
__global__ void softmax_forward_online_warp(float *in, float *out, int T, int C) {
    // thread index within a block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // thread idx within the warp
    int laneid = tid % 32;
    // which warp
    int warpid = tid / 32;
    // which row it is responsible for
    int row = blockIdx.x * blockDim.y + warpid;

    // iteratively load from in and get the thread sum and max
    float sum = 0.0f;
    float max = -INFINITY;
    #pragma unroll
    for (int i = laneid; i < C; i += 32) {
        float last_max = max;
        max = fmaxf(in[row * C + i], max);
        sum = sum * expf(last_max - max) + expf(in[row * C + i] - max);
    }
    __syncthreads();

    // do the warp reduction
    #pragma unroll
    for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
        float offsetmax = __shfl_down_sync(0xffffffff, max, offset);
        float offsetsum = __shfl_down_sync(0xffffffff, sum, offset);
        __syncthreads();
        // do the union. note that, expf is expensive, if max of current thread is larger, we don't have to modify current s
        if (max >= offsetmax) {
            offsetsum *= expf(offsetmax - max);
        } else {
            sum *= expf(max - offsetmax);
            max = offsetmax;
        }
        sum += offsetsum;
        __syncthreads();
    }

    // broadcast to all other threads
    float global_sum = __shfl_sync(0xffffffff, sum, 0);
    float global_max = __shfl_sync(0xffffffff, max, 0);
    #pragma unroll
    for (int i = laneid; i < C; i += 32) {
        out[row * C + i] = expf(in[row * C + i] - global_max) / global_sum;
    }
}

// now, add vec4 to see whether the efficiency will be better (409us)
__global__ void softmax_forward_online_warp_vec4(float *in, float *out, int T, int C) {
    // thread index within a block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // thread idx within the warp
    int laneid = 4 * (tid % 32);
    // which warp
    int warpid = tid / 32;
    // which row it is responsible for
    int row = blockIdx.x * blockDim.y + warpid;

    // iteratively load from in and get the thread sum and max
    float sum = 0.0f;
    float max = -INFINITY;
    // here is a little bit tricky, we need to obtain the max within the 4 floats, and expf sum of the 4 floats
    #pragma unroll
    for (int i = laneid; i < C; i += 128) {
        float4 reg = FLOAT4(in[row * C + i]);
        float last_max = max;
        max = fmaxf(reg.x, max);
        max = fmaxf(reg.y, max);
        max = fmaxf(reg.z, max);
        max = fmaxf(reg.w, max);
        float sum_4 = 0.0f;
        sum_4 += expf(reg.x - max);
        sum_4 += expf(reg.y - max);
        sum_4 += expf(reg.z - max);
        sum_4 += expf(reg.w - max);
        sum = sum * expf(last_max - max) + sum_4;
    }
    __syncthreads();

    // do the warp reduction
    #pragma unroll
    for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
        float offsetmax = __shfl_down_sync(0xffffffff, max, offset);
        float offsetsum = __shfl_down_sync(0xffffffff, sum, offset);
        // do the union. note that, expf is expensive, if max of current thread is larger, we don't have to modify current s
        if (max >= offsetmax) {
            offsetsum *= expf(offsetmax - max);
        } else {
            sum *= expf(max - offsetmax);
            max = offsetmax;
        }
        sum += offsetsum;
        __syncthreads();
    }

    // broadcast to all other threads
    float global_sum = __shfl_sync(0xffffffff, sum, 0);
    float global_max = __shfl_sync(0xffffffff, max, 0);
    #pragma unroll
    for (int i = laneid; i < C; i += 128) {
        float4 reg = FLOAT4(in[row * C + i]);
        float4 reg_s;
        reg_s.x = expf(reg.x - global_max) / global_sum;
        reg_s.y = expf(reg.y - global_max) / global_sum;
        reg_s.z = expf(reg.z - global_max) / global_sum;
        reg_s.w = expf(reg.w - global_max) / global_sum;
        FLOAT4(out[row * C + i]) = reg_s;
    }
}

// now let's try one final thing, make one block to responsible for one row (302us)
template <const uint NUM_THREADS>
__global__ void softmax_forward_online_block(float *in, float *out, int T, int C) {
    // get how many warps within a block
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    // initiate shared memory
    __shared__ float smem_max[NUM_WARPS];
    __shared__ float smem_sum[NUM_WARPS];
    // thread index within a block
    int tid = threadIdx.x;
    // thread index within a warp
    int laneid = tid % 32;
    // which warp this thread is in
    int warpid = tid / 32;
    // which row this block is reponsible for
    int row = blockIdx.x;

    // get thread sum and max
    float max = -INFINITY;
    float sum = 0.0f;
    #pragma unroll
    for (int i = laneid; i < C; i += NUM_THREADS) {
        float last_max = max;
        max = fmaxf(in[row * C + i], max);
        sum = sum * expf(last_max - max) + expf(in[row * C + i]-max);
    }
    __syncthreads();

    // do the warp reduce to get warp sum and max
    #pragma unroll
    for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
        float offsetmax = __shfl_down_sync(0xffffffff, max, offset);
        float offsetsum = __shfl_down_sync(0xffffffff, sum, offset);
        if (offsetmax > max) {
            sum *= expf(max - offsetmax);
            max = offsetmax;
        } else {
            offsetsum *= expf(offsetmax - max);
        }
        sum += offsetsum;
        __syncthreads();
    }

    if (laneid == 0) {
        smem_max[warpid] = max;
        smem_sum[warpid] = sum;
    }

    __syncthreads();

    // read the warp reduced max and sum
    sum = (laneid < NUM_WARPS) ? smem_sum[laneid] : 0.0f;
    max = (laneid < NUM_WARPS) ? smem_max[laneid] : -INFINITY;
    // do block reduce
    if (warpid == 0) {
        #pragma unroll
        for (int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1) {
            float offsetmax = __shfl_down_sync(0xffffffff, max, offset);
            float offsetsum = __shfl_down_sync(0xffffffff, sum, offset);
            if (offsetmax > max) {
                sum *= expf(max - offsetmax);
                max = offsetmax;
            } else {
                offsetsum *= expf(offsetmax - max);
            }
            sum += offsetsum;
            __syncthreads();
        }
    }

    __syncthreads();

    // now tid = 0 has the global sum and max, write to smem
    if (tid == 0) {
        smem_sum[0] = sum;
        smem_max[0] = max;
    }

    #pragma unroll
    for (int i = laneid; i < C; i += NUM_THREADS) {
        out[row * C + i] = expf(in[row * C + i] - smem_max[0]) / smem_sum[0];
    }

}

void test(std::string kernel, bool valid, int T, int C) {
    size_t size = T * C * sizeof(float);

    float *h_a, *h_b_bench, *h_b_gpu, *d_a, *d_b;
    h_a = (float *)malloc(size);
    h_b_bench = (float *)malloc(size);
    h_b_gpu = (float *)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    init_mat(h_a, T, C);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    if (kernel == "naive") {
        dim3 block_size(32, 1);
        dim3 grid_size((T + 31) / 32, 1);
        softmax_kernel1<<<grid_size, block_size>>>(d_a, d_b, T, C);
    } else if (kernel == "block") {
        const uint num_threads = 128;
        dim3 block_size(num_threads, 1);
        dim3 grid_size(T, 1);
        softmax_kernel_block<num_threads><<<grid_size, block_size>>>(d_a, d_b, T, C);
    } else if (kernel == "warp") {
        dim3 block_size(32, 8);
        dim3 grid_size((T + 7) / 8, 1);
        softmax_forward_warp<<<grid_size, block_size>>>(d_a, d_b, T, C);
    } else if (kernel == "block+vec4") {
        const uint num_threads = 128;
        dim3 block_size(num_threads, 1);
        dim3 grid_size(T, 1);
        softmax_kernel_block_vec4<num_threads><<<grid_size, block_size>>>(d_a, d_b, T, C);
    } else if (kernel == "online") {
        dim3 block_size(32, 1);
        dim3 grid_size((T + 31) / 32, 1);
        softmax_forward_online_naive<<<grid_size, block_size>>>(d_a, d_b, T, C);
    } else if (kernel == "online_warp") {
        dim3 block_size(32, 8);
        dim3 grid_size((T + 7) / 8, 1);
        softmax_forward_online_warp<<<grid_size, block_size>>>(d_a, d_b, T, C);
    } else if (kernel == "online_warp_vec4") {
        dim3 block_size(32, 8);
        dim3 grid_size((T + 7) / 8, 1);
        softmax_forward_online_warp_vec4<<<grid_size, block_size>>>(d_a, d_b, T, C);
    } else if (kernel == "online_block") {
        const uint num_threads = 128;
        dim3 block_size(num_threads, 1);
        dim3 grid_size(T, 1);
        softmax_forward_online_block<num_threads><<<grid_size, block_size>>>(d_a, d_b, T, C);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(h_b_gpu, d_b, size, cudaMemcpyDeviceToHost);
    if (valid) {
        softmax_forward_cpu(h_a, h_b_bench, T, C);
        bool correct = validate(h_b_bench, h_b_gpu, T, C);
        printf("validation result: %s\n", correct ? "correct" : "incorrect");
    }

    //print_mat(h_a, T, C);
    //printf("\n");
    //print_mat(h_b_bench, T, C);
    //printf("\n");
    //print_mat(h_b_gpu, T, C);

    free(h_a);
    free(h_b_bench);
    free(h_b_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
}

int main() {
    test("online_block", true, 2048, 4096);
}
