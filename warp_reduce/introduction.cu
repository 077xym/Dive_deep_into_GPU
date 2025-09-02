#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// __shfl_sync is used to let the threads within a warp to access the value stored in other thread's register
// one common use is broadcasting, where one thread get the value and other threads read from it
// but this method can have more generalized functionality for value sharing within warps.
__global__ void shfl_sync_example() {
    int tid = threadIdx.x;
    // claim a value that can be shared within warps
    int my_value = tid * 10;

    // broadcasting: each thread within the warp get my_value of thread 1. Not here, my_value represents
    // the register that stores the value of thread 1
    int from_one = __shfl_sync(0xffffffff, my_value, 1);
    printf("thread Id: %d read thread 1's my value: %d\n", tid, from_one);
    __syncthreads();

    // gets value from next thread
    int from_next = __shfl_sync(0xfffffff, my_value, (tid + 1) % 32);
    printf("thread Id: %d read thread %d's my value: %d\n", tid, (tid + 1) % 32, from_next);
    __syncthreads();

    // there is actually one really important thing left when calling __shfl_sync, that is warpwidth
    // __shfl_sync(mask, value, laneid, warpwidth)
    // by default, warpwidth is 32, but we can actually modify our warpwidth, e.g, to 16
    // the actual laneid is computed by tid & ~(warpwidth-1) | (laneid & 15)
    // the reason for this transform is that it lets the user to not have to consider about sub warp level indexing
    // i.e, user can still assume every thread is from one warp(32 threads), and do the indexing logic based on it
    // Shown on the following graph, I wrote (tid+1)%32, meaning each thread reads the next thread value
    // However, as I set warpwidth to be 16, then, t0->t1, t1->t2,..., t15->t0, t16-t17, ..., t31->t16.
    int from_next_half = __shfl_sync(0xffffffff, my_value, (tid + 1)%32, 16);
    printf("thread Id: %d read thread %d's my value: %d\n", tid, (tid & ~(16-1) | ((tid + 1)%32 & 15)), from_next_half);
}

// shuffle_up lets the thread n to read the value of thread n-d, where d is defined by the user
// shfl_up_sync(mask, value, delta, warpwidth)
__global__ void shuffle_up_example() {
    int tid = threadIdx.x;
    int my_value = tid * 10;

    // for shfl_up_sync, the actual laneid is calculated as
    // actual lane id is tid % width < delta ? tid : (tid & ~(width-1)) | ((tid - delta) & (width-1))
    int from_before_two = __shfl_up_sync(0xffffffff, my_value, 2);
    printf("thread %d reads thread %d's value %d\n", tid, tid < 2 ? tid : (tid & ~(32-1)) | ((tid - 2) & (32-1)), from_before_two);
    __syncthreads();

    // similarly, you don't have to worry about delta manipulation when you have a warpwidth less than 32
    int from_before_two_half = __shfl_up_sync(0xffffffff, my_value, 2, 16);
    printf("thread %d reads thread %d's value %d\n", tid, tid % 16 < 2 ? tid : (tid & ~(16-1)) | ((tid - 2) & (16-1)), from_before_two_half);
}

// converse to shuffle_up, shuffle_down lets the thread n to read the value of thread n + d
__global__ void shuffle_down_example() {
    int tid = threadIdx.x;
    int my_value = tid * 10;

    // similar to shfl_up_sync, the actual laneid is computed as
    // tid % width >= width - delta ? tid : (tid & ~(width-1)) | ((tid + delta) & (width-1))
    int from_behind_two = __shfl_down_sync(0xffffffff, my_value, 2);
    printf("thread %d reads thread %d's value %d\n", tid, tid % 32 >= 32 - 2 ? tid : (tid & ~(32-1)) | ((tid + 2) & (32-1)), from_behind_two);
    __syncthreads();

    // if we set our width, we can have
    int from_behind_two_half = __shfl_down_sync(0xffffffff, my_value, 2, 16);
    printf("thread %d reads thread %d's value %d\n", tid, tid % 16 >= 16 - 2 ? tid : (tid & ~(16-1)) | ((tid + 2) & (16-1)), from_behind_two_half);
    __syncthreads();
}

// we can see here how the actual laneid is computed:
// (tid & ~(width-1)) | src_lane & (width-1), where src_lane is either defined by you, or tid +- delta in up and down method

// one more sharing procedure is butterfly sharing using xor, the corresponding method is called __shfl_xor_sync
// this method is comparatively simpler than methods introduced before.
__global__ void shuffle_xor_example() {
    int tid = threadIdx.x;
    int my_value = tid * 10;

    int butterfly = __shfl_xor_sync(0xffffffff, my_value, 2);
    printf("thread %d reads thread %d's value %d\n", tid, tid ^ 2, butterfly);
    __syncthreads();
}

// among all the methods above, the most commonly and widely used is shuffle_down, many
// official kernel applies this method for warp level and block level reduction, here, I will
// use reduce sum as the example to see a practical application on __shfl_down_sync
template <const uint width>
__global__ void reduce_sum() {
    int val = 1;

    for (int offset = (width >> 1); offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset, width);
    }

    if (threadIdx.x % width == 0) {
        printf("sum reduced to thread %d, with sum = %d\n", threadIdx.x, val);
    }
    __syncthreads();
}

int main() {
    // for simplicity, one one warp within a block, one block per grid
    dim3 block_size(64, 1);
    dim3 grid_size(1, 1);
    reduce_sum<32><<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
}
