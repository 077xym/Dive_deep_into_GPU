#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cassert>

// define matrix dim
constexpr int M = 1024;
constexpr int N = 2048;
constexpr int TILE_M = 32;
constexpr int TILE_N = 64;

extern "C" __global__
void tma_load_kernel(CUtensorMap *tma_map) {
    /*
        One hardest thing to use TMA manully is to align the memory, as TMA, along with the mbarrier,
        both have requirements on memory alignment.

        One thing to know is: when I say memory aligned, it means the starting address of a chunk of memory is divisible by the alignment you claimed

        1. SHARED MEM ALIGNMENT
            - smem[] requires 16-byte alignement for TMA operations
            - __align__(16) or alignas(16) ensures the starting address is divisible by 16
            - mem align only requires the starting address. The address of the elements within does not need to satisfy the requirement unless the element
              itself requires mem align
            - total: TILE_M * TILE_N * sizeof(float)

        2. MBARRIER ALIGNMENT
            - mbar_obj requires 8-byte alignment for atomic operation
            - placed after tile data, with manual alignment for making sure the the alignment

        3. CUtensorMap ALIGNMENT
            - CUtensorMap structure requires 64-byte alignment
            - Global memory pointer must be 16-byte aligned
            - strides must be 16-byte aligned
    */

    // shared memory layout, with 16-byte aligned
    extern __shared__ __align__(16) float smem[];

    // mbarrier needs 8 byte aligned, so we need some caculation here
    constexpr int tile_elements = TILE_M * TILE_N;
    constexpr int tile_bytes = tile_elements * sizeof(float);

    // we want to find the next address that is divisible by 8
    constexpr int mbar_offset_bytes = ((tile_bytes + 7) / 8) * 8; // get the ceil of tile_bytes / 8
    constexpr int mbar_offset_elements = mbar_offset_bytes / sizeof(float); // change back to float index

    // now we are ready to get the starting address of mbar_obj
    uint64_t *mbar_obj = reinterpret_cast<uint64_t *>(&smem[mbar_offset_elements]);

    // convert to shared memory address
    const uintptr_t mbar_addr = (uintptr_t)__cvta_generic_to_shared(mbar_obj);
    const uintptr_t smem_addr = (uintptr_t)__cvta_generic_to_shared(smem);

    // for debug check on mem alignment
    if (threadIdx.x == 0) {
        assert((smem_addr % 16)==0);
        assert((mbar_addr % 8)==0);
    }

    // get which tile this block is responsible for
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // only thread 0 will do the TMA operations
    if (threadIdx.x == 0) {
        // initialize mbarrier for 1 thread
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;\n\t"
            : :"l"(mbar_addr), "r"(1) : "memory"
        );

        // set the expect_tx for mbarrier
        constexpr int transaction_bytes = TILE_M * TILE_N * sizeof(float);
        unsigned long long token = 0u;
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n\t"
            :"=l"(token) : "l"(mbar_addr), "r"(transaction_bytes) : "memory"
        );

        // Issue the TMA load operation
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n\t"
            :
            : "l"(smem_addr) // destination
              "l"(tma_map) // TMA descriptor
              "r"(tile_col) // tile coordinate 1 (col)
              "r"(tile_row) // tile coordinate 0 (row), depending on how you initialize your CUtensorMap
              "l"(mbar_addr) // mbarrier for synchronization
            : "memory"
        );
        // wait for TMA load
        unsigned done = 0u;
        do {
            // test wait will generate a predicate, and we use this predicate to check completion
            asm volatile(
                "{ .reg .pred p;                                        \n\t"
                "  mbarrier.test_wait.shared::cta.b64 p, [%1], %2;       \n\t"
                "  selp.u32 %0, 1, 0, p;                                \n\t"
                "}                                                     \n\t"
                : "=r"(done) : "l"(mbar_addr), "l"(token) : "memory"
            );
        } while (!done);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        printf("Block (%d,%d): TMA load completed for tile [%d:%d, %d:%d]\n",
               blockIdx.x, blockIdx.y,
               tile_row * TILE_M, (tile_row + 1) * TILE_M,
               tile_col * TILE_N, (tile_col + 1) * TILE_N);

        // Print first few elements to verify
        printf("First elements: %.2f, %.2f, %.2f, %.2f\n",
               smem[0], smem[1], smem[2], smem[3]);
    }
}

// HOST-SIDE
/*
    MEM ALIGMENT REQ for CUtensorMap
        1. CUtensorMap struct: 64-byte aligned
        2. Global Memory pointer: 16-byte aligned
        3. Tensor Stride: row stride must be divisible by 16
*/
CUtensorMap create_tma_descriptor(float *global_matrix) {
    // align the tma map to 64-bytes
    alignas(64) CUtensorMap tma_map;

    // verify global memory alignment (if allocated by cudaMalloc, this check should be fine)
    uintptr_t global_addr = reinterpret_cast<uintptr_t>(global_matrix);
    assert((global_addr % 16)==0);

    // set up parameters
    uint64_t tensor_shape[2] = {M, N}; // Matrix dimension
    uint64_t tensor_stride[2] = {N * sizeof(float), sizeof(float)}; // moving down one row and moving down on col
    uint32_t box_size[2] = {TILE_M, TILE_N};
    uint32_t element_strides[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map,                                        // 64-byte aligned descriptor
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,                // Data type
        2,                                              // Tensor rank (2D)
        global_matrix,                                  // 16-byte aligned global memory
        tensor_shape,                                   // Tensor dimensions
        tensor_stride,                                  // 16-byte aligned strides
        box_size,                                       // Tile dimensions
        element_strides,                                // Element order
        CU_TENSOR_MAP_INTERLEAVE_NONE,                 // No interleaving
        CU_TENSOR_MAP_SWIZZLE_NONE,                    // No swizzling
        CU_TENSOR_MAP_L2_PROMOTION_NONE,               // L2 cache behavior
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE              // Out-of-bounds handling
    );

    if (result != CUDA_SUCCESS) {
        printf("error: cuTensorMapEncodeTIled failed with code %d\n", result);
        exit(EXIT_FAILURE);
    }

    return tma_map;
}

void init_mat(float *a, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        a[i] = (float) rand() / RAND_MAX;
    }
}

int main() {
    // initialize cuda API
    cuInit(0);

    // allocate the matrix
    float *global_matrix_h, *global_matrix_d;
    size_t size = M * N * sizeof(float);

    global_matrix_h = (float *)malloc(size);
    init_mat(global_matrix_h, M, N);

    cudaMalloc(&global_matrix_d, size);
    assert((reinterpret_cast<uintptr_t>(global_matrix_d) % 16) == 0);
    cudaMemcpy(global_matrix_d, global_matrix_h, size, cudaMemcpyHostToDevice);

    // create tma descriptor
    CUtensorMap *d_tma_map;
    CUtensorMap tma_map = create_tma_descriptor(global_matrix_d);
    cudaMalloc(&d_tma_map, sizeof(CUtensorMap));
    cudaMemcpy(d_tma_map, &tma_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    dim3 block_size(32, 1, 1);
    dim3 grid_size((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);

    // compute the total smem required
    constexpr int shmem = (TILE_M * TILE_N * sizeof(float) + 7) / 8 * 8 + 8;

    tma_load_kernel<<<grid_size, block_size, shmem>>>(d_tma_map);

    cudaError_t e1 = cudaGetLastError();
    if (e1 != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s (%d)\n",
            __FILE__, __LINE__, cudaGetErrorString(e1), (int)e1);
        exit(EXIT_FAILURE);
    }

    cudaError_t e2 = cudaDeviceSynchronize();
    if (e2 != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s (%d)\n",
            __FILE__, __LINE__, cudaGetErrorString(e2), (int)e2);
        exit(EXIT_FAILURE);
    }

    printf("TMA load success");
    free(global_matrix_h);
    cudaFree(global_matrix_d);

    return 0;
}

