#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cassert>

// Matrix dimensions
constexpr int M = 128;
constexpr int N = 256;
constexpr int TILE_M = 32;
constexpr int TILE_N = 64;

/*
    TMA is a independent hardware component, we only need one thread to call it, and it will not block the execution

    So, theoretically, when doing pure TMA process, we only need 1 thread to
    1. initialize mbarrier
    2. allocating smem, pay attention to the mem alignment, as mbarrier object is 8-byte aligned and data is 16-byte aligned
    3. call TMA load
    4. check completion using mbarrier
    5. call TMA store
    6. check completion using commit_group
*/

extern "C" __global__
void tma_matrix_copy_kernel(CUtensorMap *tma_load_map, CUtensorMap *tma_store_map) {
    /*
        this kernel performs B = A using TMA operations
        only 1 thread will be required

        smem layout: |---data---|--|--mbarrier--|
        data is 16 byte aligned
        we might need padding to make sure mbarrier is 8-byte aligned

        Completion tech:
        TMA loading: mbarrier
        TMA storing: commit_group
    */

    extern __shared__ __align__(16) float smem[];

    constexpr int tile_elements = TILE_M * TILE_N;
    constexpr int tile_bytes = TILE_M * TILE_N * sizeof(float);

    // compute mbarrier position, the address must be divible by 8
    constexpr int mbar_load_offset_bytes = ((tile_bytes + 7) / 8) * 8;
    constexpr int mbar_load_offset_index = mbar_load_offset_bytes / sizeof(float);

    // get mbarrier address
    uint64_t *mbar_obj = reinterpret_cast<uint64_t *>(&smem[mbar_load_offset_index]);

    // convert address to smem address
    const uintptr_t mbar_load_addr = (uintptr_t)__cvta_generic_to_shared(mbar_obj);
    const uintptr_t smem_addr = (uintptr_t)__cvta_generic_to_shared(smem);

    // verify the alignment
    assert((smem_addr % 16) == 0);
    assert((mbar_load_addr % 8) == 0);

    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // check boundary
    if (tile_row * TILE_M >= M || tile_col * TILE_N >= N) {
        return;
    }

    printf("Block (%d, %d): Starting TMA copy for tile [%d:%d, %d:%d]\n",
            tile_row, tile_col,
            tile_row * TILE_M, (tile_row + 1) * TILE_M,
            tile_col * TILE_N, (tile_col + 1) * TILE_N
        );

    // initialize mbarrier
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;\n\t"
        : : "l"(mbar_load_addr), "r"(1) : "memory"
    );

    constexpr int transaction_bytes = TILE_M * TILE_N * sizeof(float);
    unsigned long long load_token = 0u;
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n\t"
        : "=l"(load_token) : "l"(mbar_load_addr), "r"(transaction_bytes) : "memory"
    );

    printf("TMA Load: using coordinates (%d, %d)\n", tile_col, tile_row);

    // issue TMA load operation
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n\t"
        :
        : "l"(smem_addr),        // destination: shared memory
          "l"(tma_load_map),     // TMA descriptor for matrix A
          "r"(tile_row),         // tile coordinate X
          "r"(tile_col),         // tile coordinate Y
          "l"(mbar_load_addr)    // load mbarrier
        : "memory"
    );

    // wait for TMA
    unsigned load_done = 0u;
    do {
        asm volatile(
            "{ .reg .pred p;                                        \n\t"
            "  mbarrier.test_wait.shared::cta.b64 p, [%1], %2;      \n\t"
            "  selp.u32 %0, 1, 0, p;                               \n\t"
            "}                                                     \n\t"
            : "=r"(load_done) : "l"(mbar_load_addr), "l"(load_token) : "memory"
        );
    } while (!load_done);

    printf("TMA load completed\n");

    // TMA store
    printf("TMA Store: using coordinates (%d, %d)\n", tile_col, tile_row);


    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];\n\t"
        :
        :"l"(tma_store_map),
        "r"(tile_row),
        "r"(tile_col),
        "l"(smem_addr)
        : "memory"
    );


    // commit the TMA store operation
    asm volatile(
        "cp.async.bulk.commit_group;\n\t"
        ::: "memory"
    );

    // wait for commit group completion
    asm volatile(
        "cp.async.bulk.wait_group 0;\n\t"
        ::: "memory"
    );

    __threadfence();

    printf("After wait_group\n");

    printf("TMA store completed successfully\n");

    // DEBUG: Print first few elements from shared memory to verify load
    printf("Shared memory verification: [%.1f, %.1f, %.1f, %.1f]\n",
       smem[0], smem[1], smem[2], smem[3]);

    // No need for __syncthreads() with single thread per block
    printf("Block (%d,%d): TMA copy completed successfully\n", blockIdx.x, blockIdx.y);
    printf("  Expected global range: A[%d:%d, %d:%d]\n",
        tile_row * TILE_M, (tile_row + 1) * TILE_M,
        tile_col * TILE_N, (tile_col + 1) * TILE_N);
    printf("  Copied data: %.2f, %.2f, %.2f, %.2f\n",
        smem[0], smem[1], smem[2], smem[3]);
}

// Create TMA descriptor for a matrix
CUtensorMap create_tma_descriptor(float *matrix_ptr, const char* name) {
    alignas(64) CUtensorMap tma_map;

    // Verify alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(matrix_ptr);
    printf("Creating TMA descriptor for %s:\n", name);
    printf("  Address: 0x%lx (16-byte aligned: %s)\n",
           addr, (addr % 16 == 0) ? "✓" : "✗");
    assert((addr % 16) == 0);

    // Setup parameters
    uint64_t tensor_shape[2] = {static_cast<uint64_t>(M), static_cast<uint64_t>(N)};

    uint64_t row_stride_bytes = N * sizeof(float);
    printf("  Row stride: %lu bytes (16-byte aligned: %s)\n",
           row_stride_bytes, (row_stride_bytes % 16 == 0) ? "✓" : "✗");
    assert((row_stride_bytes % 16) == 0);

    uint64_t tensor_stride[2] = {row_stride_bytes, sizeof(float)};
    uint32_t box_size[2] = {TILE_M, TILE_N};
    uint32_t element_strides[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,                          // 2D tensor
        matrix_ptr,                 // Matrix pointer
        tensor_shape,               // Dimensions
        tensor_stride,              // Strides
        box_size,                   // Tile size
        element_strides,            // Element order
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (result != CUDA_SUCCESS) {
        printf("ERROR: cuTensorMapEncodeTiled failed for %s with code %d\n", name, result);
        exit(EXIT_FAILURE);
    }

    printf("  ✅ TMA descriptor created successfully\n");
    return tma_map;
}

void init_matrix(float *matrix, int rows, int cols, float base_value) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = base_value + static_cast<float>(i % 100);
    }
}

bool verify_copy(float *A, float *B, int rows, int cols, float tolerance = 1e-6f) {
    printf("Verifying matrix copy...\n");

    int errors = 0;
    for (int i = 0; i < rows * cols && errors < 10; i++) {
        if (abs(A[i] - B[i]) > tolerance) {
            printf("  Mismatch at [%d]: A=%.6f, B=%.6f\n", i, A[i], B[i]);
            errors++;
        }
    }

    if (errors == 0) {
        printf("  ✅ Matrix copy verification PASSED\n");
        return true;
    } else {
        printf("  ❌ Matrix copy verification FAILED (%d errors)\n", errors);
        return false;
    }
}

__global__ void two_stage() {
    initialize smem that can store 2 tiles of A and B
    prefetch first tile of A and B
    mbarrier.wait
    read_idx = 0
    write_idx = 1

    loop:
        fetch the next tile and set to write_idx

        do computation based on read_idx

        computation notify
        fetch next tile notify

        swap read_idx, write_idx


}

int main() {
    printf("=== TMA Matrix Copy Test (A → B) ===\n");
    printf("Matrix size: %d × %d, Tile size: %d × %d\n", M, N, TILE_M, TILE_N);

    // Initialize CUDA
    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
        printf("cuInit failed with code %d\n", cu_result);
        return 1;
    }

    // Verify H100
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);

    size_t matrix_size = M * N * sizeof(float);

    // Allocate host matrices
    float *A_h = (float *)malloc(matrix_size);
    float *B_h = (float *)malloc(matrix_size);
    if (!A_h || !B_h) {
        printf("Failed to allocate host memory\n");
        return 1;
    }

    // Initialize matrices
    init_matrix(A_h, M, N, 1.0f);  // A: values 1.0 to 101.0
    for (int i = 0; i < M * N; i++) {
        B_h[i] = 0.0f;
    } // initialize B to 0.0

    // Allocate device matrices
    float *A_d, *B_d;
    cudaMalloc(&A_d, matrix_size);
    cudaMalloc(&B_d, matrix_size);

    // Verify device alignment
    assert((reinterpret_cast<uintptr_t>(A_d) % 16) == 0);
    assert((reinterpret_cast<uintptr_t>(B_d) % 16) == 0);

    // Copy data to device
    cudaMemcpy(A_d, A_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, matrix_size, cudaMemcpyHostToDevice); // Zero-initialized B

    // Create TMA descriptors for both matrices
    CUtensorMap tma_load_map = create_tma_descriptor(A_d, "Matrix A (source)");
    CUtensorMap tma_store_map = create_tma_descriptor(B_d, "Matrix B (destination)");

    // Copy TMA descriptors to device
    CUtensorMap *d_tma_load, *d_tma_store;
    cudaMalloc(&d_tma_load, sizeof(CUtensorMap));
    cudaMalloc(&d_tma_store, sizeof(CUtensorMap));
    cudaMemcpy(d_tma_load, &tma_load_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tma_store, &tma_store_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    // Launch configuration (OPTIMIZED: Single thread per block)
    dim3 block_size(1, 1, 1);      // Single thread per block for TMA efficiency
    dim3 grid_size((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);

    // Calculate shared memory: tile data + 1 mbarrier (only for load)
    constexpr int tile_bytes = TILE_M * TILE_N * sizeof(float);
    constexpr int mbar_load_offset = ((tile_bytes + 7) / 8) * 8;
    constexpr int total_shmem = mbar_load_offset + 8; // +8 for one load mbarrier

    printf("\nLaunching TMA matrix copy kernel...\n");
    tma_matrix_copy_kernel<<<grid_size, block_size, total_shmem>>>(d_tma_load, d_tma_store);
    cudaDeviceSynchronize();

    printf("\n✅ TMA matrix copy kernel completed successfully!\n");

    // Copy result back to host and verify
    cudaMemcpy(B_h, B_d, matrix_size, cudaMemcpyDeviceToHost);

    bool success = verify_copy(A_h, B_h, M, N);

    // Print sample values
    printf("\nSample values:\n");
    printf("  A[0:3] = %.2f, %.2f, %.2f, %.2f\n", A_h[0], A_h[1], A_h[2], A_h[3]);
    printf("  B[0:3] = %.2f, %.2f, %.2f, %.2f\n", B_h[0], B_h[1], B_h[2], B_h[3]);

    // Cleanup
    free(A_h);
    free(B_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(d_tma_load);
    cudaFree(d_tma_store);

    return success ? 0 : 1;
}