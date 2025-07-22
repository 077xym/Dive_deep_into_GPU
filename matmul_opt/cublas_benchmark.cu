#include <cublas_v2.h>
#include <stdlib.h>
#include <stdio.h>

#define M 4096
#define K 4096
#define N 4096

void init_mat(float *a, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        a[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *a, *b, *c;

    cudaMallocManaged(&a, M * K * sizeof(float));
    cudaMallocManaged(&b, K * N * sizeof(float));
    cudaMallocManaged(&c, M * N * sizeof(float));

    init_mat(a, M, K);
    init_mat(b, K, N);
    init_mat(c, M, N);

    /*
    a = (M, K), b = (K, N), c = (M, N)

    in BLAS, we want output to be c = (N, M) = (N, K) * (K, M)
    which is how BLAS view b and a

    So we have A as (N, K)
               B as (K, M)
    */
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // don't need to transpose
        N, // number of rows of matrix A, which is b^T in our case
        M, // number of cols of matrix B, which is a^T in our case
        K, // inner dimension
        &alpha,
        b, N, // A, and lead dim of A
        a, K, // B, and lead dim of B
        &beta,
        c, N // output, and lead dim of output
   );

   printf("%f", c[0]);

   return 1;
}