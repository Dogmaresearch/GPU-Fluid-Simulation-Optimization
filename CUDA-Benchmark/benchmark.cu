#include <iostream>
#include <cuda_runtime.h>

// Numero elementi
#define N (1 << 24)

// ============================================================
// BASELINE KERNEL
// ============================================================
_global_ void baseline_kernel(float* C,
                                const float* A,
                                const float* B,
                                int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        C[idx] = A[idx] * B[idx] + 0.5f;
    }
}

// ============================================================
// DOGMA OPTIMIZED KERNEL
// ============================================================
/*
 * Kernel: Dogma Optimized Vectorized Compute Kernel
 * Author: Dogma
 *
 * Uses float4 to reduce memory access and improve performance
 */
_global_ void dogma_optimized_kernel(float4* _restrict_ C,
                                       const float4* _restrict_ A,
                                       const float4* _restrict_ B,
                                       int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float4 a = A[idx];
        float4 b = B[idx];

        float4 r;

        r.x = a.x * b.x + 0.5f;
        r.y = a.y * b.y + 0.5f;
        r.z = a.z * b.z + 0.5f;
        r.w = a.w * b.w + 0.5f;

        C[idx] = r;
    }
}

// ============================================================
// MAIN
// ============================================================
int main()
{
    size_t size = N * sizeof(float);

    // Host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    float ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // =======================
    // BASELINE TEST
    // =======================
    cudaEventRecord(start);

    baseline_kernel<<<gridSize, blockSize>>>(d_C, d_A, d_B, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Baseline Time: " << ms << " ms\n";

    // =======================
    // OPTIMIZED TEST
    // =======================
    cudaEventRecord(start);

    dogma_optimized_kernel<<<gridSize / 4, blockSize>>>(
        (float4*)d_C,
        (float4*)d_A,
        (float4*)d_B,
        N / 4
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "Dogma Optimized Time: " << ms << " ms\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);

    return 0;
}