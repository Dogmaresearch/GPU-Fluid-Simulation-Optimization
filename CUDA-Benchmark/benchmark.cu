#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

#define N (1 << 24)
#define ITERATIONS 100

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << std::endl;               \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


// ============================================================
// BASELINE KERNEL
// ============================================================
__global__ void baseline_kernel(float* C, const float* A, const float* B, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] * B[idx] + 0.5f;
    }
}


// ============================================================
// DOGMA OPTIMIZED KERNEL
// Uses float4 vectorized memory access to improve coalescing
// and reduce global memory instructions.
// ============================================================
__global__ void dogma_optimized_kernel(
    float4* __restrict__ C,
    const float4* __restrict__ A,
    const float4* __restrict__ B,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
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


int main()
{
    const size_t size = N * sizeof(float);

    // ----------------------------
    // Host memory
    // ----------------------------
    float* h_A = static_cast<float*>(std::malloc(size));
    float* h_B = static_cast<float*>(std::malloc(size));
    float* h_C_baseline = static_cast<float*>(std::malloc(size));
    float* h_C_optimized = static_cast<float*>(std::malloc(size));

    if (!h_A || !h_B || !h_C_baseline || !h_C_optimized) {
        std::cerr << "Host memory allocation failed." << std::endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // ----------------------------
    // Device memory
    // ----------------------------
    float *d_A, *d_B, *d_C_baseline, *d_C_optimized;

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C_baseline, size));
    CUDA_CHECK(cudaMalloc(&d_C_optimized, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // ----------------------------
    // Launch configuration
    // ----------------------------
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    int optimizedGridSize = ((N / 4) + blockSize - 1) / blockSize;

    // ----------------------------
    // CUDA events
    // ----------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float baseline_ms = 0.0f;
    float optimized_ms = 0.0f;

    // ============================================================
    // WARMUP
    // Warm up the GPU to reduce cold-start effects in measurements
    // ============================================================
    baseline_kernel<<<gridSize, blockSize>>>(d_C_baseline, d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dogma_optimized_kernel<<<optimizedGridSize, blockSize>>>(
        reinterpret_cast<float4*>(d_C_optimized),
        reinterpret_cast<const float4*>(d_A),
        reinterpret_cast<const float4*>(d_B),
        N / 4
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ============================================================
    // BASELINE TEST
    // ============================================================
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++) {
        baseline_kernel<<<gridSize, blockSize>>>(d_C_baseline, d_A, d_B, N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&baseline_ms, start, stop));

    baseline_ms /= ITERATIONS;

    // ============================================================
    // OPTIMIZED TEST
    // ============================================================
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++) {
        dogma_optimized_kernel<<<optimizedGridSize, blockSize>>>(
            reinterpret_cast<float4*>(d_C_optimized),
            reinterpret_cast<const float4*>(d_A),
            reinterpret_cast<const float4*>(d_B),
            N / 4
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&optimized_ms, start, stop));

    optimized_ms /= ITERATIONS;

    // Make sure everything is finished
    CUDA_CHECK(cudaDeviceSynchronize());

    // ============================================================
    // COPY RESULTS BACK
    // ============================================================
    CUDA_CHECK(cudaMemcpy(h_C_baseline, d_C_baseline, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_optimized, d_C_optimized, size, cudaMemcpyDeviceToHost));

    // ============================================================
    // RESULT VERIFICATION
    // ============================================================
    bool valid = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_C_baseline[i] - h_C_optimized[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i
                      << " | baseline = " << h_C_baseline[i]
                      << " | optimized = " << h_C_optimized[i]
                      << std::endl;
            valid = false;
            break;
        }
    }

    // ============================================================
    // PERFORMANCE METRICS
    // ============================================================
    float speedup = (optimized_ms > 0.0f) ? baseline_ms / optimized_ms : 0.0f;
    float improvement = (baseline_ms > 0.0f)
        ? ((baseline_ms - optimized_ms) / baseline_ms) * 100.0f
        : 0.0f;

    size_t bytesProcessed = static_cast<size_t>(N) * sizeof(float);
    float bandwidthGBs = (optimized_ms > 0.0f)
        ? (bytesProcessed / (optimized_ms / 1000.0f)) / 1e9f
        : 0.0f;

    // ============================================================
    // OUTPUT
    // ============================================================
    std::cout << "Validation: " << (valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "Iterations: " << ITERATIONS << "\n";
    std::cout << "Elements processed: " << N << "\n";
    std::cout << "Bytes processed: " << bytesProcessed << "\n\n";

    std::cout << "Baseline Time: " << baseline_ms << " ms\n";
    std::cout << "Optimized Kernel Time: " << optimized_ms << " ms\n";
    std::cout << "Speedup: " << speedup << "x\n";
    std::cout << "Performance Improvement: " << improvement << "%\n";
    std::cout << "Approx. Memory Bandwidth: " << bandwidthGBs << " GB/s\n";

    // ============================================================
    // CLEANUP
    // ============================================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_baseline));
    CUDA_CHECK(cudaFree(d_C_optimized));

    std::free(h_A);
    std::free(h_B);
    std::free(h_C_baseline);
    std::free(h_C_optimized);

    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
