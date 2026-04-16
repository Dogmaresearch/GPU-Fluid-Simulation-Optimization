#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

using carro = float;

#define N (1 << 24)
#define ITERAZIONI 100
#define BLOCK_SIZE 256

#define CUDA_CHECK(chiamata)                                                      \
do {                                                                              \
    cudaError_t err = (chiamata);                                                 \
    if (err != cudaSuccess) {                                                     \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                    \
                  << " at line " << __LINE__ << std::endl;                        \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

// ============================================================
// KERNEL DI BASE
// ============================================================
__global__ void baseline_kernel(carro* C, const carro* A, const carro* B, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] * B[idx] + 0.5f;
    }
}

// ============================================================
// KERNEL OTTIMIZZATO VETTORIALIZZATO (float4)
// ============================================================
__global__ void vectorized_kernel(
    float4* __restrict__ C,
    const float4* __restrict__ A,
    const float4* __restrict__ B,
    int n4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n4) {
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
// KERNEL CON SHARED MEMORY
// ============================================================
__global__ void shared_memory_kernel(
    carro* C,
    const carro* A,
    const carro* B,
    int n)
{
    __shared__ carro sA[BLOCK_SIZE];
    __shared__ carro sB[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        sA[tid] = A[idx];
        sB[tid] = B[idx];
    }

    __syncthreads();

    if (idx < n) {
        C[idx] = sA[tid] * sB[tid] + 0.5f;
    }
}

int main()
{
    const size_t dimensioni = N * sizeof(carro);
    const int n4 = N / 4;

    // ============================================================
    // MEMORIA HOST
    // ============================================================
    carro* h_A = static_cast<carro*>(std::malloc(dimensioni));
    carro* h_B = static_cast<carro*>(std::malloc(dimensioni));
    carro* h_C_baseline = static_cast<carro*>(std::malloc(dimensioni));
    carro* h_C_vectorized = static_cast<carro*>(std::malloc(dimensioni));
    carro* h_C_shared = static_cast<carro*>(std::malloc(dimensioni));

    if (!h_A || !h_B || !h_C_baseline || !h_C_vectorized || !h_C_shared) {
        std::cerr << "Host memory allocation failed." << std::endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // ============================================================
    // MEMORIA DEVICE
    // ============================================================
    carro *d_A, *d_B, *d_C_baseline, *d_C_vectorized, *d_C_shared;

    CUDA_CHECK(cudaMalloc(&d_A, dimensioni));
    CUDA_CHECK(cudaMalloc(&d_B, dimensioni));
    CUDA_CHECK(cudaMalloc(&d_C_baseline, dimensioni));
    CUDA_CHECK(cudaMalloc(&d_C_vectorized, dimensioni));
    CUDA_CHECK(cudaMalloc(&d_C_shared, dimensioni));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, dimensioni, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, dimensioni, cudaMemcpyHostToDevice));

    // ============================================================
    // CONFIGURAZIONE DI LANCIO
    // ============================================================
    dim3 blocco(BLOCK_SIZE);
    dim3 griglia((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 griglia_vectorized((n4 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ============================================================
    // EVENTI CUDA
    // ============================================================
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float baseline_ms = 0.0f;
    float vectorized_ms = 0.0f;
    float shared_ms = 0.0f;

    // ============================================================
    // WARMUP
    // ============================================================
    baseline_kernel<<<griglia, blocco>>>(d_C_baseline, d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vectorized_kernel<<<griglia_vectorized, blocco>>>(
        reinterpret_cast<float4*>(d_C_vectorized),
        reinterpret_cast<const float4*>(d_A),
        reinterpret_cast<const float4*>(d_B),
        n4
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    shared_memory_kernel<<<griglia, blocco>>>(d_C_shared, d_A, d_B, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ============================================================
    // TEST BASELINE
    // ============================================================
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERAZIONI; ++i) {
        baseline_kernel<<<griglia, blocco>>>(d_C_baseline, d_A, d_B, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&baseline_ms, start, stop));
    baseline_ms /= ITERAZIONI;

    // ============================================================
    // TEST VECTORIZED
    // ============================================================
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERAZIONI; ++i) {
        vectorized_kernel<<<griglia_vectorized, blocco>>>(
            reinterpret_cast<float4*>(d_C_vectorized),
            reinterpret_cast<const float4*>(d_A),
            reinterpret_cast<const float4*>(d_B),
            n4
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&vectorized_ms, start, stop));
    vectorized_ms /= ITERAZIONI;

    // ============================================================
    // TEST SHARED MEMORY
    // ============================================================
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERAZIONI; ++i) {
        shared_memory_kernel<<<griglia, blocco>>>(d_C_shared, d_A, d_B, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&shared_ms, start, stop));
    shared_ms /= ITERAZIONI;

    CUDA_CHECK(cudaDeviceSynchronize());

    // ============================================================
    // COPIA RISULTATI SU HOST
    // ============================================================
    CUDA_CHECK(cudaMemcpy(h_C_baseline, d_C_baseline, dimensioni, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_vectorized, d_C_vectorized, dimensioni, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_shared, d_C_shared, dimensioni, cudaMemcpyDeviceToHost));

    // ============================================================
    // VALIDAZIONE
    // ============================================================
    bool valid_vectorized = true;
    bool valid_shared = true;

    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_C_baseline[i] - h_C_vectorized[i]) > 1e-5f) {
            std::cerr << "Vectorized mismatch at index " << i
                      << " | baseline = " << h_C_baseline[i]
                      << " | vectorized = " << h_C_vectorized[i]
                      << std::endl;
            valid_vectorized = false;
            break;
        }
    }

    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_C_baseline[i] - h_C_shared[i]) > 1e-5f) {
            std::cerr << "Shared-memory mismatch at index " << i
                      << " | baseline = " << h_C_baseline[i]
                      << " | shared = " << h_C_shared[i]
                      << std::endl;
            valid_shared = false;
            break;
        }
    }

    bool validazione_completa = valid_vectorized && valid_shared;

    // ============================================================
    // METRICHE
    // ============================================================
    float speedup_vectorized =
        (vectorized_ms > 0.0f) ? (baseline_ms / vectorized_ms) : 0.0f;

    float speedup_shared =
        (shared_ms > 0.0f) ? (baseline_ms / shared_ms) : 0.0f;

    float improvement_vectorized =
        (baseline_ms > 0.0f) ? ((baseline_ms - vectorized_ms) / baseline_ms) * 100.0f : 0.0f;

    float improvement_shared =
        (baseline_ms > 0.0f) ? ((baseline_ms - shared_ms) / baseline_ms) * 100.0f : 0.0f;

    size_t bytes_processed = static_cast<size_t>(N) * sizeof(carro);

    float bandwidth_vectorized =
        (vectorized_ms > 0.0f)
            ? (bytes_processed / (vectorized_ms / 1000.0f)) / 1e9f
            : 0.0f;

    float bandwidth_shared =
        (shared_ms > 0.0f)
            ? (bytes_processed / (shared_ms / 1000.0f)) / 1e9f
            : 0.0f;

    // ============================================================
    // OUTPUT
    // ============================================================
    std::cout << "=== GPU Kernel Comparison ===\n\n";

    std::cout << "Validation: " << (validazione_completa ? "PASSED" : "FAILED") << "\n";
    std::cout << "Iterations: " << ITERAZIONI << "\n";
    std::cout << "Elements processed: " << N << "\n";
    std::cout << "Bytes processed: " << bytes_processed << "\n\n";

    std::cout << "Baseline Time: " << baseline_ms << " ms\n";
    std::cout << "Vectorized Kernel Time: " << vectorized_ms << " ms\n";
    std::cout << "Shared Memory Kernel Time: " << shared_ms << " ms\n\n";

    std::cout << "Vectorized Speedup: " << speedup_vectorized << "x\n";
    std::cout << "Shared Memory Speedup: " << speedup_shared << "x\n\n";

    std::cout << "Vectorized Improvement: " << improvement_vectorized << "%\n";
    std::cout << "Shared Memory Improvement: " << improvement_shared << "%\n\n";

    std::cout << "Approx. Vectorized Bandwidth: " << bandwidth_vectorized << " GB/s\n";
    std::cout << "Approx. Shared Memory Bandwidth: " << bandwidth_shared << " GB/s\n";

    // ============================================================
    // PULIZIA
    // ============================================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_baseline));
    CUDA_CHECK(cudaFree(d_C_vectorized));
    CUDA_CHECK(cudaFree(d_C_shared));

    std::free(h_A);
    std::free(h_B);
    std::free(h_C_baseline);
    std::free(h_C_vectorized);
    std::free(h_C_shared);

// ============================================================
// BLOCK SIZE TUNING
// ============================================================

std::cout << "\n=== Block Size Tuning ===\n";

int block_sizes[] = {64, 128, 256, 512};

for (int bs : block_sizes) {

    dim3 blocco(bs);
    dim3 griglia((N + bs - 1) / bs);

    float time_ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERAZIONI; ++i) {
        baseline_kernel<<<griglia, blocco>>>(d_C_baseline, d_A, d_B, N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    time_ms /= ITERAZIONI;

    std::cout << "Block Size " << bs << " -> " << time_ms << " ms\n";
}

    return validazione_completa ? EXIT_SUCCESS : EXIT_FAILURE;
}
