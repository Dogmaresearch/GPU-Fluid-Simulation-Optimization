#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

using carro = float;

#define N (1 << 26)
#define ITERAZIONI 500
#define BLOCK_SIZE 256
#define INNER_REPEAT 100

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
__global__ void baseline_kernel(
    carro* __restrict__ C,
    const carro* __restrict__ A,
    const carro* __restrict__ B,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        carro val = 0.0f;

        #pragma unroll 8
        for (int k = 0; k < INNER_REPEAT; ++k) {
            val += A[idx] * B[idx];
        }

        C[idx] = val;
    }
}

// ============================================================
// KERNEL VETTORIALIZZATO (float4)
// ============================================================
__global__ void vectorized_kernel(
    float4* __restrict__ C,
    const float4* __restrict__ A,
    const float4* __restrict__ B,
    int n4)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n4) {
        const float4 a = A[idx];
        const float4 b = B[idx];
        float4 r;

        r.x = 0.0f;
        r.y = 0.0f;
        r.z = 0.0f;
        r.w = 0.0f;

        #pragma unroll 8
        for (int k = 0; k < INNER_REPEAT; ++k) {
            r.x += a.x * b.x;
            r.y += a.y * b.y;
            r.z += a.z * b.z;
            r.w += a.w * b.w;
        }

        C[idx] = r;
    }
}

// ============================================================
// KERNEL CON SHARED MEMORY
// ============================================================
__global__ void shared_memory_kernel(
    carro* __restrict__ C,
    const carro* __restrict__ A,
    const carro* __restrict__ B,
    int n)
{
    __shared__ carro sA[BLOCK_SIZE];
    __shared__ carro sB[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        sA[tid] = A[idx];
        sB[tid] = B[idx];
    }

    __syncthreads();

    if (idx < n) {
        carro val = 0.0f;

        #pragma unroll 8
        for (int k = 0; k < INNER_REPEAT; ++k) {
            val += sA[tid] * sB[tid];
        }

        C[idx] = val;
    }
}

// ============================================================
// HELPER DI VALIDAZIONE
// ============================================================
bool valida_array(
    const carro* riferimento,
    const carro* candidato,
    int n,
    const char* etichetta)
{
    for (int i = 0; i < n; ++i) {
        if (std::fabs(riferimento[i] - candidato[i]) > 1e-5f) {
            std::cerr << etichetta << " mismatch at index " << i
                      << " | reference = " << riferimento[i]
                      << " | candidate = " << candidato[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================
// TIMING HELPER BASELINE
// ============================================================
float esegui_baseline(
    carro* d_C,
    const carro* d_A,
    const carro* d_B,
    int n,
    dim3 griglia,
    dim3 blocco,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERAZIONI; ++i) {
        baseline_kernel<<<griglia, blocco>>>(d_C, d_A, d_B, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    return ms / static_cast<float>(ITERAZIONI);
}

// ============================================================
// TIMING HELPER VECTORIZED
// ============================================================
float esegui_vectorized(
    carro* d_C,
    const carro* d_A,
    const carro* d_B,
    int n4,
    dim3 griglia,
    dim3 blocco,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERAZIONI; ++i) {
        vectorized_kernel<<<griglia, blocco>>>(
            reinterpret_cast<float4*>(d_C),
            reinterpret_cast<const float4*>(d_A),
            reinterpret_cast<const float4*>(d_B),
            n4
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    return ms / static_cast<float>(ITERAZIONI);
}

// ============================================================
// TIMING HELPER SHARED
// ============================================================
float esegui_shared(
    carro* d_C,
    const carro* d_A,
    const carro* d_B,
    int n,
    dim3 griglia,
    dim3 blocco,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERAZIONI; ++i) {
        shared_memory_kernel<<<griglia, blocco>>>(d_C, d_A, d_B, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    return ms / static_cast<float>(ITERAZIONI);
}

int main()
{
    static_assert(N % 4 == 0, "N must be divisible by 4 for float4 vectorization.");

    const size_t dimensioni = static_cast<size_t>(N) * sizeof(carro);
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
    // CONFIGURAZIONE BASE
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
    // TEST PRINCIPALI
    // ============================================================
    const float baseline_ms =
        esegui_baseline(d_C_baseline, d_A, d_B, N, griglia, blocco, start, stop);

    const float vectorized_ms =
        esegui_vectorized(d_C_vectorized, d_A, d_B, n4, griglia_vectorized, blocco, start, stop);

    const float shared_ms =
        esegui_shared(d_C_shared, d_A, d_B, N, griglia, blocco, start, stop);

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
    const bool valid_vectorized =
        valida_array(h_C_baseline, h_C_vectorized, N, "Vectorized");

    const bool valid_shared =
        valida_array(h_C_baseline, h_C_shared, N, "Shared");

    const bool validazione_completa = valid_vectorized && valid_shared;

    // ============================================================
    // METRICHE
    // ============================================================
    const float speedup_vectorized =
        (vectorized_ms > 0.0f) ? (baseline_ms / vectorized_ms) : 0.0f;

    const float speedup_shared =
        (shared_ms > 0.0f) ? (baseline_ms / shared_ms) : 0.0f;

    const float improvement_vectorized =
        (baseline_ms > 0.0f)
            ? ((baseline_ms - vectorized_ms) / baseline_ms) * 100.0f
            : 0.0f;

    const float improvement_shared =
        (baseline_ms > 0.0f)
            ? ((baseline_ms - shared_ms) / baseline_ms) * 100.0f
            : 0.0f;

    const size_t bytes_processed = static_cast<size_t>(N) * sizeof(carro);

    const float bandwidth_vectorized =
        (vectorized_ms > 0.0f)
            ? (bytes_processed / (vectorized_ms / 1000.0f)) / 1e9f
            : 0.0f;

    const float bandwidth_shared =
        (shared_ms > 0.0f)
            ? (bytes_processed / (shared_ms / 1000.0f)) / 1e9f
            : 0.0f;

    // ============================================================
    // TUNING BLOCK SIZE SUL KERNEL VETTORIALIZZATO
    // ============================================================
    const int block_sizes[] = {64, 128, 256, 512};

    float best_vectorized_time = std::numeric_limits<float>::max();
    int best_vectorized_block = -1;

    std::cout << std::fixed << std::setprecision(5);

    for (int bs : block_sizes) {
        dim3 tuning_blocco(bs);
        dim3 tuning_griglia((n4 + bs - 1) / bs);

        const float time_ms =
            esegui_vectorized(
                d_C_vectorized,
                d_A,
                d_B,
                n4,
                tuning_griglia,
                tuning_blocco,
                start,
                stop);

        if (time_ms < best_vectorized_time) {
            best_vectorized_time = time_ms;
            best_vectorized_block = bs;
        }
    }

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
    std::cout << "Approx. Shared Memory Bandwidth: " << bandwidth_shared << " GB/s\n\n";

    std::cout << "=== Vectorized Block Size Tuning ===\n";
    for (int bs : block_sizes) {
        dim3 tuning_blocco(bs);
        dim3 tuning_griglia((n4 + bs - 1) / bs);

        const float time_ms =
            esegui_vectorized(
                d_C_vectorized,
                d_A,
                d_B,
                n4,
                tuning_griglia,
                tuning_blocco,
                start,
                stop);

        std::cout << "Block Size " << bs << " -> " << time_ms << " ms\n";
    }

    std::cout << "\nBest Vectorized Block Size: " << best_vectorized_block << "\n";
    std::cout << "Best Vectorized Time: " << best_vectorized_time << " ms\n";

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

    return validazione_completa ? EXIT_SUCCESS : EXIT_FAILURE;
}
