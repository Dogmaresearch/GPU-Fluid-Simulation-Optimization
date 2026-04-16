#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using value_t = float;

#define N (1 << 26)
#define ITERATIONS 500
#define BLOCK_SIZE 256
#define INNER_REPEAT 100

#define CUDA_CHECK(call)                                                          \
do {                                                                              \
    cudaError_t err = (call);                                                     \
    if (err != cudaSuccess) {                                                     \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                    \
                  << " at line " << __LINE__ << std::endl;                        \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

// ============================================================
// BASELINE KERNEL
// One thread processes one float element.
// ============================================================
__global__ void baseline_kernel(
    value_t* c,
    const value_t* a,
    const value_t* b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        value_t val = 0.0f;

        #pragma unroll 4
        for (int k = 0; k < INNER_REPEAT; ++k) {
            val += a[idx] * b[idx];
        }

        c[idx] = val;
    }
}

// ============================================================
// VECTORIZED KERNEL
// One thread processes one float4.
// ============================================================
__global__ void vectorized_kernel(
    float4* __restrict__ c,
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    int n4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n4) {
        float4 av = a[idx];
        float4 bv = b[idx];
        float4 rv;

        rv.x = 0.0f;
        rv.y = 0.0f;
        rv.z = 0.0f;
        rv.w = 0.0f;

        #pragma unroll 4
        for (int k = 0; k < INNER_REPEAT; ++k) {
            rv.x += av.x * bv.x;
            rv.y += av.y * bv.y;
            rv.z += av.z * bv.z;
            rv.w += av.w * bv.w;
        }

        c[idx] = rv;
    }
}

// ============================================================
// SHARED MEMORY KERNEL
// Loads input values into shared memory before repeated use.
// This is included for comparison, even if it may not help much
// for this workload.
// ============================================================
__global__ void shared_memory_kernel(
    value_t* c,
    const value_t* a,
    const value_t* b,
    int n)
{
    __shared__ value_t sA[BLOCK_SIZE];
    __shared__ value_t sB[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        sA[tid] = a[idx];
        sB[tid] = b[idx];
    }

    __syncthreads();

    if (idx < n) {
        value_t val = 0.0f;

        #pragma unroll 4
        for (int k = 0; k < INNER_REPEAT; ++k) {
            val += sA[tid] * sB[tid];
        }

        c[idx] = val;
    }
}

// ============================================================
// DOGMA ULTRA KERNEL
// Vectorized + Grid-Stride Loop + Unrolling
// ============================================================
__global__ void dogma_ultra_kernel(
    float4* __restrict__ c,
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    int n4)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n4;
         idx += blockDim.x * gridDim.x)
    {
        float4 av = a[idx];
        float4 bv = b[idx];
        float4 rv;

        rv.x = 0.0f;
        rv.y = 0.0f;
        rv.z = 0.0f;
        rv.w = 0.0f;

        #pragma unroll 10
        for (int k = 0; k < INNER_REPEAT; ++k) {
            rv.x += av.x * bv.x;
            rv.y += av.y * bv.y;
            rv.z += av.z * bv.z;
            rv.w += av.w * bv.w;
        }

        c[idx] = rv;
    }
}

// ============================================================
// Utility: compare two host arrays
// ============================================================
bool validate_arrays(
    const value_t* reference,
    const value_t* candidate,
    int n,
    const char* label)
{
    for (int i = 0; i < n; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > 1e-5f) {
            std::cerr << label << " mismatch at index " << i
                      << " | reference = " << reference[i]
                      << " | candidate = " << candidate[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================
// Utility: run baseline timing
// ============================================================
float run_baseline(
    value_t* d_c,
    const value_t* d_a,
    const value_t* d_b,
    int n,
    dim3 grid,
    dim3 block,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; ++i) {
        baseline_kernel<<<grid, block>>>(d_c, d_a, d_b, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    return ms / static_cast<float>(ITERATIONS);
}

// ============================================================
// Utility: run vectorized timing
// ============================================================
float run_vectorized(
    value_t* d_c,
    const value_t* d_a,
    const value_t* d_b,
    int n4,
    dim3 grid,
    dim3 block,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; ++i) {
        vectorized_kernel<<<grid, block>>>(
            reinterpret_cast<float4*>(d_c),
            reinterpret_cast<const float4*>(d_a),
            reinterpret_cast<const float4*>(d_b),
            n4);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    return ms / static_cast<float>(ITERATIONS);
}

// ============================================================
// Utility: run shared memory timing
// ============================================================
float run_shared(
    value_t* d_c,
    const value_t* d_a,
    const value_t* d_b,
    int n,
    dim3 grid,
    dim3 block,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; ++i) {
        shared_memory_kernel<<<grid, block>>>(d_c, d_a, d_b, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    return ms / static_cast<float>(ITERATIONS);
}

// ============================================================
// Utility: run dogma ultra timing
// ============================================================
float run_dogma_ultra(
    value_t* d_c,
    const value_t* d_a,
    const value_t* d_b,
    int n4,
    dim3 grid,
    dim3 block,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; ++i) {
        dogma_ultra_kernel<<<grid, block>>>(
            reinterpret_cast<float4*>(d_c),
            reinterpret_cast<const float4*>(d_a),
            reinterpret_cast<const float4*>(d_b),
            n4);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    return ms / static_cast<float>(ITERATIONS);
}

int main()
{
    static_assert(N % 4 == 0, "N must be divisible by 4 for float4 vectorization.");

    const int n = N;
    const int n4 = N / 4;
    const size_t bytes = static_cast<size_t>(n) * sizeof(value_t);

    // ============================================================
    // HOST MEMORY
    // ============================================================
    value_t* h_a = static_cast<value_t*>(std::malloc(bytes));
    value_t* h_b = static_cast<value_t*>(std::malloc(bytes));
    value_t* h_c_baseline = static_cast<value_t*>(std::malloc(bytes));
    value_t* h_c_vectorized = static_cast<value_t*>(std::malloc(bytes));
    value_t* h_c_shared = static_cast<value_t*>(std::malloc(bytes));
    value_t* h_c_dogma = static_cast<value_t*>(std::malloc(bytes));

    if (!h_a || !h_b || !h_c_baseline || !h_c_vectorized || !h_c_shared || !h_c_dogma) {
        std::cerr << "Host memory allocation failed." << std::endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // ============================================================
    // DEVICE MEMORY
    // ============================================================
    value_t *d_a, *d_b, *d_c_baseline, *d_c_vectorized, *d_c_shared, *d_c_dogma;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_baseline, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_vectorized, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_shared, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_dogma, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ============================================================
    // DEVICE INFO
    // ============================================================
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));

    dim3 default_block(BLOCK_SIZE);
    dim3 default_grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 default_grid_vec((n4 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // For dogma ultra: choose a grid size tied to SM count
    dim3 dogma_grid(num_sms * 32);

    // ============================================================
    // CUDA EVENTS
    // ============================================================
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ============================================================
    // WARMUP
    // ============================================================
    baseline_kernel<<<default_grid, default_block>>>(d_c_baseline, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vectorized_kernel<<<default_grid_vec, default_block>>>(
        reinterpret_cast<float4*>(d_c_vectorized),
        reinterpret_cast<const float4*>(d_a),
        reinterpret_cast<const float4*>(d_b),
        n4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    shared_memory_kernel<<<default_grid, default_block>>>(d_c_shared, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dogma_ultra_kernel<<<dogma_grid, default_block>>>(
        reinterpret_cast<float4*>(d_c_dogma),
        reinterpret_cast<const float4*>(d_a),
        reinterpret_cast<const float4*>(d_b),
        n4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ============================================================
    // TIMING
    // ============================================================
    const float baseline_ms = run_baseline(
        d_c_baseline, d_a, d_b, n,
        default_grid, default_block, start, stop);

    const float vectorized_ms = run_vectorized(
        d_c_vectorized, d_a, d_b, n4,
        default_grid_vec, default_block, start, stop);

    const float shared_ms = run_shared(
        d_c_shared, d_a, d_b, n,
        default_grid, default_block, start, stop);

    const float dogma_ms = run_dogma_ultra(
        d_c_dogma, d_a, d_b, n4,
        dogma_grid, default_block, start, stop);

    CUDA_CHECK(cudaDeviceSynchronize());

    // ============================================================
    // COPY RESULTS BACK
    // ============================================================
    CUDA_CHECK(cudaMemcpy(h_c_baseline, d_c_baseline, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_vectorized, d_c_vectorized, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_shared, d_c_shared, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_dogma, d_c_dogma, bytes, cudaMemcpyDeviceToHost));

    // ============================================================
    // VALIDATION
    // ============================================================
    const bool valid_vectorized = validate_arrays(h_c_baseline, h_c_vectorized, n, "Vectorized");
    const bool valid_shared = validate_arrays(h_c_baseline, h_c_shared, n, "Shared");
    const bool valid_dogma = validate_arrays(h_c_baseline, h_c_dogma, n, "Dogma Ultra");

    const bool all_valid = valid_vectorized && valid_shared && valid_dogma;

    // ============================================================
    // METRICS
    // ============================================================
    const float speedup_vectorized = (vectorized_ms > 0.0f) ? (baseline_ms / vectorized_ms) : 0.0f;
    const float speedup_shared = (shared_ms > 0.0f) ? (baseline_ms / shared_ms) : 0.0f;
    const float speedup_dogma = (dogma_ms > 0.0f) ? (baseline_ms / dogma_ms) : 0.0f;

    const float improvement_vectorized =
        (baseline_ms > 0.0f) ? ((baseline_ms - vectorized_ms) / baseline_ms) * 100.0f : 0.0f;
    const float improvement_shared =
        (baseline_ms > 0.0f) ? ((baseline_ms - shared_ms) / baseline_ms) * 100.0f : 0.0f;
    const float improvement_dogma =
        (baseline_ms > 0.0f) ? ((baseline_ms - dogma_ms) / baseline_ms) * 100.0f : 0.0f;

    const float bandwidth_vectorized =
        (vectorized_ms > 0.0f) ? (bytes / (vectorized_ms / 1000.0f)) / 1e9f : 0.0f;
    const float bandwidth_shared =
        (shared_ms > 0.0f) ? (bytes / (shared_ms / 1000.0f)) / 1e9f : 0.0f;
    const float bandwidth_dogma =
        (dogma_ms > 0.0f) ? (bytes / (dogma_ms / 1000.0f)) / 1e9f : 0.0f;

    // ============================================================
    // OUTPUT
    // ============================================================
    std::cout << std::fixed << std::setprecision(5);

    std::cout << "=== GPU Kernel Comparison ===\n\n";

    std::cout << "Validation: " << (all_valid ? "PASSED" : "FAILED") << "\n";
    std::cout << "Iterations: " << ITERATIONS << "\n";
    std::cout << "Elements processed: " << n << "\n";
    std::cout << "Bytes processed: " << bytes << "\n";
    std::cout << "SM count: " << num_sms << "\n\n";

    std::cout << "Baseline Time: " << baseline_ms << " ms\n";
    std::cout << "Vectorized Kernel Time: " << vectorized_ms << " ms\n";
    std::cout << "Shared Memory Kernel Time: " << shared_ms << " ms\n";
    std::cout << "Dogma Ultra Kernel Time: " << dogma_ms << " ms\n\n";

    std::cout << "Vectorized Speedup: " << speedup_vectorized << "x\n";
    std::cout << "Shared Memory Speedup: " << speedup_shared << "x\n";
    std::cout << "Dogma Ultra Speedup: " << speedup_dogma << "x\n\n";

    std::cout << "Vectorized Improvement: " << improvement_vectorized << "%\n";
    std::cout << "Shared Memory Improvement: " << improvement_shared << "%\n";
    std::cout << "Dogma Ultra Improvement: " << improvement_dogma << "%\n\n";

    std::cout << "Approx. Vectorized Bandwidth: " << bandwidth_vectorized << " GB/s\n";
    std::cout << "Approx. Shared Memory Bandwidth: " << bandwidth_shared << " GB/s\n";
    std::cout << "Approx. Dogma Ultra Bandwidth: " << bandwidth_dogma << " GB/s\n";

    // ============================================================
    // BLOCK SIZE TUNING FOR DOGMA ULTRA
    // ============================================================
    std::cout << "\n=== Dogma Ultra Block Size Tuning ===\n";

    const std::vector<int> block_sizes = {64, 128, 256, 512};

    for (int bs : block_sizes) {
        dim3 tuning_block(bs);
        dim3 tuning_grid(num_sms * 32);

        float time_ms = 0.0f;

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERATIONS; ++i) {
            dogma_ultra_kernel<<<tuning_grid, tuning_block>>>(
                reinterpret_cast<float4*>(d_c_dogma),
                reinterpret_cast<const float4*>(d_a),
                reinterpret_cast<const float4*>(d_b),
                n4);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

        time_ms /= static_cast<float>(ITERATIONS);

        std::cout << "Block Size " << bs << " -> " << time_ms << " ms\n";
    }

    // ============================================================
    // CLEANUP
    // ============================================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c_baseline));
    CUDA_CHECK(cudaFree(d_c_vectorized));
    CUDA_CHECK(cudaFree(d_c_shared));
    CUDA_CHECK(cudaFree(d_c_dogma));

    std::free(h_a);
    std::free(h_b);
    std::free(h_c_baseline);
    std::free(h_c_vectorized);
    std::free(h_c_shared);
    std::free(h_c_dogma);

    return all_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
