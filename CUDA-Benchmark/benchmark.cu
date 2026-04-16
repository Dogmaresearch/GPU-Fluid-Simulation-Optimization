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
// DOGMA FINAL KERNEL
// ============================================================
__global__ void dogma_final_kernel(
    float4* __restrict__ c,
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    int n4)
{
    int stride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < (n4 / 2);
         idx += stride)
    {
        int base = idx * 2;

        float4 a0 = a[base];
        float4 a1 = a[base + 1];
        float4 b0 = b[base];
        float4 b1 = b[base + 1];

        float4 r0;
        float4 r1;

        r0.x = 0.0f; r0.y = 0.0f; r0.z = 0.0f; r0.w = 0.0f;
        r1.x = 0.0f; r1.y = 0.0f; r1.z = 0.0f; r1.w = 0.0f;

        #pragma unroll 20
        for (int k = 0; k < INNER_REPEAT; ++k) {
            r0.x += a0.x * b0.x;
            r0.y += a0.y * b0.y;
            r0.z += a0.z * b0.z;
            r0.w += a0.w * b0.w;

            r1.x += a1.x * b1.x;
            r1.y += a1.y * b1.y;
            r1.z += a1.z * b1.z;
            r1.w += a1.w * b1.w;
        }

        c[base]     = r0;
        c[base + 1] = r1;
    }
}

// ============================================================
// DOGMA GHOST KERNEL
// ============================================================
__global__ void dogma_ghost_kernel(
    float4* __restrict__ c,
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    int n4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; idx < n4; idx += stride) {
        float4 av = a[idx];
        float4 bv = b[idx];
        float4 rv;

        rv.x = 0.0f;
        rv.y = 0.0f;
        rv.z = 0.0f;
        rv.w = 0.0f;

        #pragma unroll 8
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
// DOGMA APEX KERNEL
// Lean vectorized kernel with low register pressure,
// grid-stride loop, and 128-thread friendly launch.
// ============================================================
__global__ void dogma_apex_kernel(
    float4* __restrict__ c,
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    int n4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n4; i += stride) {
        float4 av = a[i];
        float4 bv = b[i];
        float4 res;

        res.x = 0.0f;
        res.y = 0.0f;
        res.z = 0.0f;
        res.w = 0.0f;

        #pragma unroll 16
        for (int k = 0; k < INNER_REPEAT; ++k) {
            res.x += av.x * bv.x;
            res.y += av.y * bv.y;
            res.z += av.z * bv.z;
            res.w += av.w * bv.w;
        }

        c[i] = res;
    }
}

// ============================================================
// Validation helper
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
// Timing helpers
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

float run_dogma_final(
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
        dogma_final_kernel<<<grid, block>>>(
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

float run_dogma_ghost(
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
        dogma_ghost_kernel<<<grid, block>>>(
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

float run_dogma_apex(
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
        dogma_apex_kernel<<<grid, block>>>(
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
    static_assert((N / 4) % 2 == 0, "N/4 must be even for dogma_final_kernel.");

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
    value_t* h_c_dogma_ultra = static_cast<value_t*>(std::malloc(bytes));
    value_t* h_c_dogma_final = static_cast<value_t*>(std::malloc(bytes));
    value_t* h_c_dogma_ghost = static_cast<value_t*>(std::malloc(bytes));
    value_t* h_c_dogma_apex = static_cast<value_t*>(std::malloc(bytes));

    if (!h_a || !h_b || !h_c_baseline || !h_c_vectorized || !h_c_shared ||
        !h_c_dogma_ultra || !h_c_dogma_final || !h_c_dogma_ghost || !h_c_dogma_apex) {
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
    value_t *d_a, *d_b;
    value_t *d_c_baseline, *d_c_vectorized, *d_c_shared;
    value_t *d_c_dogma_ultra, *d_c_dogma_final, *d_c_dogma_ghost, *d_c_dogma_apex;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_baseline, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_vectorized, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_shared, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_dogma_ultra, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_dogma_final, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_dogma_ghost, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_dogma_apex, bytes));

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

    dim3 dogma_grid(num_sms * 32);
    dim3 dogma_block(BLOCK_SIZE);

    dim3 ghost_grid(num_sms * 16);
    dim3 ghost_block(128);

    dim3 apex_grid(num_sms * 32);
    dim3 apex_block(128);

    // ============================================================
    // EVENTS
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

    dogma_ultra_kernel<<<dogma_grid, dogma_block>>>(
        reinterpret_cast<float4*>(d_c_dogma_ultra),
        reinterpret_cast<const float4*>(d_a),
        reinterpret_cast<const float4*>(d_b),
        n4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dogma_final_kernel<<<dogma_grid, dogma_block>>>(
        reinterpret_cast<float4*>(d_c_dogma_final),
        reinterpret_cast<const float4*>(d_a),
        reinterpret_cast<const float4*>(d_b),
        n4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dogma_ghost_kernel<<<ghost_grid, ghost_block>>>(
        reinterpret_cast<float4*>(d_c_dogma_ghost),
        reinterpret_cast<const float4*>(d_a),
        reinterpret_cast<const float4*>(d_b),
        n4);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dogma_apex_kernel<<<apex_grid, apex_block>>>(
        reinterpret_cast<float4*>(d_c_dogma_apex),
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

    const float dogma_ultra_ms = run_dogma_ultra(
        d_c_dogma_ultra, d_a, d_b, n4,
        dogma_grid, dogma_block, start, stop);

    const float dogma_final_ms = run_dogma_final(
        d_c_dogma_final, d_a, d_b, n4,
        dogma_grid, dogma_block, start, stop);

    const float dogma_ghost_ms = run_dogma_ghost(
        d_c_dogma_ghost, d_a, d_b, n4,
        ghost_grid, ghost_block, start, stop);

    const float dogma_apex_ms = run_dogma_apex(
        d_c_dogma_apex, d_a, d_b, n4,
        apex_grid, apex_block, start, stop);

    CUDA_CHECK(cudaDeviceSynchronize());

    // ============================================================
    // COPY RESULTS BACK
    // ============================================================
    CUDA_CHECK(cudaMemcpy(h_c_baseline, d_c_baseline, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_vectorized, d_c_vectorized, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_shared, d_c_shared, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_dogma_ultra, d_c_dogma_ultra, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_dogma_final, d_c_dogma_final, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_dogma_ghost, d_c_dogma_ghost, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c_dogma_apex, d_c_dogma_apex, bytes, cudaMemcpyDeviceToHost));

    // ============================================================
    // VALIDATION
    // ============================================================
    const bool valid_vectorized =
        validate_arrays(h_c_baseline, h_c_vectorized, n, "Vectorized");

    const bool valid_shared =
        validate_arrays(h_c_baseline, h_c_shared, n, "Shared");

    const bool valid_dogma_ultra =
        validate_arrays(h_c_baseline, h_c_dogma_ultra, n, "Dogma Ultra");

    const bool valid_dogma_final =
        validate_arrays(h_c_baseline, h_c_dogma_final, n, "Dogma Final");

    const bool valid_dogma_ghost =
        validate_arrays(h_c_baseline, h_c_dogma_ghost, n, "Dogma Ghost");

    const bool valid_dogma_apex =
        validate_arrays(h_c_baseline, h_c_dogma_apex, n, "Dogma Apex");

    const bool all_valid =
        valid_vectorized && valid_shared &&
        valid_dogma_ultra && valid_dogma_final &&
        valid_dogma_ghost && valid_dogma_apex;

    // ============================================================
    // METRICS
    // ============================================================
    const float speedup_vectorized =
        (vectorized_ms > 0.0f) ? (baseline_ms / vectorized_ms) : 0.0f;

    const float speedup_shared =
        (shared_ms > 0.0f) ? (baseline_ms / shared_ms) : 0.0f;

    const float speedup_dogma_ultra =
        (dogma_ultra_ms > 0.0f) ? (baseline_ms / dogma_ultra_ms) : 0.0f;

    const float speedup_dogma_final =
        (dogma_final_ms > 0.0f) ? (baseline_ms / dogma_final_ms) : 0.0f;

    const float speedup_dogma_ghost =
        (dogma_ghost_ms > 0.0f) ? (baseline_ms / dogma_ghost_ms) : 0.0f;

    const float speedup_dogma_apex =
        (dogma_apex_ms > 0.0f) ? (baseline_ms / dogma_apex_ms) : 0.0f;

    const float improvement_vectorized =
        (baseline_ms > 0.0f) ? ((baseline_ms - vectorized_ms) / baseline_ms) * 100.0f : 0.0f;

    const float improvement_shared =
        (baseline_ms > 0.0f) ? ((baseline_ms - shared_ms) / baseline_ms) * 100.0f : 0.0f;

    const float improvement_dogma_ultra =
        (baseline_ms > 0.0f) ? ((baseline_ms - dogma_ultra_ms) / baseline_ms) * 100.0f : 0.0f;

    const float improvement_dogma_final =
        (baseline_ms > 0.0f) ? ((baseline_ms - dogma_final_ms) / baseline_ms) * 100.0f : 0.0f;

    const float improvement_dogma_ghost =
        (baseline_ms > 0.0f) ? ((baseline_ms - dogma_ghost_ms) / baseline_ms) * 100.0f : 0.0f;

    const float improvement_dogma_apex =
        (baseline_ms > 0.0f) ? ((baseline_ms - dogma_apex_ms) / baseline_ms) * 100.0f : 0.0f;

    const float bandwidth_vectorized =
        (vectorized_ms > 0.0f) ? (bytes / (vectorized_ms / 1000.0f)) / 1e9f : 0.0f;

    const float bandwidth_shared =
        (shared_ms > 0.0f) ? (bytes / (shared_ms / 1000.0f)) / 1e9f : 0.0f;

    const float bandwidth_dogma_ultra =
        (dogma_ultra_ms > 0.0f) ? (bytes / (dogma_ultra_ms / 1000.0f)) / 1e9f : 0.0f;

    const float bandwidth_dogma_final =
        (dogma_final_ms > 0.0f) ? (bytes / (dogma_final_ms / 1000.0f)) / 1e9f : 0.0f;

    const float bandwidth_dogma_ghost =
        (dogma_ghost_ms > 0.0f) ? (bytes / (dogma_ghost_ms / 1000.0f)) / 1e9f : 0.0f;

    const float bandwidth_dogma_apex =
        (dogma_apex_ms > 0.0f) ? (bytes / (dogma_apex_ms / 1000.0f)) / 1e9f : 0.0f;

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
    std::cout << "Dogma Ultra Kernel Time: " << dogma_ultra_ms << " ms\n";
    std::cout << "Dogma Final Kernel Time: " << dogma_final_ms << " ms\n";
    std::cout << "Dogma Ghost Kernel Time: " << dogma_ghost_ms << " ms\n";
    std::cout << "Dogma Apex Kernel Time: " << dogma_apex_ms << " ms\n\n";

    std::cout << "Vectorized Speedup: " << speedup_vectorized << "x\n";
    std::cout << "Shared Memory Speedup: " << speedup_shared << "x\n";
    std::cout << "Dogma Ultra Speedup: " << speedup_dogma_ultra << "x\n";
    std::cout << "Dogma Final Speedup: " << speedup_dogma_final << "x\n";
    std::cout << "Dogma Ghost Speedup: " << speedup_dogma_ghost << "x\n";
    std::cout << "Dogma Apex Speedup: " << speedup_dogma_apex << "x\n\n";

    std::cout << "Vectorized Improvement: " << improvement_vectorized << "%\n";
    std::cout << "Shared Memory Improvement: " << improvement_shared << "%\n";
    std::cout << "Dogma Ultra Improvement: " << improvement_dogma_ultra << "%\n";
    std::cout << "Dogma Final Improvement: " << improvement_dogma_final << "%\n";
    std::cout << "Dogma Ghost Improvement: " << improvement_dogma_ghost << "%\n";
    std::cout << "Dogma Apex Improvement: " << improvement_dogma_apex << "%\n\n";

    std::cout << "Approx. Vectorized Bandwidth: " << bandwidth_vectorized << " GB/s\n";
    std::cout << "Approx. Shared Memory Bandwidth: " << bandwidth_shared << " GB/s\n";
    std::cout << "Approx. Dogma Ultra Bandwidth: " << bandwidth_dogma_ultra << " GB/s\n";
    std::cout << "Approx. Dogma Final Bandwidth: " << bandwidth_dogma_final << " GB/s\n";
    std::cout << "Approx. Dogma Ghost Bandwidth: " << bandwidth_dogma_ghost << " GB/s\n";
    std::cout << "Approx. Dogma Apex Bandwidth: " << bandwidth_dogma_apex << " GB/s\n";

    // ============================================================
    // BLOCK SIZE TUNING FOR DOGMA APEX
    // ============================================================
    std::cout << "\n=== Dogma Apex Block Size Tuning ===\n";

    const std::vector<int> apex_block_sizes = {64, 128, 256, 512};

    for (int bs : apex_block_sizes) {
        dim3 tuning_block(bs);
        dim3 tuning_grid(num_sms * 32);

        float time_ms = 0.0f;

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERATIONS; ++i) {
            dogma_apex_kernel<<<tuning_grid, tuning_block>>>(
                reinterpret_cast<float4*>(d_c_dogma_apex),
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
    CUDA_CHECK(cudaFree(d_c_dogma_ultra));
    CUDA_CHECK(cudaFree(d_c_dogma_final));
    CUDA_CHECK(cudaFree(d_c_dogma_ghost));
    CUDA_CHECK(cudaFree(d_c_dogma_apex));

    std::free(h_a);
    std::free(h_b);
    std::free(h_c_baseline);
    std::free(h_c_vectorized);
    std::free(h_c_shared);
    std::free(h_c_dogma_ultra);
    std::free(h_c_dogma_final);
    std::free(h_c_dogma_ghost);
    std::free(h_c_dogma_apex);

    return all_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
