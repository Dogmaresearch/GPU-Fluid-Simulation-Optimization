// ===================================
// BASELINE TEST
// ===================================
float baseline_ms = 0.0f;
float optimized_ms = 0.0f;

cudaEventRecord(start);

baseline_kernel<<<gridSize, blockSize>>>(d_C, d_A, d_B, N);

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&baseline_ms, start, stop);

std::cout << "Baseline Time: " << baseline_ms << " ms\n";


// ===================================
// OPTIMIZED TEST
// ===================================
int optimizedGridSize = ((N / 4) + blockSize - 1) / blockSize;

cudaEventRecord(start);

dogma_optimized_kernel<<<optimizedGridSize, blockSize>>>(
    (float4*)d_C,
    (float4*)d_A,
    (float4*)d_B,
    N / 4
);

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&optimized_ms, start, stop);

std::cout << "Dogma Optimized Time: " << optimized_ms << " ms\n";


// ===================================
// RESULTS
// ===================================
cudaDeviceSynchronize();

float speedup = baseline_ms / optimized_ms;
float improvement = ((baseline_ms - optimized_ms) / baseline_ms) * 100.0f;

std::cout << "Speedup: " << speedup << "x\n";
std::cout << "Performance Improvement: " << improvement << "%\n";


// ===================================
// CLEANUP
// ===================================
cudaEventDestroy(start);
cudaEventDestroy(stop);

cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

free(h_A);
free(h_B);

return 0;
