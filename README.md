# ⚠️ License Notice

This project is provided for research and demonstration purposes only.  
Commercial use is not allowed without explicit permission.

---

# Real-Time GPU Fluid Simulation & CUDA Kernel Optimization

## 🔥 Overview

This project explores GPU-based computation through:

- Real-time fluid simulation in Unity (compute shaders)
- Analysis of data flow inside the GPU
- CUDA kernel optimization focused on memory access and parallelism

The goal wasn’t just to make it work, but to understand:

👉 how data moves across the GPU  
👉 where performance is lost  
👉 how to optimize the computation flow  

---

## 🧪 Fluid Simulation

A GPU-based fluid simulation implemented in Unity using compute shaders.

This provides the visual and practical context for understanding GPU computation patterns.

---

## ⚙️ CUDA Optimization

Comparison between:

- Baseline kernel  
- Optimized kernel (vectorized float4)

The optimization focuses on improving memory access patterns and parallel execution efficiency.

---

## 🧠 Key Concepts

- GPU parallelism  
- Memory coalescing  
- Vectorized memory access  
- Performance benchmarking  

---

## 🎥 Demo

(Video coming soon)

---

## 🚀 How to Run (CUDA Benchmark)

### Requirements
- NVIDIA GPU
- CUDA Toolkit installed (recommended 11.x or newer)

### Compile
```bash
nvcc CUDA-Benchmark/benchmark.cu -o benchmark
```

### Run
```bash
./benchmark
```

### Expected Output
```text
Baseline Time: 12.4 ms
Dogma Optimized Time: 7.1 ms
Speedup: 1.74x
Performance Improvement: 42.7%
```
