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

## 🚀 How to Run (CUDA Kernel)

### Requirements
- NVIDIA GPU
- CUDA Toolkit installed (recommended 11.x or newer)

### Compile
```bash
nvcc CUDA-Benchmark/kernel.cu -o kernel
```

### Run
```bash
./kernel
```
## 🚀 Results

The optimized kernel shows improved memory access patterns and better parallel execution efficiency compared to the baseline version.

Preliminary observations:

- Reduced memory access overhead
- More efficient thread utilization
- Smoother execution behavior under load

This is an ongoing study, but the current implementation demonstrates how low-level memory optimizations can impact GPU performance.
