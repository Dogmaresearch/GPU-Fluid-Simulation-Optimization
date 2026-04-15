# 🚀 GPU Compute Optimization & Memory Access Analysis for High-Performance Systems

This project investigates how low-level memory access patterns affect GPU performance, with a focus on building intuition for optimizing real-world high-performance systems.

---

## 🔥 Overview

This project explores GPU-based computation through:

- Real-time fluid simulation in Unity (compute shaders)
- Analysis of data flow inside the GPU
- CUDA kernel optimization focused on memory access and parallel execution

The goal was not just to make it work, but to understand:

👉 how data moves across the GPU  
👉 where performance bottlenecks occur  
👉 how to optimize computation flow at a low level  

---

## 🧠 System-Level Performance Perspective

This project goes beyond basic CUDA implementation by analyzing performance from a system-level perspective.

Key considerations:

- Global memory access efficiency  
- Memory coalescing behavior  
- Data movement between host and device  
- Kernel launch configuration and scalability  

These aspects are critical in high-performance systems used in AI, HPC, and large-scale data processing.

---

## 🌍 Real-World Relevance

The optimization techniques explored in this project are directly applicable to:

- GPU-accelerated AI workloads  
- High-performance computing (HPC)  
- Large-scale data processing pipelines  
- Networking systems requiring high throughput and low latency  

Understanding memory behavior at the GPU level is essential for minimizing bottlenecks in distributed and high-performance environments.

---

## 💧 Fluid Simulation (Unity)

A GPU-based fluid simulation implemented in Unity using compute shaders.

This provides a visual and practical context for understanding GPU computation patterns and performance behavior.

---

## ⚙️ CUDA Optimization

Comparison between:

- Baseline kernel  
- Optimized kernel (vectorized using `float4`)  

The optimization focuses on:

- Improving memory access patterns  
- Reducing global memory transactions  
- Increasing parallel efficiency  

---

## 🧩 Key Concepts

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

---

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
