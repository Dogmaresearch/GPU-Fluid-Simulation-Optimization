#!/bin/bash

set -e

echo "Compiling CUDA benchmark..."
nvcc CUDA-Benchmark/benchmark.cu -o benchmark

echo "Running benchmark..."
./benchmark

echo "Running Python automation..."
python3 scripts/run_benchmark.py
