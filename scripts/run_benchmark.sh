#!/bin/bash

set -e

echo "Compiling CUDA benchmark..."
nvcc ../CUDA-Benchmark/benchmark.cu -o ../benchmark

echo "Running benchmark..."
../benchmark | tee ../benchmark_output.txt

echo "Validating benchmark output..."
python3 scripts/check_benchmark_output.py ../benchmark_output.txt

echo "Running Python automation..."
python3 scripts/run_benchmark.py
