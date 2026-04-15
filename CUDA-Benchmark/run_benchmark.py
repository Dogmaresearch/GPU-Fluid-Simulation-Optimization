import subprocess
import re
import statistics
from pathlib import Path

# Assumes this script is run from the repository root.
EXECUTABLE = Path("./benchmark")
RUNS = 5


def run_once() -> str:
    """Run the benchmark executable once and return stdout."""
    result = subprocess.run(
        [str(EXECUTABLE)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Benchmark execution failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout


def extract_time(output: str) -> float | None:
    """Extract Optimized Kernel Time from benchmark output."""
    match = re.search(r"Optimized Kernel Time:\s+([\d.]+)\s+ms", output)
    if match:
        return float(match.group(1))
    return None


def main() -> None:
    times: list[float] = []

    print("Running benchmark automation...\n")

    if not EXECUTABLE.exists():
        print(f"Error: executable not found at {EXECUTABLE}")
        print("Compile first with:")
        print("nvcc CUDA-Benchmark/benchmark.cu -o benchmark")
        return

    for i in range(RUNS):
        print(f"Run {i + 1}/{RUNS}")
        output = run_once()
        print(output)

        t = extract_time(output)
        if t is not None:
            times.append(t)

    if times:
        avg = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)

        print("Summary:")
        print(f"Average Optimized Time: {avg:.3f} ms")
        print(f"Min Optimized Time: {min_time:.3f} ms")
        print(f"Max Optimized Time: {max_time:.3f} ms")
    else:
        print("No valid timing data found in benchmark output.")


if __name__ == "__main__":
    main()