import subprocess
import time
import re

EXECUTABLE = "./benchmark"
RUNS = 5

def run_once():
    result = subprocess.run(EXECUTABLE, capture_output=True, text=True)
    return result.stdout

def extract_time(output):
    match = re.search(r"Optimized Kernel Time:\s+([\d.]+)", output)
    if match:
        return float(match.group(1))
    return None

def main():
    times = []

    print("Running benchmark...")

    for i in range(RUNS):
        print(f"Run {i+1}/{RUNS}")
        output = run_once()
        print(output)

        t = extract_time(output)
        if t:
            times.append(t)

    if times:
        avg = sum(times) / len(times)
        print(f"\nAverage Optimized Time: {avg:.3f} ms")

if __name__ == "__main__":
    main()