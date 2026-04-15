import sys
from pathlib import Path

REQUIRED_STRINGS = [
    "Validation: PASSED",
    "Baseline Time:",
    "Optimized Kernel Time:",
    "Speedup:",
    "Performance Improvement:",
]

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/check_benchmark_output.py <output_file>")
        sys.exit(1)

    output_file = Path(sys.argv[1])

    if not output_file.exists():
        print(f"Error: file not found: {output_file}")
        sys.exit(1)

    content = output_file.read_text(encoding="utf-8", errors="ignore")

    missing = [item for item in REQUIRED_STRINGS if item not in content]

    if missing:
        print("Benchmark output validation failed.")
        print("Missing expected lines:")
        for item in missing:
            print(f"- {item}")
        sys.exit(1)

    print("Benchmark output validation passed.")

if __name__ == "__main__":
    main()
