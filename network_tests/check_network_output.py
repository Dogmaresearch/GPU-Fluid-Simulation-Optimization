import sys
from pathlib import Path

REQUIRED_STRINGS = [
    "ping",
    "time=",
    "Network test completed"
]

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_network_output.py <file>")
        sys.exit(1)

    file = Path(sys.argv[1])

    if not file.exists():
        print("File not found")
        sys.exit(1)

    content = file.read_text()

    missing = [s for s in REQUIRED_STRINGS if s not in content]

    if missing:
        print("Network test validation failed")
        for m in missing:
            print(f"Missing: {m}")
        sys.exit(1)

    print("Network test validation passed")

if __name__ == "__main__":
    main()
