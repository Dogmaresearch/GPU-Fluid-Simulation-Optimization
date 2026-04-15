from pathlib import Path
import sys

REQUIRED_STRINGS = [
    "State",
]

def main():
    file = Path("network_tests/socket_output.txt")

    if not file.exists():
        print("Socket output file not found")
        sys.exit(1)

    content = file.read_text(encoding="utf-8", errors="ignore")

    missing = [s for s in REQUIRED_STRINGS if s not in content]

    if missing:
        print("Socket output validation failed")
        for item in missing:
            print(f"Missing: {item}")
        sys.exit(1)

    print("Socket output validation passed")

if __name__ == "__main__":
    main()
