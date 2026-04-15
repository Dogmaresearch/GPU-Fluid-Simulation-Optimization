import re
from pathlib import Path

def extract_ping_times(content):
    times = re.findall(r"time=(\d+\.?\d*)", content)
    return [float(t) for t in times]

def main():
    file = Path("network_tests/mock_network_output.txt")

    if not file.exists():
        print("No network output found")
        return

    content = file.read_text()

    times = extract_ping_times(content)

    if not times:
        print("No ping data found")
        return

    avg = sum(times) / len(times)
    print(f"Average latency: {avg:.2f} ms")

    if avg > 100:
        print("⚠️ High latency detected")
    else:
        print("✅ Network latency is good")

if __name__ == "__main__":
    main()
