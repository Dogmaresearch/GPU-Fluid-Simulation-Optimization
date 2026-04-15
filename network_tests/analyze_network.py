def main():
    file = Path("network_tests/real_network_output.txt")

    if not file.exists():
        print("No network output found, skipping test")
        return

    content = file.read_text()

    times = extract_ping_times(content)

    if not times:
        print("No ping data found, skipping analysis")
        return

    avg = sum(times) / len(times)
    print(f"Average latency: {avg:.2f} ms")

    if avg > 100:
        print("⚠️ High latency detected")
    else:
        print("✅ Network latency is good")
