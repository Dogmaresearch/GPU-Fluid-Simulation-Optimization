## Networking Fundamentals (Practical Notes)

### TCP vs UDP
TCP (Transmission Control Protocol) is connection-oriented and guarantees reliable, ordered data delivery. It includes congestion control and retransmission mechanisms, making it suitable for applications where data integrity is critical.

UDP (User Datagram Protocol) is connectionless and does not guarantee delivery or ordering. It has lower overhead and is typically used in latency-sensitive applications such as real-time streaming or high-performance data pipelines.

---

### Latency vs Throughput
Latency refers to the time it takes for data to travel from source to destination (e.g., round-trip time). It is critical in real-time systems where responsiveness matters.

Throughput represents the amount of data transferred per unit of time (e.g., Mbps or Gbps). High-throughput systems focus on maximizing bandwidth utilization.

In high-performance systems, there is often a trade-off between minimizing latency and maximizing throughput.

---

### iperf (Throughput Measurement)
`iperf3` is used to measure network throughput between two endpoints.

In this project, it is used to:
- Evaluate data transfer performance
- Measure achievable bandwidth under controlled conditions
- Validate network performance consistency

This is relevant for high-throughput environments such as datacenter networking and distributed systems.

---

### ss (Socket Inspection)
`ss` is a Linux tool used to inspect socket states and network connections.

In this project, it is used to:
- Verify active TCP/UDP connections
- Inspect listening ports
- Analyze socket states (e.g., LISTEN, ESTABLISHED)

This helps validate that network processes behave correctly and that connections are properly established during tests.

---

### Relevance to High-Performance Systems
Efficient networking is tightly coupled with system performance.

Key considerations include:
- Minimizing latency for real-time responsiveness
- Maximizing throughput for data-intensive workloads
- Ensuring stable and predictable network behavior

These principles are directly applicable to datacenter environments, AI workloads, and high-performance computing systems.
