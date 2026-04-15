#!/bin/bash

set +e

echo "Starting local iperf3 server..."
iperf3 -s -D

sleep 2

echo "Running local iperf3 client..."
iperf3 -c 127.0.0.1 -t 3 > network_tests/iperf_output.txt

echo "Stopping iperf3 server..."
pkill iperf3 || true

echo "iperf3 local test completed"
