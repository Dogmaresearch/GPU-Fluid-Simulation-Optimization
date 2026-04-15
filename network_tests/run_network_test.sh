#!/bin/bash

set -e

echo "Running ping test..."
ping -c 4 google.com > network_output.txt

echo "Running iperf test..."
iperf -c speedtest.serverius.net -p 5002 -t 5 >> network_output.txt || echo "iperf failed" >> network_output.txt

echo "Network test completed."
