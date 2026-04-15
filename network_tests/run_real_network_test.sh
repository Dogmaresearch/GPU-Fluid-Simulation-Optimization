#!/bin/bash

set +e  # <-- IMPORTANTE (non far fallire se ping fallisce)

echo "Running real network test..."

ping -c 4 8.8.8.8 > network_tests/real_network_output.txt

echo "Network test completed"
