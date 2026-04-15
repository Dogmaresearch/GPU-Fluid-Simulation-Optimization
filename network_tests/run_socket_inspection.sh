#!/bin/bash
set -e

echo "Inspecting socket state..."

if command -v ss >/dev/null 2>&1; then
    ss -tuln > network_tests/socket_output.txt
else
    netstat -tuln > network_tests/socket_output.txt
fi

echo "Socket inspection completed"
