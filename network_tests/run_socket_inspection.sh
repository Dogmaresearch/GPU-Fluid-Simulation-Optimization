#!/bin/bash

set +e

echo "Inspecting socket state..."
ss -tuln > network_tests/socket_output.txt

echo "Socket inspection completed"
