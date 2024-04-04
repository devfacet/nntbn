#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
ARGS="${ARGS-}"
TARGET="${TARGET-}"

# If the TARGET environment variable is not set then
if [ -z "$TARGET" ]; then
    echo "invalid TARGET value"
    exit 1
fi

# Determine the executable path
exec_path="$(pwd)/build/$TARGET"
if [ ! -x "$exec_path" ]; then
    echo "$TARGET not found"
    exit 1
fi

# Run
$exec_path "$ARGS"
