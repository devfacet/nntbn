#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
ARGS="${ARGS-}"
EXAMPLE="${EXAMPLE-}"

# If the EXAMPLE environment variable is not set then
if [ -z "$EXAMPLE" ]; then
    echo "invalid EXAMPLE value"
    exit 1
fi

# Determine the executable path
executable_path="$(pwd)/build/examples/$EXAMPLE"
if [ ! -x "$executable_path" ]; then
    echo "$EXAMPLE not found"
    exit 1
fi

# Run
$executable_path "$ARGS"
