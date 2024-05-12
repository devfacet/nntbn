#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
ARGS="${ARGS-}"
ARTIFACT="${ARTIFACT-}"

# If the ARTIFACT environment variable is not set then
if [ -z "$ARTIFACT" ] ; then
    echo "invalid ARTIFACT value"
    exit 1
fi

# Determine the executable path
exec_path="$(pwd)/build/$ARTIFACT"
if [ ! -x "$exec_path" ]; then
    echo "$ARTIFACT not found"
    exit 1
fi

# Run
$exec_path "$ARGS"
