#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
ARTIFACT="${ARTIFACT-}"
CC="${CC:-clang}"
CFLAGS_ARRAY=(-Wall -fdiagnostics-color=always -std=c99)

# Check and add CFLAGS to CFLAGS_ARRAY
if [ -n "${CFLAGS-}" ]; then
    IFS=' ' read -ra CFLAGS_PARTS <<< "$CFLAGS"
    CFLAGS_ARRAY+=("${CFLAGS_PARTS[@]}")
fi

# Check and add LDFLAGS to LDFLAGS_ARRAY
if [ -n "${LDFLAGS-}" ]; then
    IFS=' ' read -ra LDFLAGS_ARRAY <<< "$LDFLAGS"
else
    LDFLAGS_ARRAY=()
fi

# If the ARTIFACT environment variable is not set then
if [ -z "$ARTIFACT" ] ; then
    echo "invalid ARTIFACT value"
    exit 1
fi

# Ensure that the build directory exists
mkdir -p "$(pwd)/build/$(dirname "$ARTIFACT")"

# Read source files from includes.txt in the ARTIFACT directory
SRC_FILES=()
while IFS= read -r line; do
    SRC_FILES+=("$line")
done < "$(pwd)/$ARTIFACT/include.txt"

# For debugging
# CFLAGS_ARRAY+=(-E)
# SRC_FILES=()

COMPILE_CMD=("$CC" "${CFLAGS_ARRAY[@]}" \
    -Iinclude/ \
    "${SRC_FILES[@]:-}" \
    "$(pwd)/$ARTIFACT"/main.c \
    -o "$(pwd)/build/$ARTIFACT")

if [ ${#LDFLAGS_ARRAY[@]} -ne 0 ]; then
    COMPILE_CMD+=("${LDFLAGS_ARRAY[@]}")
fi
# echo "${COMPILE_CMD[@]}"
"${COMPILE_CMD[@]}"
