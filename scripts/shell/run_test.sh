#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
CC=clang
CFLAGS=(-Wall -fdiagnostics-color=always)
LDFLAGS=(-lm)
ARCH="${ARCH-}"
TEST="${TEST-}"

# If the ARCH environment variable is not set then
if [ -z "$ARCH" ]; then
    echo "invalid ARCH value"
    exit 1
fi

# If the TEST environment variable is not set then
if [ -z "$TEST" ]; then
    echo "invalid TEST value"
    exit 1
fi

# If the DEFINES environment variable is set then
if declare -p DEFINES &>/dev/null; then
    DEFINES_FLAGS=$(printf -- '-D%s ' "${DEFINES[@]}")
else
    DEFINES_FLAGS=""
fi

# Ensure that the build directory exists
mkdir -p "$(pwd)/build/tests/$(dirname "$TEST")"

# Compile
if [ "$ARCH" = "arm" ]; then
    # Compile for ARM
    $CC "${CFLAGS[@]}" \
        "$DEFINES_FLAGS" \
        -Iinclude/ \
        -Ilib/CMSIS-DSP/Include \
        lib/CMSIS-DSP/Source/BasicMathFunctions/arm_dot_prod_f32.c \
        src/arch/arm/neon/*.c \
        src/arch/arm/cmsis/*.c \
        src/*.c \
        tests/"$TEST"/main.c \
        -o "$(pwd)/build/tests/$TEST" "${LDFLAGS[@]}"
elif [ "$ARCH" = "generic" ]; then
    # Compile for generic architecture
    $CC "${CFLAGS[@]}" \
        "$DEFINES_FLAGS" \
        -Iinclude/ \
        src/*.c \
        tests/"$TEST"/main.c \
        -o "$(pwd)/build/tests/$TEST" "${LDFLAGS[@]}"
fi

# Run
"$(pwd)/build/tests/$TEST"
