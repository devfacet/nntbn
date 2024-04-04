#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
CC=clang
CFLAGS=(-Wall -fdiagnostics-color=always)
LDFLAGS=(-lm)
ARCH="${ARCH-}"
TARGET="${TARGET-}"

# If the ARCH environment variable is not set then
if [ -z "$ARCH" ]; then
    echo "invalid ARCH value"
    exit 1
fi

# If the TARGET environment variable is not set then
if [ -z "$TARGET" ]; then
    echo "invalid TARGET value"
    exit 1
fi

# If the DEFINES environment variable is set then
if declare -p DEFINES &>/dev/null; then
    DEFINES_FLAGS=$(printf -- '-D%s ' "${DEFINES[@]}")
else
    DEFINES_FLAGS=""
fi

# Ensure that the build directory exists
mkdir -p "$(pwd)/build/$(dirname "$TARGET")"

# Compile
if [ "$ARCH" = "arm" ]; then
    # Arm
    $CC "${CFLAGS[@]}" \
        "$DEFINES_FLAGS" \
        -Iinclude/ \
        -Ilib/CMSIS-DSP/Include \
        lib/CMSIS-DSP/Source/BasicMathFunctions/arm_dot_prod_f32.c \
        src/arch/arm/neon/*.c \
        src/arch/arm/cmsis/*.c \
        src/*.c \
        "$(pwd)/$TARGET"/main.c \
        -o "$(pwd)/build/$TARGET" "${LDFLAGS[@]}"

elif [ "$ARCH" = "generic" ]; then
    # Generic
    $CC "${CFLAGS[@]}" \
        "$DEFINES_FLAGS" \
        -Iinclude/ \
        src/*.c \
        "$(pwd)/$TARGET"/main.c \
        -o "$(pwd)/build/$TARGET" "${LDFLAGS[@]}"

fi
