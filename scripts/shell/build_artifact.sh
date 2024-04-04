#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
ARCH="${ARCH-}"
TARGET="${TARGET-}"
ARTIFACT="${ARTIFACT-}"
CC=clang
CFLAGS+=(-Wall -fdiagnostics-color=always)
LDFLAGS+=(-lm)

# If the ARCH environment variable is not set then
if [ -z "$ARCH" ]; then
    echo "invalid ARCH value"
    exit 1
fi

# If the ARTIFACT environment variable is not set then
if [ -z "$ARTIFACT" ] || [ "$ARTIFACT" = "examples/" ] || [ "$ARTIFACT" = "tests/" ]; then
    echo "invalid ARTIFACT value"
    exit 1
fi

# If the DEFINES environment variable is set then
if declare -p DEFINES &>/dev/null; then
    DEFINES_FLAGS=$(printf -- '-D%s ' "${DEFINES[@]}")
else
    DEFINES_FLAGS=""
fi

# Ensure that the build directory exists
mkdir -p "$(pwd)/build/$(dirname "$ARTIFACT")"

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
        "$(pwd)/$ARTIFACT"/main.c \
        -o "$(pwd)/build/$ARTIFACT" "${LDFLAGS[@]}"

elif [ "$ARCH" = "generic" ]; then
    # Generic
    $CC "${CFLAGS[@]}" \
        "$DEFINES_FLAGS" \
        -Iinclude/ \
        src/*.c \
        "$(pwd)/$ARTIFACT"/main.c \
        -o "$(pwd)/build/$ARTIFACT" "${LDFLAGS[@]}"

fi
