#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
ARCH="${ARCH-}"
EXAMPLE="${EXAMPLE-}"

# If the ARCH environment variable is not set then
if [ -z "$ARCH" ]; then
    echo "invalid ARCH value"
    exit 1
fi

# If the EXAMPLE environment variable is not set then
if [ -z "$EXAMPLE" ]; then
    echo "invalid EXAMPLE value"
    exit 1
fi

# If the DEFINES environment variable is set then
if declare -p DEFINES &>/dev/null; then
    DEFINES_FLAGS=$(printf -- '-D%s ' "${DEFINES[@]}")
else
    DEFINES_FLAGS=""
fi


# Ensure that the build directory exists
mkdir -p "$(pwd)/build/examples/$(dirname "$EXAMPLE")"

# Compile
if [ "$ARCH" = "arm" ]; then
    # Compile for ARM
    /usr/bin/gcc \
        -Wall \
        -fdiagnostics-color=always \
        -g \
        "$DEFINES_FLAGS" \
        -Iinclude/ \
        -Ilib/CMSIS-DSP/Include \
        lib/CMSIS-DSP/Source/BasicMathFunctions/arm_dot_prod_f32.c \
        src/arch/arm/neon/*.c \
        src/arch/arm/cmsis/*.c \
        src/*.c \
        examples/"$EXAMPLE"/main.c \
        -o "$(pwd)/build/examples/$EXAMPLE"
    exit 0
elif [ "$ARCH" = "generic" ]; then
    # Compile for generic architecture
    /usr/bin/gcc \
        -Wall \
        -fdiagnostics-color=always \
        -g \
        "$DEFINES_FLAGS" \
        -Iinclude/ \
        src/*.c \
        examples/"$EXAMPLE"/main.c \
        -o "$(pwd)/build/examples/$EXAMPLE"
fi
