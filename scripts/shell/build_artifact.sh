#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
ARCH="${ARCH-}"
TECH="${TECH-}"
ARTIFACT="${ARTIFACT-}"
CC="${CC:-clang}"
CFLAGS_ARRAY=(-Wall -fdiagnostics-color=always)

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

# Ensure that the build directory exists
mkdir -p "$(pwd)/build/$(dirname "$ARTIFACT")"

# Compile
CC_PARTS=()
# Convert comma-separated TECH to an array
IFS=',' read -ra TECHS <<< "$TECH"
for tech in "${TECHS[@]:-}"; do
    if [ "$tech" = "neon" ]; then
        # Arm NEON
        CC_PARTS+=(src/arch/arm/neon/*.c)
    fi
    if [ "$tech" = "cmsis-dsp" ]; then
        # Arm CMSIS-DSP
        CC_PARTS+=(-Ilib/CMSIS_6/CMSIS/Core/Include -Ilib/CMSIS-DSP/Include lib/CMSIS-DSP/Source/BasicMathFunctions/arm_dot_prod_f32.c src/arch/arm/cmsis-dsp/*.c)
    fi
done

COMPILE_CMD=("$CC" "${CFLAGS_ARRAY[@]}" \
    -Iinclude/ \
    "${CC_PARTS[@]:-}" \
    src/*.c \
    "$(pwd)/$ARTIFACT"/main.c \
    -o "$(pwd)/build/$ARTIFACT")

if [ ${#LDFLAGS_ARRAY[@]} -ne 0 ]; then
    COMPILE_CMD+=("${LDFLAGS_ARRAY[@]}")
fi
# echo "${COMPILE_CMD[@]}"
"${COMPILE_CMD[@]}"
