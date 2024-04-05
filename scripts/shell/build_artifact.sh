#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Init vars
ARCH="${ARCH-}"
TECH="${TECH-}"
ARTIFACT="${ARTIFACT-}"
CC="${CC:-clang}"
CFLAGS+=(-Wall -fdiagnostics-color=always)

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

# set -x
$CC "${CFLAGS[@]}" \
    "$DEFINES_FLAGS" \
    -Iinclude/ \
    "${CC_PARTS[@]:-}" \
    src/*.c \
    "$(pwd)/$ARTIFACT"/main.c \
    -o "$(pwd)/build/$ARTIFACT" "${LDFLAGS[@]:-}"
# set +x
