#!/usr/bin/env bash
set -euo pipefail

# Ensure we are inside an activated conda environment
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "Error: No conda environment is currently active." >&2
  exit 1
fi

CRYPT_SRC="/usr/include/crypt.h"
INCLUDE_DIR="$CONDA_PREFIX/include"

# Copy crypt.h
mkdir -p "$INCLUDE_DIR"
cp "$CRYPT_SRC" "$INCLUDE_DIR/crypt.h"

# Set up activation hook
HOOK_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$HOOK_DIR"
echo 'export CPATH=$CONDA_PREFIX/include:$CPATH' > "$HOOK_DIR/env_vars.sh"

echo "crypt.h installed to $INCLUDE_DIR"
echo "Activation hook created at $HOOK_DIR/env_vars.sh"
echo "It will set CPATH automatically when this environment is activated."