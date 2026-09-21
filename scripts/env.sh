#!/usr/bin/env bash
# ReKep Skill dev environment: force ALL state/cache/build/download onto /data7.
# Root disk "/" has ~4.7G free; NEVER let tools write to /tmp, /root, ~/.cache, etc.
# Usage: source scripts/env.sh
set -a
PAOS_REKEP_ROOT="${PAOS_REKEP_ROOT:-/data7/home/linjiongxiao/.paos-rekep}"
export PAOS_REKEP_ROOT
export TMPDIR="$PAOS_REKEP_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PIP_CACHE_DIR="$PAOS_REKEP_ROOT/pip-cache"
export PIP_TMPDIR="$PAOS_REKEP_ROOT/tmp"
export UV_CACHE_DIR="$PAOS_REKEP_ROOT/uv-cache"
export XDG_CACHE_HOME="$PAOS_REKEP_ROOT/cache"
export XDG_DATA_HOME="$PAOS_REKEP_ROOT/data"
export HF_HOME="$PAOS_REKEP_ROOT/hf"
export TORCH_HOME="$PAOS_REKEP_ROOT/torch"
export CONDA_PKGS_DIRS="$PAOS_REKEP_ROOT/conda-pkgs"
export PYTHONPYCACHEPREFIX="$PAOS_REKEP_ROOT/pycache"
export PAOS_HOME="${PAOS_HOME:-/data7/home/linjiongxiao/.PhyAgentOS}"
export OMNIGIBSON_DATA_PATH="$PAOS_REKEP_ROOT/data/omnigibson"
export OMNIGIBSON_ASSET_PATH="$PAOS_REKEP_ROOT/data/omnigibson/assets"
export ISAAC_CACHE_PATH="$PAOS_REKEP_ROOT/data/isaac"
export UV_PYTHON_INSTALL_DIR="$PAOS_REKEP_ROOT/python"
set +a
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$CONDA_PKGS_DIRS" "$PYTHONPYCACHEPREFIX" "$OMNIGIBSON_DATA_PATH" "$ISAAC_CACHE_PATH" "$UV_PYTHON_INSTALL_DIR"
