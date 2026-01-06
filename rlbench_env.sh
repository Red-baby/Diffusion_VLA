#!/usr/bin/env bash
set -euo pipefail

# ---- CoppeliaSim path (adjust if you move it) ----
export COPPELIASIM_ROOT=/rl/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"

# ---- Qt/X11 backend ----
export QT_QPA_PLATFORM=xcb

# ---- (optional) avoid Qt warning; safe to set ----
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p "$XDG_RUNTIME_DIR" || true
chmod 700 "$XDG_RUNTIME_DIR" || true

# ---- Run under virtual X server ----
exec xvfb-run -a -s "-screen 0 1280x720x24" "$@"

export TORCH_HOME=/rl/torch_cache

