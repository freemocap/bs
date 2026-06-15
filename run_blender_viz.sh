#!/bin/bash
# ================================================================
# HOW TO RUN THIS:
#
#   cd ~/code_stuff/github/freemocap/clients/bs
#   ./run_blender_viz.sh
#
# The Python bootstrap in __main_blender.py handles:
#   - pip-installing freemocap_blender_addon
#   - checking/installing all Python dependencies
#   - cloning/updating git-sourced packages (bs, skellytracker, etc.)
#   - setting up sys.path
# ================================================================

set -euo pipefail

# ── Locate ourselves ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"                    # clients/bs/
BLENDER="${BLENDER_BIN:-$HOME/software/blender-5.1.2-linux-x64/blender}"

# ── Run ───────────────────────────────────────────────────────
echo "=== Running Blender ==="
exec "$BLENDER" --background --python-use-system-env \
    --python "$SCRIPT_DIR/python_code/viz/blender/__main_blender.py" "$@"
