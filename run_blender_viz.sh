#!/bin/bash
# ================================================================
# HOW TO RUN THIS:
#
#   cd ~/code_stuff/github/freemocap/clients/bs
#   ./run_blender_viz.sh
#
#   That's it. First run installs deps (~100MB, one-time).
# ================================================================

set -euo pipefail

# ── Locate ourselves ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"                    # clients/bs/
MONOREPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"              # github/freemocap/
BLENDER="$HOME/software/blender-5.1.2-linux-x64/blender"

# ── One-time: install external packages into Blender's Python ─
echo "=== Checking Blender dependencies ==="
"$BLENDER" --background --python-use-system-env --python-expr "
import subprocess, sys
deps = ['polars', 'pydantic', 'opencv-contrib-python', 'beartype']
for dep in deps:
    try:
        __import__(dep.replace('-', '_'))
    except ImportError:
        print(f'Installing {dep}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', dep],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('Dependencies ready.')
" 2>&1 | grep -v "^Read blend\|^Fra:\|^Blender\|^00:"

# ── PYTHONPATH (last prepended = searched FIRST) ──────────────
# Build in reverse: lowest priority first, highest last.
_BLENDER_USER_SITE="$HOME/.local/lib/python3.13/site-packages"

PYTHONPATH="$SCRIPT_DIR"                                                     # 5. lowest: python_code
PYTHONPATH="$MONOREPO_ROOT/project/skellytracker:$PYTHONPATH"                # 4. skellytracker source
PYTHONPATH="$MONOREPO_ROOT/project/skellycam:$PYTHONPATH"                    # 3. skellycam source
PYTHONPATH="$MONOREPO_ROOT/project/freemocap:$PYTHONPATH"                    # 2. freemocap source
PYTHONPATH="$MONOREPO_ROOT/project/freemocap_blender_addon:$PYTHONPATH"      # 1. blender addon source
PYTHONPATH="$_BLENDER_USER_SITE:$PYTHONPATH"                                 # 0. HIGHEST: py3.13 pkgs
export PYTHONPATH

# ── Run ───────────────────────────────────────────────────────
echo "=== Running Blender ==="
exec "$BLENDER" --background --python-use-system-env --python "$SCRIPT_DIR/python_code/viz/blender/__main_blender.py" "$@"
