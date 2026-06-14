"""``bs_blender_addon`` — Simple Blender UI for the BS recording pipeline.

Provides a panel in the 3D Viewport sidebar (press N → "BS" tab) with:
- **Clear Scene** — wipe all objects / collections / orphan data
- **Recording Folder** — directory picker for a recording folder
- **Load Recording** — run the full pipeline on the selected folder

Architecture
------------
Operators are **thin wrappers** — they call into ``blender_helpers/`` which
contains the actual working code.  No business logic lives in UI files.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── bl_info (legacy format — works Blender 3.x → 5.x) ───────────
bl_info = {
    "name": "BS Recorder",
    "author": "Jon Matthis",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "3D Viewport > Sidebar > BS",
    "description": (
        "Load BS ferret recording data into Blender — "
        "keypoint trajectories, rigid-body kinematics, eye/gaze data, "
        "calibrated cameras with video planes."
    ),
    "category": "Animation",
}

# ── Path setup ──────────────────────────────────────────────────
# When Blender loads this addon (via symlink in scripts/addons/) it
# does NOT have our project directories on sys.path.  Set them up so
# ``python_code`` and the monorepo packages are importable.
_THIS_FILE = Path(__file__).resolve()
_BS_ROOT = _THIS_FILE.parents[4]  # clients/bs/
_MONOREPO = _THIS_FILE.parents[6]  # github/freemocap/

# Blender's bundled Python user site-packages
_BLENDER_SITE = (
    Path.home()
    / ".local"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)

# Insert in reverse priority order: lowest first, highest last.
# After the loop, each subsequent insert(0) pushes earlier entries down,
# so the LAST item in the list ends up at sys.path[0] (highest priority).
#
# VENV goes FIRST (lowest priority) so that source-directory
# ``freemocap`` / ``skellycam`` / etc. override any stale uv-managed
# installs of the same packages in the venv.
_VENV = next((_BS_ROOT / ".venv" / "lib").glob("python*/site-packages"), None)
if _VENV and str(_VENV) not in sys.path:
    sys.path.insert(0, str(_VENV))

for _p in [
    _BS_ROOT,
    _MONOREPO / "project" / "skellytracker",
    _MONOREPO / "project" / "skellycam",
    _MONOREPO / "project" / "freemocap",
    _MONOREPO / "project" / "freemocap_blender_addon",
    _BLENDER_SITE,
]:
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

# ── Dependency bootstrap ─────────────────────────────────────────
# Install missing packages into Blender's Python (one-time, ~100 MB).
# Same list as ``__main_blender.py``.
try:
    from freemocap_blender_addon import check_and_install_dependencies

    _BLENDER_DEPS = [
        "polars",
        "pydantic",
        "opencv-contrib-python",
        "tabulate",
        "toml",
        "pyyaml",
        "scipy",
        "numpydantic",
        {"git": "https://github.com/freemocap/skellylogs"},
    ]
    check_and_install_dependencies(_BLENDER_DEPS)
except Exception as _e:
    print(
        f"[bs_blender_addon] WARNING: Dependency check failed: {_e}\n"
        f"  Run 'run_blender_viz.sh' once to install dependencies, "
        f"or install them manually."
    )


# ── Registration ─────────────────────────────────────────────────
def register() -> None:
    """Register property groups, operators, and panels with Blender."""
    import bpy

    print("[bs_blender_addon] Registering BS Recorder addon …")

    # 1. Properties (must come first — operators/panels reference them)
    from .properties import BS_PROPERTY_GROUPS

    for cls in BS_PROPERTY_GROUPS:
        bpy.utils.register_class(cls)

    # Attach recording-path property to Scene
    bpy.types.Scene.bs_recording_path = bpy.props.StringProperty(
        name="Recording Path",
        description="Path to a BS recording folder (the inner full_recording directory)",
        subtype="DIR_PATH",
        default="",
    )

    # 2. Operators
    from .operators import BS_OPERATORS

    for cls in BS_OPERATORS:
        bpy.utils.register_class(cls)

    # 3. Panels
    from .panels import BS_PANELS

    for cls in BS_PANELS:
        bpy.utils.register_class(cls)

    print("[bs_blender_addon] Registration complete.")


def unregister() -> None:
    """Unregister panels, operators, and properties (LIFO order)."""
    import bpy

    print("[bs_blender_addon] Unregistering BS Recorder addon …")

    # 1. Panels
    from .panels import BS_PANELS

    for cls in reversed(BS_PANELS):
        bpy.utils.unregister_class(cls)

    # 2. Operators
    from .operators import BS_OPERATORS

    for cls in reversed(BS_OPERATORS):
        bpy.utils.unregister_class(cls)

    # 3. Properties
    from .properties import BS_PROPERTY_GROUPS

    for cls in reversed(BS_PROPERTY_GROUPS):
        bpy.utils.unregister_class(cls)

    # Detach Scene property
    del bpy.types.Scene.bs_recording_path

    print("[bs_blender_addon] Unregistration complete.")


# ── Direct run (for development / testing) ───────────────────────
if __name__ == "__main__":
    register()
