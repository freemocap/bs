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

import subprocess
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

# ── Bootstrap ────────────────────────────────────────────────────────
# Blender's bundled Python user site-packages (pip installs land here)
_BLENDER_SITE = (
    Path.home()
    / ".local"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
if str(_BLENDER_SITE) not in sys.path:
    sys.path.insert(0, str(_BLENDER_SITE))

# One-shot: pip-install freemocap_blender_addon so we can import the
# dependency manager.  After the first run this is a no-op.
try:
    import freemocap_blender_addon  # noqa: F401 (checked by name)
except ImportError:
    # Blender's Python may not have pip yet — ensure it first
    subprocess.run([sys.executable, "-m", "ensurepip", "--user"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        check=False,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/freemocap/freemocap_blender_addon@development",
        ],
        check=True,
    )

from freemocap_blender_addon import (  # noqa: E402 (path setup must come first)
    check_and_install_dependencies,
    resolve_git_sources,
)

# ── Dependencies ──────────────────────────────────────────────────────

# PyPI / pip-installable packages
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

# Git-cloned source packages — cloned to cache, updated on every run
_GIT_SOURCES = [
    {"git": "https://github.com/freemocap/bs", "branch": "jon/dev"},
    {"git": "https://github.com/freemocap/skellytracker", "branch": "development"},
    {"git": "https://github.com/freemocap/skellycam", "branch": "development"},
    {"git": "https://github.com/freemocap/freemocap", "branch": "jon/development"},
]
for _p in resolve_git_sources(_GIT_SOURCES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


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
