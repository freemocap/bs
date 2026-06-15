import subprocess
import sys
from pathlib import Path

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

# PyPI / pip-installable packages (handled by freemocap_blender_addon)
BLENDER_DEPENDENCIES = [
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
check_and_install_dependencies(BLENDER_DEPENDENCIES)

# Git-cloned source packages — cloned/fetched to a cache dir and added
# to sys.path.  The manager checks for new commits on every run.
GIT_SOURCES = [
    {"git": "https://github.com/freemocap/bs", "branch": "jon/dev"},
    {"git": "https://github.com/freemocap/skellytracker", "branch": "development"},
    {"git": "https://github.com/freemocap/skellycam", "branch": "development"},
    {"git": "https://github.com/freemocap/freemocap", "branch": "jon/development"},
]
_RESOLVED_PATHS: list[Path] = resolve_git_sources(GIT_SOURCES)
for _p in _RESOLVED_PATHS:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Dev reload: nuke edited modules from sys.modules ─────────────────
# TODO - this is hacky nonsense, need better fix
def _discover_editable_modules(root: Path, package_prefix: str) -> list[str]:
    """Walk ``root`` and return every non-``__init__`` .py file as a
    dotted module name.

    Packages (``__init__.py``) are skipped — popping a package from
    ``sys.modules`` breaks import resolution for all its children.
    """
    modules: list[str] = []
    for py_file in root.rglob("*.py"):
        if py_file.stem == "__init__":
            continue  # skip packages — only pop leaf modules
        relative = py_file.relative_to(root)
        parts = list(relative.with_suffix("").parts)
        modules.append(f"{package_prefix}." + ".".join(parts))
    return modules

# Find the git-cached bs repo (first path whose name is "bs")
_BS_CACHE = next((_p for _p in _RESOLVED_PATHS if _p.name == "bs"), None)
if _BS_CACHE:
    _EDITABLE_MODULES: list[str] = _discover_editable_modules(
        _BS_CACHE / "python_code", "python_code"
    )
    for _mod in _EDITABLE_MODULES:
        sys.modules.pop(_mod, None)
from python_code.viz.blender.blender_helpers.pipeline_runner import (
    run_pipeline as main_blender,
)


if __name__ == "__main__" or __name__ == "<run_path>":
    # TODO - Shouldn't need to target the INNER `full_recording` folder
    RECORDING_PATH = Path(
        "/media/jon-alien-pop/DATA/bs/session_2025-10-22_ferret_420_EO13/session_2025-10-22_ferret_420_EO13/full_recording")
    # RECORDING_PATH = Path(
    #     "/media/jon-alien-pop/DATA/bs/session_2026-03-14_ferret_407_P47_E14/session_2026-03-14_ferret_407_P47_E14/full_recording/") #BROKEN!!! MISSING MOCAP SYNC VIDEOS?? MISSING OUTPUT_DATA FOLDER??? DOESNT FAIL ON LOAD JUST SILENTLY RETURNS `NONE` FOLDER!!!!!
    # RECORDING_PATH = Path(
    #     "/media/jon-alien-pop/DATA/bs/session_2026-03-18_ferret_416_P51_E12/session_2026-03-18_ferret_416_P51_E12/full_recording/") # LEFT EYE IS GARBAGE - CAMERA LOOSE????

    main_blender(RECORDING_PATH)
    print('Done!')
                