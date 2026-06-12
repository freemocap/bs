import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
BS_ROOT = THIS_FILE.parents[3]                                 # clients/bs/
MONOREPO = THIS_FILE.parents[5]                                # github/freemocap/
_BLENDER_SITE = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

# Insert in reverse: lowest priority first, highest last
_VENV = next((BS_ROOT / ".venv" / "lib").glob("python*/site-packages"), None)
if _VENV and str(_VENV) not in sys.path:
    sys.path.insert(0, str(_VENV))
for _p in [BS_ROOT,
           MONOREPO / "project" / "skellytracker",
           MONOREPO / "project" / "skellycam",
           MONOREPO / "project" / "freemocap",
           MONOREPO / "project" / "freemocap_blender_addon",
           _BLENDER_SITE]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Dependencies ──────────────────────────────────────────────
from freemocap_blender_addon import check_and_install_dependencies

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

# ── Dev reload: nuke edited modules from sys.modules so runpy picks up changes ──
# TODO - this is hacky nonsense, need better fix
def _discover_editable_modules(root: Path, package_prefix: str) -> list[str]:
    """Walk ``root`` and return every non-``__init__`` .py file as a dotted module name.

    Packages (``__init__.py``) are skipped — popping a package from ``sys.modules``
    breaks import resolution for all its children.
    """
    modules: list[str] = []
    for py_file in root.rglob("*.py"):
        if py_file.stem == "__init__":
            continue  # skip packages — only pop leaf modules
        relative = py_file.relative_to(root)
        parts = list(relative.with_suffix("").parts)
        modules.append(f"{package_prefix}." + ".".join(parts))
    return modules

# --- python_code modules (auto-discovered) ---
_BS_PYTHON_CODE = BS_ROOT / "python_code"
_FREEMOCAP_CODE = BS_ROOT / "python_code"
_EDITABLE_MODULES: list[str] = _discover_editable_modules(_BS_PYTHON_CODE, "python_code")




for _mod in _EDITABLE_MODULES:
    sys.modules.pop(_mod, None)

from python_code.viz.blender.blender_helpers.blender_recording_model import BlenderRecording
from python_code.viz.blender.blender_helpers.create_blender_scene import create_blender_scene 



def main_blender(recording_path: Path | str):
    print(f"Creating Blender scene for recording at {recording_path}...")
    blender_recording = BlenderRecording.from_recording_path(recording_path)
    print(f"Blender recording created for {recording_path}\n\n\n")
    create_blender_scene(blender_recording)
    print(f"Blender scene created for {recording_path}")


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
                