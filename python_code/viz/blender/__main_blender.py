import sys
from pathlib import Path


# Add project root to sys.path so Blender can find the `python_code` module
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add the project's venv site-packages so Blender's Python can find installed packages
_VENV_SITE_PACKAGES = next((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"), None)
if _VENV_SITE_PACKAGES and str(_VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE_PACKAGES))

# Install dependencies to Blender's python
from freemocap_blender_addon import check_and_install_dependencies

BLENDER_DEPENDENCIES = ["polars", "pydantic", { "git": "https://github.com/freemocap/freemocap", "branch": "development" }]
check_and_install_dependencies(BLENDER_DEPENDENCIES)

from python_code.viz.blender.blender_helpers.blender_recording_model import BlenderRecording
from python_code.viz.blender.blender_helpers.create_blender_scene import create_blender_scene


def main_blender(recording_path: Path | str):
    blender_recording = BlenderRecording.from_recording_path(recording_path)
    create_blender_scene(blender_recording)


if __name__ == "__main__":
    # TODO - Shouldn't need to target the INNER `full_recording` folder
    RECORDING_PATH = Path(
        "/media/jon-alien-pop/DATA/bs/session_2025-10-22_ferret_420_EO13/session_2025-10-22_ferret_420_EO13/full_recording")
    main_blender(RECORDING_PATH)
    print('Done!')
                