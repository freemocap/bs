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
_EDITABLE_MODULES = [
    "python_code.viz.blender.blender_helpers.blender_recording_model",
    "python_code.viz.blender.blender_helpers.create_blender_scene",
    "python_code.viz.blender.blender_helpers.set_scene_parameters",
    "python_code.viz.blender.blender_helpers.create_arena",
    "python_code.viz.blender.blender_helpers.add_cameras",
    "python_code.kinematics_core.keypoint_trajectories",
    "python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization",
    "freemocap.core.tasks.calibration.shared.camera_model",
    "freemocap.core.pipeline.posthoc.video_group_helper",
]
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
    main_blender(RECORDING_PATH)
    print('Done!')
                