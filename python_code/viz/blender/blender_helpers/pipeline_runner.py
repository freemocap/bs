"""Thin wrapper that builds a Blender scene from a recording path.

This module exists separately from ``__main_blender.py`` so both the
headless script and the ``bs_blender_addon`` operator can import it
without triggering ``__main_blender``'s side effects (path setup, module
cache purging, dependency installation).
"""

from pathlib import Path

from python_code.viz.blender.blender_helpers.blender_recording_model import (
    BlenderRecording,
)
from python_code.viz.blender.blender_helpers.create_blender_scene import (
    create_blender_scene,
)


def run_pipeline(recording_path: Path | str) -> None:
    """Load a recording from disk and build the full Blender scene.

    Args:
        recording_path: Path to the recording directory (the inner
            ``full_recording`` folder containing calibration TOML,
            synchronized videos, trajectories, kinematics, etc.).
    """
    recording_path = Path(recording_path)
    print(f"Creating Blender scene for recording at {recording_path}...")
    blender_recording = BlenderRecording.from_recording_path(recording_path)
    print(f"Blender recording created for {recording_path}\n\n\n")
    create_blender_scene(blender_recording)
    print(f"Blender scene created for {recording_path}")
