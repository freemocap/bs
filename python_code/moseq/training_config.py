"""
Per-project record of which recordings a Keypoint MoSeq model was trained on.

`main()` in `run_moseq_pipeline.py` writes one YAML sidecar file per kpms
project directory (`training_recordings.yaml`) recording the loader and the
exact `RecordingFolder` paths used for training, so that downstream scripts
(`replot_session.py`, `visualization/eye_syllable_viz.py`) never have to
duplicate that list by hand and risk drifting from what the model was
actually fit on -- they just read it back from the project directory.

Usage
-----
    from python_code.moseq.training_config import (
        load_training_config,
        load_training_recording_folders,
    )

    recording_folders = load_training_recording_folders(project_dir)
    loader = KPMS_Loader(load_training_config(project_dir)["loader"])
"""
from pathlib import Path

import yaml

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

_CONFIG_FILENAME = "training_recordings.yaml"


def training_config_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / _CONFIG_FILENAME


def save_training_config(
    project_dir: str | Path,
    loader: str,
    recording_folders: list[RecordingFolder],
) -> None:
    """
    Write the loader name and exact recording folder paths used for training
    to `{project_dir}/training_recordings.yaml`, overwriting any existing
    file for this project.

    Parameters
    ----------
    loader:
        A `KPMS_Loader` value string (e.g. `KPMS_Loader.HEAD_WITH_PUPIL_POINTS.value`).
    recording_folders:
        The exact `RecordingFolder`s passed to training, in order.
    """
    path = training_config_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "loader": loader,
        "recording_paths": [str(rf.folder_path) for rf in recording_folders],
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def load_training_config(project_dir: str | Path) -> dict:
    """Raw `{"loader": ..., "recording_paths": [...]}` dict for a project."""
    path = training_config_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No training config found at {path}. This project was likely set up "
            "before training-config tracking was added -- pass loader/source "
            "explicitly instead of relying on this file."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def load_training_recording_folders(project_dir: str | Path) -> list[RecordingFolder]:
    """Rebuild the exact list of `RecordingFolder`s a project's model was
    trained on, from its saved training config, in the original order."""
    config = load_training_config(project_dir)
    return [RecordingFolder.from_folder_path(p) for p in config["recording_paths"]]
