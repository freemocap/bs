import shutil
from pathlib import Path

import numpy as np
import polars as pl

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


def create_validation_analyzable_output(recording_folder: RecordingFolder) -> Path | None:
    solver_output = recording_folder.mocap_solver_output
    if solver_output is None:
        print("Skull solver output not found — skipping analyzable_output creation")
        return None

    analyzable_output_dir = recording_folder.folder_path / "analyzable_output"
    skull_kinematics_dir = analyzable_output_dir / "skull_kinematics"
    analyzable_output_dir.mkdir(exist_ok=True)
    skull_kinematics_dir.mkdir(exist_ok=True)

    shutil.copy2(solver_output / "skull_kinematics.csv", skull_kinematics_dir / "skull_kinematics.csv")
    shutil.copy2(solver_output / "skull_reference_geometry.json", skull_kinematics_dir / "skull_reference_geometry.json")
    shutil.copy2(solver_output / "skull_and_spine_topology.json", skull_kinematics_dir / "skull_and_spine_topology.json")
    shutil.copy2(solver_output / "skull_and_spine_trajectories.csv", analyzable_output_dir / "skull_and_spine_trajectories_resampled.csv")

    df = pl.read_csv(solver_output / "skull_kinematics.csv")
    timestamps = (
        df.select(["frame", "timestamp_s"])
        .unique()
        .sort(by="frame")
        ["timestamp_s"]
        .to_numpy()
        .astype(np.float64)
    )
    timestamps -= timestamps[0]
    np.save(analyzable_output_dir / "common_timestamps.npy", timestamps)

    print(f"Analyzable output created at {analyzable_output_dir}")
    return analyzable_output_dir
