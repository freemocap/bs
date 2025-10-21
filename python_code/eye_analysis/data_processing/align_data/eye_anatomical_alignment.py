"""Spatial correction for eye tracking data to establish anatomical coordinate system.

Applies transformations to align data with anatomical axes:
1. Translate: tear_duct → origin (0,0)
2. Rotate: eye_outer → X-axis (lateral-nasal alignment)
3. Center: mode of pupil center → origin (resting position)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

from python_code.eye_analysis.data_models.csv_io import load_trajectory_dataset
from python_code.eye_analysis.data_models.trajectory_dataset import (
    Trajectory2D,
    ProcessedTrajectory,
    TrajectoryDataset,
    DEFAULT_BUTTERWORTH_CUTOFF,
    DEFAULT_BUTTERWORTH_ORDER,
    DEFAULT_MIN_CONFIDENCE,
)
from python_code.eye_analysis.data_processing.align_data.alignment_diagnostics import (
    plot_correction_comparison,
    get_correction_summary,
)


def apply_spatial_transform_to_trajectory(
    *,
    trajectory: Trajectory2D,
    stabilize_on: np.ndarray,  # (n_frames, 2)
    rotate_by: np.ndarray,  # (n_frames,)
    center_on: np.ndarray,  # (2,)
) -> Trajectory2D:
    n_frames: int = len(trajectory.data)
    transformed_data: np.ndarray = np.zeros_like(a=trajectory.data)

    for frame_number in range(n_frames):
        # Step 1: Translate by tear duct
        translated_point: np.ndarray = (
            trajectory.data[frame_number] - stabilize_on[frame_number]
        )

        # Step 2: Rotate to align eye_outer with X-axis
        angle: float = float(rotate_by[frame_number])
        cos_a: float = float(np.cos(angle))
        sin_a: float = float(np.sin(angle))
        rotation_matrix: np.ndarray = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_point: np.ndarray = rotation_matrix @ translated_point

        # Step 3: Shift by mode offset
        transformed_data[frame_number] = rotated_point - center_on

    return Trajectory2D(
        name=trajectory.name,
        data=transformed_data,
        timestamps=trajectory.timestamps,
        confidence=trajectory.confidence,
        metadata={**trajectory.metadata, "spatially_corrected": True},
    )


def compute_spatial_correction_parameters(
    *,
    stabilize_on: np.ndarray,  # (n_frames, 2)
    align_to: np.ndarray,  # (n_frames, 2)
    center_on: np.ndarray,  # (n_frames,2)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames: int = stabilize_on.shape[0]
    if n_frames != align_to.shape[0]:
        raise ValueError(
            "stabilize_on and align_to must have the same number of frames"
        )

    # Compute rotation angles
    # After translating by `stabilize_on` point, compute angle to rotate `align_to` point onto X-axis
    translated_align_to: np.ndarray = align_to - stabilize_on  # (n_frames, 2)
    rotation_angles: np.ndarray = -np.arctan2(
        translated_align_to[:, 1], translated_align_to[:, 0]
    )

    transformed_center_on: np.ndarray = np.zeros_like(a=center_on)
    for frame_number in range(n_frames):

        translated = center_on[frame_number] - stabilize_on[frame_number]

        angle = float(rotation_angles[frame_number])
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        transformed_center_on[frame_number] = rotation_matrix @ translated

    transformed_centering_offset: np.ndarray = np.median(
        transformed_center_on, axis=0
    )  # (2,)
    return stabilize_on, rotation_angles, transformed_centering_offset


def apply_spatial_correction_to_dataset(
    *,
    dataset: TrajectoryDataset,
    stabilize_on: str,
    align_to: str,
    center_on: list[str],
) -> TrajectoryDataset:
    # Validate required landmarks exist
    required: list[str] = [stabilize_on, align_to] + center_on
    missing: list[str] = [name for name in required if name not in dataset.trajectories]
    if missing:
        raise ValueError(f"Missing required landmarks: {missing}")

    aligned_trajectories: dict[str, ProcessedTrajectory] = {}

    for name, traj in dataset.trajectories.items():
        # Process raw trajectory
        # Compute correction parameters from cleaned data

        pupil_center_points = np.asarray(
            [dataset.trajectories[pname].cleaned.data for pname in center_on]
        )
        pupil_center_median_trajectory = np.nanmedian(
            pupil_center_points, axis=0
        )  # TODO: we'll want to save this too, probably add as trajectory

        tear_duct_pos, rotation_angles, transformed_pupil_global_median = (
            compute_spatial_correction_parameters(
                stabilize_on=dataset.trajectories[stabilize_on].cleaned.data,
                align_to=dataset.trajectories[align_to].cleaned.data,
                center_on=pupil_center_median_trajectory,
            )
        )

        corrected_raw: Trajectory2D = apply_spatial_transform_to_trajectory(
            trajectory=traj.raw,
            stabilize_on=tear_duct_pos,
            rotate_by=rotation_angles,
            center_on=transformed_pupil_global_median,
        )

        corrected_cleaned: Trajectory2D = apply_spatial_transform_to_trajectory(
            trajectory=traj.cleaned,
            stabilize_on=tear_duct_pos,
            rotate_by=rotation_angles,
            center_on=transformed_pupil_global_median,
        )

        aligned_trajectories[name] = ProcessedTrajectory(
            raw=corrected_raw, cleaned=corrected_cleaned
        )

    return TrajectoryDataset(
        name=dataset.name,
        trajectories=aligned_trajectories,
        frame_indices=dataset.frame_indices,
        metadata={
            **dataset.metadata,
            "spatially_corrected": True,
            "correction_params": {
                "tear_duct_name": stabilize_on,
                "outer_eye_name": align_to,
            },
            "alignment_trajectories": {
                "stabilize_on": stabilize_on,
                "align_to": align_to,
                "center_on": center_on,
            },
        },
    )

def merge_eye_output_csvs(
    eye_data_path: Path,
) -> pd.DataFrame:
    output_data_path = eye_data_path / "output_data"

    eye_0_output = pd.read_csv(output_data_path / "eye0_data.csv")
    eye_1_output = pd.read_csv(output_data_path / "eye1_data.csv")

    return pd.concat([eye_0_output, eye_1_output], axis=0).sort_values(["frame", "keypoint"])

def eye_alignment_main(
    recording_name: str, csv_path: Path, timestamps_path: Path, output_path: Path
) -> TrajectoryDataset:
    """Run spatial correction example."""

    eye_name = csv_path.stem.split("_")[0]

    # Load dataset
    print("Loading eye tracking dataset...")
    eye_dataset: TrajectoryDataset = load_trajectory_dataset(
        filepath=csv_path,
        timestamps=np.load(timestamps_path) / 1e9,  # Convert ns to s
        butterworth_cutoff=DEFAULT_BUTTERWORTH_CUTOFF,
        butterworth_order=DEFAULT_BUTTERWORTH_ORDER,
        min_confidence=DEFAULT_MIN_CONFIDENCE,
        name=f"{recording_name}_eye_dataset_raw",
    )

    # Apply spatial correction
    print("\nApplying spatial correction...")
    print("  Step 1: Translating by tear duct position...")
    print("  Step 2: Rotating to align outer eye with X-axis...")
    print("  Step 3: Centering by pupil global median...")

    corrected_dataset: TrajectoryDataset = apply_spatial_correction_to_dataset(
        dataset=eye_dataset,
        stabilize_on="tear_duct",
        align_to="outer_eye",
        center_on=[f"p{i}" for i in range(1, 9)],
    )  # TODO: dump to (1 tidy)csv, contains raw and cleaned trajectories for everything
    # frame number, timestamp (s from recording start), eye, x, y, processing level (raw, clean) (maybe include correction params)
    # recordingname_eye_data.csv
    # maybe some day add extras
    tidy_dataframe = corrected_dataset.to_tidy_dataset(eye_name=eye_name)
    tidy_dataframe.to_csv(
        output_path / f"{eye_name}_data.csv", index=False
    )

    summary: dict[str, float] = get_correction_summary(
        original_dataset=eye_dataset,
        corrected_dataset=corrected_dataset,
        stabilized_on="tear_duct",
        aligned_to="outer_eye",
    )
    print(json.dumps(summary, indent=4))
    with open(output_path / f"{eye_name}_alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Plot comparison
    print("\nGenerating comparison plot...")
    plot_correction_comparison(
        original_dataset=eye_dataset,
        corrected_dataset=corrected_dataset,
        output_path=output_path / f"{eye_name}_correction_comparison.png",
    )

    return corrected_dataset


if __name__ == "__main__":
    # Setup paths
    _csv_path: Path = Path(
        "/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/eye_data/dlc_output/model_outputs_iteration_11/eye1_clipped_4358_11527DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )
    _timestamps_npy_path: Path = Path(
        "/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/eye_data/eye_videos/eye1_timestamps_utc_clipped_4358_11527.npy"
    )
    _recording_name: str = (
        "2025-07-11_ferret_757_EyeCameras_P43_E15__1_0m37s-1m37s_eye1"
    )
    _output_path: Path = Path(
        "/Users/philipqueen/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/eye_data/output_data"
    )
    _output_path.mkdir(parents=True, exist_ok=True)
    eye_alignment_main(
        csv_path=_csv_path,
        timestamps_path=_timestamps_npy_path,
        recording_name=_recording_name,
        output_path=_output_path,
    )
