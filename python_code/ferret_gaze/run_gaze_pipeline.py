"""
Ferret Gaze Pipeline Runner

Loads skull kinematics, eye kinematics, computes gaze vectors,
and launches the Rerun visualization.
"""
from pathlib import Path

from python_code.ferret_gaze.kinematics_calculators.ferret_eye_kinematics import load_eye_data
from python_code.ferret_gaze.kinematics_calculators.ferret_gaze_kinematics import load_body_trajectory_data, \
    compute_gaze_kinematics, resample_data
from python_code.ferret_gaze.kinematics_calculators.ferret_skull_kinematics import load_skull_pose, \
    compute_skull_kinematics
from python_code.ferret_gaze.visualization.ferret_gaze_rerun import run_visualization


def run_gaze_pipeline(clip_path: Path) -> None:
    """Run the full gaze kinematics pipeline for a clip.

    Args:
        clip_path: Path to the clip folder containing mocap_data and eye_data subfolders
    """
    # Construct paths
    skull_pose_csv = clip_path / "mocap_data" / "output_data" / "solver_output" / "skull_pose.csv"
    trajectory_csv = clip_path / "mocap_data" / "output_data" / "solver_output" / "skull_trajectories.csv"
    left_eye_data_csv = clip_path / "eye_data" / "output_data" / "eye0_data.csv"
    right_eye_data_csv = clip_path / "eye_data" / "output_data" / "eye1_data.csv"
    analyzable_output_path = clip_path / "analyzable_output"
    analyzable_output_path.mkdir(parents=True, exist_ok=True)

    # Load skull data
    print(f"Loading skull pose from {skull_pose_csv}...")
    timestamps, positions, quaternions = load_skull_pose(skull_pose_csv)
    print(f"  Loaded {len(timestamps)} frames")

    print("Computing skull kinematics...")
    skull = compute_skull_kinematics(
        timestamps=timestamps,
        positions=positions,
        quaternions=quaternions,
    )

    # Save skull kinematics CSV
    skull_kinematics_csv = skull_pose_csv.parent / "skull_kinematics.csv"
    skull.to_dataframe().to_csv(skull_kinematics_csv, index=False)
    print(f"  Saved: {skull_kinematics_csv}")

    # Load trajectory data
    print(f"Loading trajectory data from {trajectory_csv}...")
    body_trajectories = load_body_trajectory_data(trajectory_csv)

    # Load eye data
    print(f"Loading eye data from {left_eye_data_csv}...")
    left_eye = load_eye_data(left_eye_data_csv)
    print(f"  Left eye: {len(left_eye.timestamps)} frames")

    print(f"Loading eye data from {right_eye_data_csv}...")
    right_eye = load_eye_data(right_eye_data_csv)
    print(f"  Right eye: {len(right_eye.timestamps)} frames")

    # Save eye kinematics CSV
    left_eye_kinematics_csv = left_eye_data_csv.parent / "left_eye_kinematics.csv"
    left_eye.to_dataframe().to_csv(left_eye_kinematics_csv, index=False)
    print(f"  Saved: {left_eye_kinematics_csv}")
    right_eye_kinematics_csv = right_eye_data_csv.parent / "right_eye_kinematics.csv"
    right_eye.to_dataframe().to_csv(right_eye_kinematics_csv, index=False)
    print(f"  Saved: {right_eye_kinematics_csv}")

    # Compute gaze kinematics (resamples skull and eye to 120Hz)
    print("Computing gaze kinematics...")
    (skull_resampled,
     left_eye_resampled,
     right_eye_resampled,
     body_trajectories_resampled) = resample_data(
        skull=skull,
        left_eye=left_eye,
        right_eye=right_eye,
        body_trajectories=body_trajectories,
    )

    gaze = compute_gaze_kinematics(
        skull=skull_resampled,
        left_eye=left_eye_resampled,
        right_eye=right_eye_resampled,
        body_trajectories=body_trajectories_resampled,
    )

    # Save gaze kinematics CSV
    gaze_kinematics_csv = analyzable_output_path / "gaze_kinematics.csv"
    gaze.to_dataframe().to_csv(gaze_kinematics_csv, index=False)
    print(f"  Saved: {gaze_kinematics_csv}")

    # Save resampled kinematics CSVs
    skull_resampled_csv = analyzable_output_path / "skull_kinematics_resampled.csv"
    skull_resampled.to_dataframe().to_csv(skull_resampled_csv, index=False)
    print(f"  Saved: {skull_resampled_csv}")

    left_eye_resampled_csv = analyzable_output_path / "eye_kinematics_resampled.csv"
    left_eye_resampled.to_dataframe().to_csv(left_eye_resampled_csv, index=False)
    print(f"  Saved: {left_eye_resampled_csv}")

    right_eye_resampled_csv = analyzable_output_path / "eye_kinematics_resampled.csv"
    right_eye_resampled.to_dataframe().to_csv(right_eye_resampled_csv, index=False)
    print(f"  Saved: {right_eye_resampled_csv}")

    gaze_kinematics_resampled_csv = analyzable_output_path / "gaze_kinematics_resampled.csv"
    gaze.to_dataframe().to_csv(gaze_kinematics_resampled_csv, index=False)
    print(f"  Saved: {gaze_kinematics_resampled_csv}")

    # Run visualization with resampled data
    print("Launching visualization...")
    run_visualization(
        hk=skull_resampled,
        trajectory_data=body_trajectories,
        trajectory_timestamps=timestamps,  # Original timestamps for skeleton
        gaze=gaze,
        ek=left_eye,
        application_id="ferret_gaze_kinematics",
    )


if __name__ == "__main__":
    _clip_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s")
    run_gaze_pipeline(_clip_path)
