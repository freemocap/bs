"""
Ferret Gaze Pipeline Runner

Loads head kinematics, eye kinematics, computes gaze vectors,
and launches the Rerun visualization.
"""
from pathlib import Path

from ferret_head_kinematics import HeadKinematics, load_skull_pose, compute_head_kinematics
from ferret_eye_kinematics import EyeKinematics, load_eye_data
from ferret_gaze_kinematics import GazeKinematics, load_trajectory_data, compute_gaze_kinematics
from ferret_gaze_visualization import run_visualization


def run_gaze_pipeline(clip_path: Path) -> None:
    """Run the full gaze kinematics pipeline for a clip.

    Args:
        clip_path: Path to the clip folder containing mocap_data and eye_data subfolders
    """
    # Construct paths
    skull_pose_csv = clip_path / "mocap_data" / "output_data" / "solver_output" / "rotation_translation_data.csv"
    trajectory_csv = clip_path / "mocap_data" / "output_data" / "solver_output" / "tidy_trajectory_data.csv"
    eye_data_csv = clip_path / "eye_data" / "eye_data.csv"

    # Load head data
    print(f"Loading skull pose from {skull_pose_csv}...")
    timestamps, positions, quaternions = load_skull_pose(skull_pose_csv)
    print(f"  Loaded {len(timestamps)} frames")

    print("Computing head kinematics...")
    hk = compute_head_kinematics(
        timestamps=timestamps,
        positions=positions,
        quaternions=quaternions,
    )

    # Save head kinematics CSV
    head_kinematics_csv = skull_pose_csv.parent / "head_kinematics.csv"
    hk.to_dataframe().to_csv(head_kinematics_csv, index=False)
    print(f"  Saved: {head_kinematics_csv}")

    # Load trajectory data
    print(f"Loading trajectory data from {trajectory_csv}...")
    trajectory_data = load_trajectory_data(trajectory_csv)
    print(f"  Loaded {len(trajectory_data)} frames")

    # Load eye data
    print(f"Loading eye data from {eye_data_csv}...")
    ek = load_eye_data(eye_data_csv)
    print(f"  Left eye: {len(ek.left_eye_timestamps)} frames")
    print(f"  Right eye: {len(ek.right_eye_timestamps)} frames")

    # Save eye kinematics CSV
    eye_kinematics_csv = eye_data_csv.parent / "eye_kinematics.csv"
    ek.to_dataframe().to_csv(eye_kinematics_csv, index=False)
    print(f"  Saved: {eye_kinematics_csv}")

    # Compute gaze kinematics (resamples head and eye to 120Hz)
    print("Computing gaze kinematics...")
    gk, hk_resampled, ek_resampled = compute_gaze_kinematics(
        hk=hk,
        ek=ek,
        trajectory_data=trajectory_data,
    )

    # Save gaze kinematics CSV
    gaze_kinematics_csv = skull_pose_csv.parent / "gaze_kinematics.csv"
    gk.to_dataframe().to_csv(gaze_kinematics_csv, index=False)
    print(f"  Saved: {gaze_kinematics_csv}")

    # Save resampled kinematics CSVs
    head_resampled_csv = skull_pose_csv.parent / "head_kinematics_resampled.csv"
    hk_resampled.to_dataframe().to_csv(head_resampled_csv, index=False)
    print(f"  Saved: {head_resampled_csv}")

    eye_resampled_csv = eye_data_csv.parent / "eye_kinematics_resampled.csv"
    ek_resampled.to_dataframe().to_csv(eye_resampled_csv, index=False)
    print(f"  Saved: {eye_resampled_csv}")

    # Run visualization with resampled data
    print("Launching visualization...")
    run_visualization(
        hk=hk_resampled,
        trajectory_data=trajectory_data,
        trajectory_timestamps=timestamps,  # Original timestamps for skeleton
        gk=gk,
        ek=ek_resampled,
        application_id="ferret_gaze_kinematics",
    )


if __name__ == "__main__":
    clip_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s")
    run_gaze_pipeline(clip_path)