"""
Ferret Gaze Pipeline Runner

Loads skull kinematics, eye kinematics, computes gaze vectors,
and launches the Rerun visualization.
"""
from pathlib import Path

from python_code.ferret_gaze.kinematics_calculators.ferret_eye_kinematics import load_eye_data
from python_code.ferret_gaze.kinematics_calculators.ferret_gaze_kinematics import compute_gaze_kinematics
from python_code.ferret_gaze.helpers.data_resampling_helpers import resample_data
from python_code.ferret_gaze.helpers.body_trajectory_helpers import load_body_trajectory_data, \
    body_trajectories_to_dataframe
from python_code.ferret_gaze.kinematics_calculators.ferret_skull_kinematics import load_skull_pose, compute_skull_kinematics
from python_code.ferret_gaze.visualization.ferret_gaze_rerun import run_visualization
from python_code.ferret_gaze.visualization.plot_head_yaw_vs_eye_horizontal_velocity import \
    plot_head_yaw_vs_eye_horizontal_velocity


def run_gaze_pipeline(clip_path: Path, launch_visualization: bool = True) -> None:
    """Run the full gaze kinematics pipeline for a clip.

    Args:
        clip_path: Path to the clip folder containing mocap_data and eye_data subfolders
        launch_visualization: Whether to launch the Rerun visualization after processing
    """
    clip_path = Path(clip_path)
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip path does not exist: {clip_path}")

    # Construct paths
    skull_pose_csv = clip_path / "mocap_data" / "output_data" / "solver_output" / "skull_pose_data.csv"
    trajectory_csv = clip_path / "mocap_data" / "output_data" / "solver_output" / "skull_trajectory_data.csv"
    skull_reference_geometry_json = clip_path / "mocap_data" / "solver_output" / "skull_reference_geometry.json"
    left_eye_data_csv = clip_path / "eye_data" / "output_data" / "eye0_data.csv"
    right_eye_data_csv = clip_path / "eye_data" / "output_data" / "eye1_data.csv"
    analyzable_output_path = clip_path / "analyzable_output"
    analyzable_output_path.mkdir(parents=True, exist_ok=True)

    # Validate all required input files exist
    required_files = [skull_pose_csv, trajectory_csv, left_eye_data_csv, right_eye_data_csv]
    for filepath in required_files:
        if not filepath.exists():
            raise FileNotFoundError(f"Required input file not found: {filepath}")

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

    # Compute gaze kinematics (resamples skull and eye to common framerate)
    print("Resampling data to common timestamps...")
    (
        skull_resampled,
        left_eye_resampled,
        right_eye_resampled,
        body_trajectories_resampled,
    ) = resample_data(
        skull=skull,
        left_eye=left_eye,
        right_eye=right_eye,
        trajectory_data=body_trajectories,
    )

    print("Computing gaze kinematics...")
    gaze = compute_gaze_kinematics(
        skull=skull_resampled,
        left_eye=left_eye_resampled,
        right_eye=right_eye_resampled,
        trajectory_data=body_trajectories_resampled,
    )

    # Save gaze kinematics CSV
    gaze_kinematics_csv = analyzable_output_path / "gaze_kinematics.csv"
    gaze.to_dataframe().to_csv(gaze_kinematics_csv, index=False)
    print(f"  Saved: {gaze_kinematics_csv}")

    # Save resampled kinematics CSVs
    skull_resampled_csv = analyzable_output_path / "skull_kinematics.csv"
    skull_resampled.to_dataframe().to_csv(skull_resampled_csv, index=False)
    print(f"  Saved: {skull_resampled_csv}")

    left_eye_resampled_csv = analyzable_output_path / "left_eye_kinematics.csv"
    left_eye_resampled.to_dataframe().to_csv(left_eye_resampled_csv, index=False)
    print(f"  Saved: {left_eye_resampled_csv}")

    right_eye_resampled_csv = analyzable_output_path / "right_eye_kinematics.csv"
    right_eye_resampled.to_dataframe().to_csv(right_eye_resampled_csv, index=False)
    print(f"  Saved: {right_eye_resampled_csv}")

    body_keypoint_trajectories_csv = analyzable_output_path / "body_keypoint_trajectories.csv"
    body_trajectories_df = body_trajectories_to_dataframe(body_trajectories_resampled)
    body_trajectories_df.to_csv(body_keypoint_trajectories_csv, index=False)
    print(f"  Saved: {body_keypoint_trajectories_csv}")
    print("\nPipeline complete!")
    print(f"  Output directory: {analyzable_output_path}")

    plot_head_yaw_vs_eye_horizontal_velocity(skull=skull_resampled,
                                             left_eye=left_eye_resampled,
                                            right_eye=right_eye_resampled)
    # Launch visualization if requested
    if launch_visualization:
        print("\nLaunching visualization...")
        run_visualization(
            skull=skull_resampled,
            trajectory_data=body_trajectories_resampled,
            trajectory_timestamps=skull_resampled.timestamps,
            gaze=gaze,
            left_eye_resampled=left_eye_resampled,
            right_eye_resampled=right_eye_resampled,
            application_id="ferret_gaze_kinematics",
            spawn=True,
        )


if __name__ == "__main__":
    _clip_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s"
    )
    run_gaze_pipeline(_clip_path, launch_visualization=True)