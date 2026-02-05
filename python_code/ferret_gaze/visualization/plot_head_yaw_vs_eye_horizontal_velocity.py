"""
Analysis Helper: Head Yaw Velocity vs Eye Horizontal Velocity

Loads ferret kinematics data from disk and creates a scatter plot
comparing head yaw velocity with eye horizontal velocity for both eyes.

Usage:
    python analyze_head_yaw_vs_eye_velocity.py [analyzable_output_dir] [output_figure_path]

    If no arguments provided, uses DEFAULT_ANALYZABLE_OUTPUT_DIR below.

Directory Structure Expected:
    analyzable_output/
    ├── skull_kinematics/
    │   ├── skull_kinematics.csv
    │   └── skull_reference_geometry.json
    ├── left_eye_kinematics/
    │   ├── left_eye_kinematics.csv
    │   └── left_eye_reference_geometry.json
    └── right_eye_kinematics/
        ├── right_eye_kinematics.csv
        └── right_eye_reference_geometry.json
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from numpy.typing import NDArray

# =============================================================================
# Derived paths (computed from analyzable output directory)
# =============================================================================

SKULL_KINEMATICS_SUBDIR = "skull_kinematics"
SKULL_KINEMATICS_CSV = "skull_kinematics.csv"
SKULL_REFERENCE_GEOMETRY_JSON = "skull_reference_geometry.json"

LEFT_EYE_KINEMATICS_SUBDIR = "left_eye_kinematics"
LEFT_EYE_KINEMATICS_CSV = "left_eye_kinematics.csv"
LEFT_EYE_REFERENCE_GEOMETRY_JSON = "left_eye_reference_geometry.json"

RIGHT_EYE_KINEMATICS_SUBDIR = "right_eye_kinematics"
RIGHT_EYE_KINEMATICS_CSV = "right_eye_kinematics.csv"
RIGHT_EYE_REFERENCE_GEOMETRY_JSON = "right_eye_reference_geometry.json"

# =============================================================================


def load_skull_kinematics_from_directory(
    kinematics_dir: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load skull kinematics and extract timestamps and yaw angular velocity.

    Args:
        kinematics_dir: Directory containing skull_kinematics.csv and
                        skull_reference_geometry.json

    Returns:
        Tuple of (timestamps, head_yaw_velocity_deg_s)
    """
    kinematics_csv = kinematics_dir / SKULL_KINEMATICS_CSV
    if not kinematics_csv.exists():
        raise FileNotFoundError(f"Skull kinematics CSV not found: {kinematics_csv}")

    df = pl.read_csv(kinematics_csv)

    # Extract timestamps
    timestamps = (
        df.select(["frame", "timestamp_s"])
        .unique()
        .sort("frame")
        ["timestamp_s"]
        .to_numpy()
        .astype(np.float64)
    )

    # Extract local angular velocity yaw component
    # Angular velocity local has components [roll, pitch, yaw] based on the serialization
    angular_velocity_local_df = df.filter(pl.col("trajectory") == "angular_velocity_local")

    # Get the yaw component (head yaw is rotation around the vertical axis)
    yaw_df = (
        angular_velocity_local_df
        .filter(pl.col("component") == "yaw")
        .sort("frame")
    )

    head_yaw_velocity_rad_s = yaw_df["value"].to_numpy().astype(np.float64)
    head_yaw_velocity_deg_s = np.rad2deg(head_yaw_velocity_rad_s)

    return timestamps, head_yaw_velocity_deg_s


def load_eye_kinematics_from_directory(
    kinematics_dir: Path,
    eye_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load eye kinematics and extract timestamps and horizontal angular velocity.

    Args:
        kinematics_dir: Directory containing {eye_name}_kinematics.csv and
                        {eye_name}_reference_geometry.json
        eye_name: Either 'left_eye' or 'right_eye'

    Returns:
        Tuple of (timestamps, eye_horizontal_velocity_deg_s)
    """
    if eye_name == "left_eye":
        kinematics_csv_name = LEFT_EYE_KINEMATICS_CSV
    elif eye_name == "right_eye":
        kinematics_csv_name = RIGHT_EYE_KINEMATICS_CSV
    else:
        raise ValueError(f"eye_name must be 'left_eye' or 'right_eye', got '{eye_name}'")

    kinematics_csv = kinematics_dir / kinematics_csv_name
    if not kinematics_csv.exists():
        raise FileNotFoundError(f"Eye kinematics CSV not found: {kinematics_csv}")

    df = pl.read_csv(kinematics_csv)

    # Extract timestamps
    timestamps = (
        df.select(["frame", "timestamp_s"])
        .unique()
        .sort("frame")
        ["timestamp_s"]
        .to_numpy()
        .astype(np.float64)
    )

    # Extract angular velocity global
    # For eye kinematics, horizontal velocity corresponds to adduction/abduction
    # which is rotation around the Y-axis (global angular velocity y-component)
    angular_velocity_global_df = df.filter(pl.col("trajectory") == "angular_velocity_global")

    # The Y component of global angular velocity gives horizontal (adduction) velocity
    y_component_df = (
        angular_velocity_global_df
        .filter(pl.col("component") == "y")
        .sort("frame")
    )

    eye_horizontal_velocity_rad_s = y_component_df["value"].to_numpy().astype(np.float64)

    # Apply anatomical sign correction for consistent adduction-positive convention
    # Right eye: +Y rotation = adduction (toward nose) -> sign = +1
    # Left eye: +Y rotation = abduction (away from nose) -> sign = -1
    anatomical_sign = 1.0 if eye_name == "right_eye" else -1.0
    eye_horizontal_velocity_rad_s = anatomical_sign * eye_horizontal_velocity_rad_s

    eye_horizontal_velocity_deg_s = np.rad2deg(eye_horizontal_velocity_rad_s)

    return timestamps, eye_horizontal_velocity_deg_s


def plot_head_yaw_vs_eye_horizontal_velocity(
    head_yaw_velocity_deg_s: NDArray[np.float64],
    left_eye_horizontal_velocity_deg_s: NDArray[np.float64],
    right_eye_horizontal_velocity_deg_s: NDArray[np.float64],
    output_path: Path | None = None,
) -> None:
    """
    Create scatter plot with regression lines comparing head yaw to eye horizontal velocity.

    Args:
        head_yaw_velocity_deg_s: Head yaw angular velocity in deg/s
        left_eye_horizontal_velocity_deg_s: Left eye horizontal angular velocity in deg/s
        right_eye_horizontal_velocity_deg_s: Right eye horizontal angular velocity in deg/s
        output_path: If provided, save figure to this path instead of showing
    """
    plt.figure(figsize=(10, 6))

    sns.regplot(
        x=head_yaw_velocity_deg_s,
        y=left_eye_horizontal_velocity_deg_s,
        label="Left Eye",
        scatter_kws={"alpha": 0.5},
    )
    sns.regplot(
        x=head_yaw_velocity_deg_s,
        y=right_eye_horizontal_velocity_deg_s,
        label="Right Eye",
        scatter_kws={"alpha": 0.5},
        color="orange",
    )

    plt.xlabel("Head Yaw Velocity (degrees/s)")
    plt.ylabel("Eye Horizontal Velocity (degrees/s)")
    plt.title("Head Yaw vs Eye Horizontal Velocity")
    plt.legend()
    plt.grid(True)

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def run_analysis(
    analyzable_output_dir: Path,
    output_path: Path | None = None,
) -> None:
    """
    Load data and run the head yaw vs eye horizontal velocity analysis.

    Args:
        analyzable_output_dir: Path to the analyzable_output directory
        output_path: If provided, save figure to this path instead of showing
    """
    # Validate directory structure
    skull_dir = analyzable_output_dir / SKULL_KINEMATICS_SUBDIR
    left_eye_dir = analyzable_output_dir / LEFT_EYE_KINEMATICS_SUBDIR
    right_eye_dir = analyzable_output_dir / RIGHT_EYE_KINEMATICS_SUBDIR

    for dir_path, name in [
        (skull_dir, SKULL_KINEMATICS_SUBDIR),
        (left_eye_dir, LEFT_EYE_KINEMATICS_SUBDIR),
        (right_eye_dir, RIGHT_EYE_KINEMATICS_SUBDIR),
    ]:
        if not dir_path.exists():
            raise FileNotFoundError(
                f"{name} directory not found: {dir_path}\n"
                f"Make sure you've run the gaze pipeline first."
            )

    print(f"Loading skull kinematics from {skull_dir}...")
    skull_timestamps, head_yaw_velocity_deg_s = load_skull_kinematics_from_directory(skull_dir)
    print(f"  Loaded {len(skull_timestamps)} frames")

    print(f"Loading left eye kinematics from {left_eye_dir}...")
    left_eye_timestamps, left_eye_horizontal_velocity_deg_s = load_eye_kinematics_from_directory(
        kinematics_dir=left_eye_dir,
        eye_name="left_eye",
    )
    print(f"  Loaded {len(left_eye_timestamps)} frames")

    print(f"Loading right eye kinematics from {right_eye_dir}...")
    right_eye_timestamps, right_eye_horizontal_velocity_deg_s = load_eye_kinematics_from_directory(
        kinematics_dir=right_eye_dir,
        eye_name="right_eye",
    )
    print(f"  Loaded {len(right_eye_timestamps)} frames")

    # Verify timestamps match (they should since data was resampled together)
    if not np.allclose(skull_timestamps, left_eye_timestamps, rtol=1e-9):
        raise ValueError("Skull and left eye timestamps do not match!")
    if not np.allclose(skull_timestamps, right_eye_timestamps, rtol=1e-9):
        raise ValueError("Skull and right eye timestamps do not match!")

    print("\nCreating plot...")
    plot_head_yaw_vs_eye_horizontal_velocity(
        head_yaw_velocity_deg_s=head_yaw_velocity_deg_s,
        left_eye_horizontal_velocity_deg_s=left_eye_horizontal_velocity_deg_s,
        right_eye_horizontal_velocity_deg_s=right_eye_horizontal_velocity_deg_s,
        output_path=output_path / "head_yaw_vs_eye_horizontal_velocity.png",
    )


if __name__ == "__main__":
    """Main entry point for command-line usage."""
    recording = Path("")
    # Use command line args if provided, otherwise use defaults
    if len(sys.argv) >= 2:
        recording_folder = Path(sys.argv[1])
    else:
        recording_folder = recording
        print(f"Using default analyzable output directory: {recording_folder}")

    analyzable_output_dir = recording_folder / "analyzable_output"

    if not analyzable_output_dir.exists():
        print(f"Error: Directory does not exist: {analyzable_output_dir}")
        print("\nUsage: python analyze_head_yaw_vs_eye_velocity.py [analyzable_output_dir] [output_figure_path]")
        print("\nEdit recording at the bottom of this file to set your default path.")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path: Path | None = Path(sys.argv[2])
    else:
        output_path = None

    run_analysis(
        analyzable_output_dir=analyzable_output_dir,
        output_path=output_path,
    )
