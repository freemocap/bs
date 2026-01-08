"""
Ferret Eye Kinematics Analysis

Loads eye tracking data and computes eye-in-head (socket-relative) orientation.
Eye angles represent gaze direction relative to the skull reference frame.

Convention:
- X angle: medial(-) / lateral(+) in radians
- Y angle: superior(+) / inferior(-) in radians
- Torsion is assumed zero
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


PUPIL_KEYPOINTS: list[str] = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]


@dataclass
class EyeKinematics:
    """Eye-in-head orientation data for both eyes."""

    timestamps: NDArray[np.float64]  # (N_left,) seconds
    eye_angle_x_rad: NDArray[np.float64]  # (N_left,) medial(-)/lateral(+)
    eye_angle_y_rad: NDArray[np.float64]  # (N_left,) superior(+)/inferior(-)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert EyeKinematics to a pandas DataFrame.

        Returns separate dataframes for each eye since they may have different lengths.
        """
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "angle_x_rad": self.eye_angle_x_rad,
            "angle_y_rad": self.eye_angle_y_rad,
        })


def compute_eye_width_pixels(
    tear_duct_x: float,
    tear_duct_y: float,
    outer_eye_x: float,
    outer_eye_y: float,
) -> float:
    """Compute eye width in pixels from tear duct to outer eye."""
    return np.sqrt((outer_eye_x - tear_duct_x) ** 2 + (outer_eye_y - tear_duct_y) ** 2)


def compute_pupil_center(pupil_df: pd.DataFrame) -> tuple[float, float]:
    """Compute pupil center as centroid of p1-p8 keypoints."""
    return float(pupil_df["x"].mean()), float(pupil_df["y"].mean())


def load_eye_data(
    csv_path: Path,
    processing_level: str = "cleaned",
) -> EyeKinematics:
    """Load eye tracking data from CSV and compute eye angles.

    Each eye is processed independently with its own timestamps since

    Args:
        csv_path: Path to eye_data.csv
        processing_level: "cleaned" or "raw"

    Returns:
        EyeKinematics with eye angles for both eyes in radians
    """
    df = pd.read_csv(csv_path)

    # Filter by processing level
    df = df[df["processing_level"] == processing_level]

    # Process each eye independently
    timestamps: list[float] = []
    eye_angle_x: list[float] = []
    eye_angle_y: list[float] = []


    frames = sorted(df["frame"].unique())

    for frame in frames:
        frame_df = df[df["frame"] == frame]

        # Get timestamp for this frame
        timestamp = frame_df["timestamp"].iloc[0]

        # Get pupil keypoints
        pupil_df = frame_df[frame_df["keypoint"].isin(PUPIL_KEYPOINTS)]
        if len(pupil_df) < len(PUPIL_KEYPOINTS):
            continue

        # Compute pupil center
        pupil_x, pupil_y = compute_pupil_center(pupil_df)

        # Get tear_duct and outer_eye for calibration
        tear_duct_df = frame_df[frame_df["keypoint"] == "tear_duct"]
        outer_eye_df = frame_df[frame_df["keypoint"] == "outer_eye"]

        if len(tear_duct_df) == 0 or len(outer_eye_df) == 0:
            continue

        # Compute eye width for pixels_to_radians conversion
        eye_width_pixels = compute_eye_width_pixels(
            tear_duct_x=tear_duct_df["x"].iloc[0],
            tear_duct_y=tear_duct_df["y"].iloc[0],
            outer_eye_x=outer_eye_df["x"].iloc[0],
            outer_eye_y=outer_eye_df["y"].iloc[0],
        )

        if eye_width_pixels < 1e-6:
            raise ValueError(f"Eye width too small: {eye_width_pixels}")

        pixels_to_radians = (1.0 / eye_width_pixels)* (2* np.pi ) # This is kinda made up nonsense - need to calculate this more carefully

        # Store results
        timestamps.append(timestamp)
        eye_angle_x.append(pupil_x * pixels_to_radians)
        eye_angle_y.append(pupil_y * pixels_to_radians)




    # Validate timestamps

    if len(timestamps) == 0:
        raise ValueError(f"No valid frames for eye data in {csv_path}")
    timestamp_diffs = np.diff(np.asarray(timestamps))
    if not np.all(timestamp_diffs > 0):
        non_increasing = np.where(timestamp_diffs <= 0)[0]
        raise ValueError(
            f"eye timestamps are not monotonically increasing "
            f"at indices: {non_increasing[:10].tolist()} for {csv_path}"
        )

    return EyeKinematics(
        timestamps=np.asarray(timestamps),
        eye_angle_x_rad=np.asarray(eye_angle_x),
        eye_angle_y_rad=np.asarray(eye_angle_y),
    )


if __name__ == "__main__":
    # Example usage - edit path as needed
    left_eye_data_csv = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye0_data.csv")
    right_eye_data_csv = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye1_data.csv")

    print(f"Loading eye data from {left_eye_data_csv}...")
    left_ek = load_eye_data(left_eye_data_csv)
    print(f"  Left eye: {len(left_ek.timestamps)} frames")
    print(f"Loading eye data from {right_eye_data_csv}...")
    right_ek = load_eye_data(right_eye_data_csv)
    print(f"  Right eye: {len(right_ek.timestamps)} frames")




    # Save each eye separately
    left_output_path = left_eye_data_csv.parent / "left_eye_kinematics.csv"
    right_output_path = right_eye_data_csv.parent / "right_eye_kinematics.csv"
    left_ek.to_dataframe().to_csv(left_output_path, index=False)
    right_ek.to_dataframe().to_csv(right_output_path, index=False)
    print(f"Saved: {left_output_path}")
    print(f"Saved: {right_output_path}")