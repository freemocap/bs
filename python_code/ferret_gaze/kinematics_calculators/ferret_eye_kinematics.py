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
class EyeballKinematics:
    """Eye-in-head orientation data for one eye."""

    timestamps: NDArray[np.float64]  # (N,) seconds
    eyeball_angle_azimuth_rad: NDArray[np.float64]  # (N,) medial(-)/lateral(+)
    eyeball_angle_elevation_rad: NDArray[np.float64]  # (N,) superior(+)/inferior(-)

    @property
    def eye_horizontal_velocity_rad_s(self) -> NDArray[np.float64]:
        """Compute horizontal eye velocity in radians/s.

        Returns array of length N (same as timestamps) by padding first frame
        with second frame's value, matching skull kinematics convention.
        """
        if len(self.timestamps) < 2:
            raise ValueError("Need at least 2 frames to compute velocity")
        dt = np.diff(self.timestamps)
        if np.any(dt <= 0):
            raise ValueError("Timestamps must be strictly increasing for velocity computation")
        dx = np.diff(self.eyeball_angle_azimuth_rad)
        velocity = dx / dt
        # Pad first frame with second frame's value (same convention as skull kinematics)
        return np.concatenate([[velocity[0]], velocity])

    def __post_init__(self) -> None:
        """Validate array shapes match."""
        n_frames = len(self.timestamps)
        if self.timestamps.ndim != 1:
            raise ValueError(f"timestamps must be 1D, got shape {self.timestamps.shape}")
        if self.eyeball_angle_azimuth_rad.shape != (n_frames,):
            raise ValueError(
                f"eye_angle_x_rad shape {self.eyeball_angle_azimuth_rad.shape} must match "
                f"timestamps length ({n_frames},)"
            )
        if self.eyeball_angle_elevation_rad.shape != (n_frames,):
            raise ValueError(
                f"eye_angle_y_rad shape {self.eyeball_angle_elevation_rad.shape} must match "
                f"timestamps length ({n_frames},)"
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert EyeKinematics to a pandas DataFrame."""
        frame_number = np.arange(len(self.timestamps))
        return pd.DataFrame({
            "frame": frame_number,
            "timestamp": self.timestamps,
            "angle_x_rad": self.eyeball_angle_azimuth_rad,
            "angle_y_rad": self.eyeball_angle_elevation_rad,
        })


def compute_eye_width_pixels(
    tear_duct_x: float,
    tear_duct_y: float,
    outer_eye_x: float,
    outer_eye_y: float,
) -> float:
    """Compute eye width in pixels from tear duct to outer eye."""
    return float(np.sqrt((outer_eye_x - tear_duct_x) ** 2 + (outer_eye_y - tear_duct_y) ** 2))


def compute_pupil_center(pupil_df: pd.DataFrame) -> tuple[float, float]:
    """Compute pupil center as centroid of p1-p8 keypoints."""
    if len(pupil_df) == 0:
        raise ValueError("Cannot compute pupil center from empty DataFrame")
    return float(pupil_df["x"].mean()), float(pupil_df["y"].mean())


PUPIL_PIXEL_TO_EYE_RADIAN_SCALE_FACTOR=1 # Need to figure this number
def load_eye_data(
    csv_path: Path,
    processing_level: str = "cleaned",
) -> EyeballKinematics:
    """Load eye tracking data from CSV and compute eye angles.

    Args:
        csv_path: Path to eye_data.csv
        processing_level: "cleaned" or "raw"

    Returns:
        EyeKinematics with eye angles in radians

    Raises:
        FileNotFoundError: If csv_path doesn't exist
        ValueError: If no valid frames found or timestamps not monotonic
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Eye data CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter by processing level
    df = df[df["processing_level"] == processing_level]
    if len(df) == 0:
        raise ValueError(f"No data with processing_level='{processing_level}' in {csv_path}")

    # Process each frame
    timestamps: list[float] = []
    eye_angle_x: list[float] = []
    eye_angle_y: list[float] = []

    frames = sorted(df["frame"].unique())

    for frame in frames:
        frame_df = df[df["frame"] == frame]

        # Get timestamp for this frame
        timestamp = float(frame_df["timestamp"].iloc[0])

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
            tear_duct_x=float(tear_duct_df["x"].iloc[0]),
            tear_duct_y=float(tear_duct_df["y"].iloc[0]),
            outer_eye_x=float(outer_eye_df["x"].iloc[0]),
            outer_eye_y=float(outer_eye_df["y"].iloc[0]),
        )

        if eye_width_pixels < 1e-6:
            raise ValueError(f"Eye width too small at frame {frame}: {eye_width_pixels}")

        # TODO: This conversion factor is approximate - needs calibration
        pixels_to_radians = (1.0 / eye_width_pixels) * PUPIL_PIXEL_TO_EYE_RADIAN_SCALE_FACTOR

        # Store results
        timestamps.append(timestamp)
        eye_angle_x.append(pupil_x * pixels_to_radians)
        eye_angle_y.append(pupil_y * pixels_to_radians)

    # Validate we have data
    if len(timestamps) == 0:
        raise ValueError(f"No valid frames for eye data in {csv_path}")

    # Convert to numpy arrays
    timestamps_arr = np.asarray(timestamps, dtype=np.float64)
    eye_angle_x_arr = np.asarray(eye_angle_x, dtype=np.float64)
    eye_angle_y_arr = np.asarray(eye_angle_y, dtype=np.float64)

    # Validate timestamps are monotonically increasing
    timestamp_diffs = np.diff(timestamps_arr)
    if not np.all(timestamp_diffs > 0):
        non_increasing = np.where(timestamp_diffs <= 0)[0]
        raise ValueError(
            f"Eye timestamps are not monotonically increasing "
            f"at indices: {non_increasing[:10].tolist()} for {csv_path}"
        )

    return EyeballKinematics(
        timestamps=timestamps_arr,
        eyeball_angle_azimuth_rad=eye_angle_x_arr,
        eyeball_angle_elevation_rad=eye_angle_y_arr,
    )


if __name__ == "__main__":
    # Example usage - edit path as needed
    left_eye_data_csv = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye0_data.csv"
    )
    right_eye_data_csv = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\output_data\eye1_data.csv"
    )

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