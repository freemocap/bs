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

    # Left eye (eye0)
    left_eye_timestamps: NDArray[np.float64]  # (N_left,) seconds
    left_eye_angle_x_rad: NDArray[np.float64]  # (N_left,) medial(-)/lateral(+)
    left_eye_angle_y_rad: NDArray[np.float64]  # (N_left,) superior(+)/inferior(-)
    # Right eye (eye1)
    right_eye_timestamps: NDArray[np.float64]  # (N_right,) seconds
    right_eye_angle_x_rad: NDArray[np.float64]  # (N_right,) medial(-)/lateral(+)
    right_eye_angle_y_rad: NDArray[np.float64]  # (N_right,) superior(+)/inferior(-)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert EyeKinematics to a pandas DataFrame.

        Returns separate dataframes for each eye since they may have different lengths.
        """
        left_df = pd.DataFrame({
            "timestamp": self.left_eye_timestamps,
            "eye": "left",
            "angle_x_rad": self.left_eye_angle_x_rad,
            "angle_y_rad": self.left_eye_angle_y_rad,
        })
        right_df = pd.DataFrame({
            "timestamp": self.right_eye_timestamps,
            "eye": "right",
            "angle_x_rad": self.right_eye_angle_x_rad,
            "angle_y_rad": self.right_eye_angle_y_rad,
        })
        return pd.concat([left_df, right_df], ignore_index=True)


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
    eye0 and eye1 cameras may have slightly different frame rates.

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
    left_timestamps: list[float] = []
    left_angle_x: list[float] = []
    left_angle_y: list[float] = []

    right_timestamps: list[float] = []
    right_angle_x: list[float] = []
    right_angle_y: list[float] = []

    for video, timestamps_list, angle_x_list, angle_y_list in [
        ("eye0", left_timestamps, left_angle_x, left_angle_y),
        ("eye1", right_timestamps, right_angle_x, right_angle_y),
    ]:
        video_df = df[df["video"] == video]
        frames = sorted(video_df["frame"].unique())

        for frame in frames:
            frame_df = video_df[video_df["frame"] == frame]

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

            pixels_to_radians = 1.0 / eye_width_pixels

            # Store results
            timestamps_list.append(timestamp)
            angle_x_list.append(pupil_x * pixels_to_radians)
            angle_y_list.append(pupil_y * pixels_to_radians)

    # Convert to numpy arrays
    left_eye_timestamps = np.array(left_timestamps, dtype=np.float64)
    left_eye_angle_x = np.array(left_angle_x, dtype=np.float64)
    left_eye_angle_y = np.array(left_angle_y, dtype=np.float64)

    right_eye_timestamps = np.array(right_timestamps, dtype=np.float64)
    right_eye_angle_x = np.array(right_angle_x, dtype=np.float64)
    right_eye_angle_y = np.array(right_angle_y, dtype=np.float64)

    # Validate timestamps
    for eye_name, timestamps in [("left", left_eye_timestamps), ("right", right_eye_timestamps)]:
        if len(timestamps) == 0:
            raise ValueError(f"No valid frames for {eye_name} eye")

        if not np.all(np.diff(timestamps) > 0):
            non_increasing = np.where(np.diff(timestamps) <= 0)[0]
            raise ValueError(
                f"{eye_name.capitalize()} eye timestamps are not monotonically increasing "
                f"at indices: {non_increasing[:10].tolist()}"
            )

    return EyeKinematics(
        left_eye_timestamps=left_eye_timestamps,
        left_eye_angle_x_rad=left_eye_angle_x,
        left_eye_angle_y_rad=left_eye_angle_y,
        right_eye_timestamps=right_eye_timestamps,
        right_eye_angle_x_rad=right_eye_angle_x,
        right_eye_angle_y_rad=right_eye_angle_y,
    )


if __name__ == "__main__":
    # Example usage - edit path as needed
    eye_data_csv = Path(r"D:\bs\ferret_recordings\example\eye_data.csv")

    print(f"Loading eye data from {eye_data_csv}...")
    ek = load_eye_data(eye_data_csv)
    print(f"  Left eye: {len(ek.left_eye_timestamps)} frames")
    print(f"  Right eye: {len(ek.right_eye_timestamps)} frames")

    # Save CSV
    output_path = eye_data_csv.parent / "eye_kinematics.csv"
    df = ek.to_dataframe()
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")