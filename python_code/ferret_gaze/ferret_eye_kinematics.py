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

    timestamps: NDArray[np.float64]  # (N,) seconds
    # Left eye (eye0) angles in head-local frame
    left_eye_angle_x_rad: NDArray[np.float64]  # (N,) medial(-)/lateral(+)
    left_eye_angle_y_rad: NDArray[np.float64]  # (N,) superior(+)/inferior(-)
    # Right eye (eye1) angles in head-local frame
    right_eye_angle_x_rad: NDArray[np.float64]  # (N,) medial(-)/lateral(+)
    right_eye_angle_y_rad: NDArray[np.float64]  # (N,) superior(+)/inferior(-)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert EyeKinematics to a pandas DataFrame."""
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "left_eye_angle_x_rad": self.left_eye_angle_x_rad,
            "left_eye_angle_y_rad": self.left_eye_angle_y_rad,
            "right_eye_angle_x_rad": self.right_eye_angle_x_rad,
            "right_eye_angle_y_rad": self.right_eye_angle_y_rad,
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

    Args:
        csv_path: Path to eye_data.csv
        processing_level: "cleaned" or "raw"

    Returns:
        EyeKinematics with eye angles for both eyes in radians
    """
    df = pd.read_csv(csv_path)

    # Filter by processing level
    df = df[df["processing_level"] == processing_level]

    # Get unique frames
    frames = sorted(df["frame"].unique())
    n_frames = len(frames)

    timestamps = np.zeros(n_frames, dtype=np.float64)
    left_eye_angle_x = np.zeros(n_frames, dtype=np.float64)
    left_eye_angle_y = np.zeros(n_frames, dtype=np.float64)
    right_eye_angle_x = np.zeros(n_frames, dtype=np.float64)
    right_eye_angle_y = np.zeros(n_frames, dtype=np.float64)

    for i, frame in enumerate(frames):
        frame_df = df[df["frame"] == frame]

        # Use eye0 (left eye) timestamp as primary
        eye0_df = frame_df[frame_df["video"] == "eye0"]
        if len(eye0_df) > 0:
            timestamps[i] = eye0_df["timestamp"].iloc[0]

        # Process each eye
        for video, angle_x_arr, angle_y_arr in [
            ("eye0", left_eye_angle_x, left_eye_angle_y),
            ("eye1", right_eye_angle_x, right_eye_angle_y),
        ]:
            video_df = frame_df[frame_df["video"] == video]
            if len(video_df) == 0:
                continue

            # Get pupil keypoints
            pupil_df = video_df[video_df["keypoint"].isin(PUPIL_KEYPOINTS)]
            if len(pupil_df) < len(PUPIL_KEYPOINTS):
                continue

            # Compute pupil center
            pupil_x, pupil_y = compute_pupil_center(pupil_df)

            # Get tear_duct and outer_eye for calibration
            tear_duct_df = video_df[video_df["keypoint"] == "tear_duct"]
            outer_eye_df = video_df[video_df["keypoint"] == "outer_eye"]

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

            # Convert pupil position to radians
            angle_x_arr[i] = pupil_x * pixels_to_radians
            angle_y_arr[i] = pupil_y * pixels_to_radians

    return EyeKinematics(
        timestamps=timestamps,
        left_eye_angle_x_rad=left_eye_angle_x,
        left_eye_angle_y_rad=left_eye_angle_y,
        right_eye_angle_x_rad=right_eye_angle_x,
        right_eye_angle_y_rad=right_eye_angle_y,
    )


if __name__ == "__main__":
    # Example usage - edit path as needed
    eye_data_csv = Path(r"D:\bs\ferret_recordings\example\eye_data.csv")

    print(f"Loading eye data from {eye_data_csv}...")
    ek = load_eye_data(eye_data_csv)
    print(f"  Loaded {len(ek.timestamps)} frames")

    # Save CSV
    output_path = eye_data_csv.parent / "eye_kinematics.csv"
    df = ek.to_dataframe()
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
