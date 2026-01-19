"""
Ferret Gaze Kinematics Analysis

Combines skull kinematics and eye kinematics to compute gaze vectors in world frame.

Coordinate conventions:
- Left eye at rest (0, 0) points +Y in skull frame
- Right eye at rest (0, 0) points -Y in skull frame
- Eye angle X: medial(-) / lateral(+) -> maps to skull X axis (toward/away from nose)
- Eye angle Y: superior(+) / inferior(-) -> maps to skull Z axis (up/down)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_code.ferret_gaze.kinematics_calculators.ferret_eye_kinematics import EyeballKinematics
from python_code.ferret_gaze.kinematics_calculators.ferret_skull_kinematics import SkullKinematics

GAZE_VECTOR_LENGTH_MM: float = 100.0  # 10 cm


@dataclass
class GazeKinematics:
    """Gaze kinematics data for both eyes in world frame."""

    timestamps: NDArray[np.float64]  # (N,) seconds
    # Left eye
    left_eyeball_center_xyz_mm: NDArray[np.float64]  # (N, 3) eyeball center in world frame
    left_gaze_azimuth_rad: NDArray[np.float64]  # (N,) azimuth in world frame
    left_gaze_elevation_rad: NDArray[np.float64]  # (N,) elevation in world frame
    left_gaze_endpoint_mm: NDArray[np.float64]  # (N, 3) gaze vector endpoint in world frame
    left_gaze_direction: NDArray[np.float64]  # (N, 3) unit gaze direction in world frame
    # Right eye
    right_eyeball_center_xyz_mm: NDArray[np.float64]  # (N, 3) eyeball center in world frame
    right_gaze_azimuth_rad: NDArray[np.float64]  # (N,) azimuth in world frame
    right_gaze_elevation_rad: NDArray[np.float64]  # (N,) elevation in world frame
    right_gaze_endpoint_mm: NDArray[np.float64]  # (N, 3) gaze vector endpoint in world frame
    right_gaze_direction: NDArray[np.float64]  # (N, 3) unit gaze direction in world frame

    def __post_init__(self) -> None:
        """Validate all array shapes are consistent."""
        n_frames = len(self.timestamps)

        if self.timestamps.ndim != 1:
            raise ValueError(f"timestamps must be 1D, got shape {self.timestamps.shape}")

        # Left eye validations
        if self.left_eyeball_center_xyz_mm.shape != (n_frames, 3):
            raise ValueError(
                f"left_eyeball_center_xyz_mm shape {self.left_eyeball_center_xyz_mm.shape} "
                f"must be ({n_frames}, 3)"
            )
        if self.left_gaze_azimuth_rad.shape != (n_frames,):
            raise ValueError(
                f"left_gaze_azimuth_rad shape {self.left_gaze_azimuth_rad.shape} must be ({n_frames},)"
            )
        if self.left_gaze_elevation_rad.shape != (n_frames,):
            raise ValueError(
                f"left_gaze_elevation_rad shape {self.left_gaze_elevation_rad.shape} must be ({n_frames},)"
            )
        if self.left_gaze_endpoint_mm.shape != (n_frames, 3):
            raise ValueError(
                f"left_gaze_endpoint_mm shape {self.left_gaze_endpoint_mm.shape} must be ({n_frames}, 3)"
            )
        if self.left_gaze_direction.shape != (n_frames, 3):
            raise ValueError(
                f"left_gaze_direction shape {self.left_gaze_direction.shape} must be ({n_frames}, 3)"
            )

        # Right eye validations
        if self.right_eyeball_center_xyz_mm.shape != (n_frames, 3):
            raise ValueError(
                f"right_eyeball_center_xyz_mm shape {self.right_eyeball_center_xyz_mm.shape} "
                f"must be ({n_frames}, 3)"
            )
        if self.right_gaze_azimuth_rad.shape != (n_frames,):
            raise ValueError(
                f"right_gaze_azimuth_rad shape {self.right_gaze_azimuth_rad.shape} must be ({n_frames},)"
            )
        if self.right_gaze_elevation_rad.shape != (n_frames,):
            raise ValueError(
                f"right_gaze_elevation_rad shape {self.right_gaze_elevation_rad.shape} must be ({n_frames},)"
            )
        if self.right_gaze_endpoint_mm.shape != (n_frames, 3):
            raise ValueError(
                f"right_gaze_endpoint_mm shape {self.right_gaze_endpoint_mm.shape} must be ({n_frames}, 3)"
            )
        if self.right_gaze_direction.shape != (n_frames, 3):
            raise ValueError(
                f"right_gaze_direction shape {self.right_gaze_direction.shape} must be ({n_frames}, 3)"
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert GazeKinematics to a pandas DataFrame."""
        frame_numbers = np.arange(len(self.timestamps))
        return pd.DataFrame({
            "frame": frame_numbers,
            "timestamp": self.timestamps,
            # Left eye
            "left_eyeball_center_x_mm": self.left_eyeball_center_xyz_mm[:, 0],
            "left_eyeball_center_y_mm": self.left_eyeball_center_xyz_mm[:, 1],
            "left_eyeball_center_z_mm": self.left_eyeball_center_xyz_mm[:, 2],
            "left_gaze_azimuth_rad": self.left_gaze_azimuth_rad,
            "left_gaze_elevation_rad": self.left_gaze_elevation_rad,
            "left_gaze_endpoint_x_mm": self.left_gaze_endpoint_mm[:, 0],
            "left_gaze_endpoint_y_mm": self.left_gaze_endpoint_mm[:, 1],
            "left_gaze_endpoint_z_mm": self.left_gaze_endpoint_mm[:, 2],
            "left_gaze_direction_x": self.left_gaze_direction[:, 0],
            "left_gaze_direction_y": self.left_gaze_direction[:, 1],
            "left_gaze_direction_z": self.left_gaze_direction[:, 2],
            # Right eye
            "right_eye_center_x_mm": self.right_eyeball_center_xyz_mm[:, 0],
            "right_eye_center_y_mm": self.right_eyeball_center_xyz_mm[:, 1],
            "right_eye_center_z_mm": self.right_eyeball_center_xyz_mm[:, 2],
            "right_gaze_azimuth_rad": self.right_gaze_azimuth_rad,
            "right_gaze_elevation_rad": self.right_gaze_elevation_rad,
            "right_gaze_endpoint_x_mm": self.right_gaze_endpoint_mm[:, 0],
            "right_gaze_endpoint_y_mm": self.right_gaze_endpoint_mm[:, 1],
            "right_gaze_endpoint_z_mm": self.right_gaze_endpoint_mm[:, 2],
            "right_gaze_direction_x": self.right_gaze_direction[:, 0],
            "right_gaze_direction_y": self.right_gaze_direction[:, 1],
            "right_gaze_direction_z": self.right_gaze_direction[:, 2],
        })


def compute_gaze_direction_skull_frame(
    eyeball_angle_azimuth_rad: float,
    eyeball_angle_elevation_rad: float,
    is_left_eye: bool,
) -> NDArray[np.float64]:
    """Compute gaze direction in skull frame from eye angles.

    Args:
        eyeball_angle_azimuth_rad: Medial(-)/lateral(+) angle in radians
        eyeball_angle_elevation_rad: Superior(+)/inferior(-) angle in radians
        is_left_eye: True for left eye (+Y rest), False for right eye (-Y rest)

    Returns:
        Unit gaze direction vector in skull frame [x, y, z]
    """
    # Eye angle X: medial(-) = toward nose (+X_skull), lateral(+) = away from nose (-X_skull)
    # So skull_x_component = -eye_angle_x
    # Eye angle Y: superior(+) = up (+Z_skull), inferior(-) = down (-Z_skull)
    # So skull_z_component = eye_angle_y

    # NOTE: This calculation may need refinement based on actual eye model
    skull_x = -eyeball_angle_azimuth_rad
    skull_z = eyeball_angle_elevation_rad

    if is_left_eye:
        # Left eye at rest points +Y
        gaze_skull = np.array([skull_x, 1.0, skull_z], dtype=np.float64)
    else:
        # Right eye at rest points -Y
        gaze_skull = np.array([skull_x, -1.0, skull_z], dtype=np.float64)

    # Normalize to unit vector
    norm = float(np.linalg.norm(gaze_skull))
    if norm < 1e-10:
        raise ValueError("Gaze direction is zero - check input angles")
    return gaze_skull / norm


def transform_to_world_frame(
    vector_skull: NDArray[np.float64],
    basis_x: NDArray[np.float64],
    basis_y: NDArray[np.float64],
    basis_z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform a vector from skull frame to world frame.

    Args:
        vector_skull: Vector in skull frame [x, y, z], shape (3,)
        basis_x: Skull X axis in world frame (unit vector), shape (3,)
        basis_y: Skull Y axis in world frame (unit vector), shape (3,)
        basis_z: Skull Z axis in world frame (unit vector), shape (3,)

    Returns:
        Vector in world frame [x, y, z], shape (3,)
    """
    if vector_skull.shape != (3,):
        raise ValueError(f"vector_skull must have shape (3,), got {vector_skull.shape}")
    if basis_x.shape != (3,):
        raise ValueError(f"basis_x must have shape (3,), got {basis_x.shape}")
    if basis_y.shape != (3,):
        raise ValueError(f"basis_y must have shape (3,), got {basis_y.shape}")
    if basis_z.shape != (3,):
        raise ValueError(f"basis_z must have shape (3,), got {basis_z.shape}")

    return (
        vector_skull[0] * basis_x +
        vector_skull[1] * basis_y +
        vector_skull[2] * basis_z
    )


def compute_azimuth_elevation(direction: NDArray[np.float64]) -> tuple[float, float]:
    """Compute azimuth and elevation from a direction vector.

    Args:
        direction: Unit direction vector in world frame [x, y, z], shape (3,)

    Returns:
        (azimuth_rad, elevation_rad) where:
        - azimuth: angle in XY plane from +X axis (positive toward +Y)
        - elevation: angle above XY plane (positive toward +Z)
    """
    if direction.shape != (3,):
        raise ValueError(f"direction must have shape (3,), got {direction.shape}")
    azimuth = np.arctan2(direction[1], direction[0])
    elevation = np.arcsin(np.clip(direction[2], -1.0, 1.0))
    return float(azimuth), float(elevation)


def compute_gaze_kinematics(
    skull: SkullKinematics,
    left_eye: EyeballKinematics,
    right_eye: EyeballKinematics,
    trajectory_data: dict[str, NDArray[np.float64]],
) -> GazeKinematics:
    """Compute gaze kinematics from skull and eye data.

    Args:
        skull: Skull kinematics (orientation and basis vectors)
        left_eye: Eye kinematics (eye-in-skull angles) for left eye
        right_eye: Eye kinematics (eye-in-skull angles) for right eye
        trajectory_data: Trajectory data for head and body markers
            Must contain "left_eyeball_center" and "right_eyeball_center" keys

    Returns:
        GazeKinematics

    Raises:
        ValueError: If array lengths don't match or required keys missing
    """
    n_frames = len(skull.timestamps)

    # Validate all inputs have the same length
    if len(left_eye.timestamps) != n_frames:
        raise ValueError(
            f"left_eye has {len(left_eye.timestamps)} frames, skull has {n_frames} frames. "
            f"Resample to common timestamps first."
        )
    if len(right_eye.timestamps) != n_frames:
        raise ValueError(
            f"right_eye has {len(right_eye.timestamps)} frames, skull has {n_frames} frames. "
            f"Resample to common timestamps first."
        )

    # Extract eye centers from trajectory - these must exist
    left_eyeball_center = trajectory_data.get("left_eye")
    right_eyeball_center = trajectory_data.get("right_eye")

    if left_eyeball_center is None:
        raise ValueError("trajectory_data must contain 'left_eye' key")
    if right_eyeball_center is None:
        raise ValueError("trajectory_data must contain 'right_eye' key")

    # Validate eye center shapes
    if left_eyeball_center.shape != (n_frames, 3):
        raise ValueError(
            f"left_eyeball_center shape {left_eyeball_center.shape} must be ({n_frames}, 3)"
        )
    if right_eyeball_center.shape != (n_frames, 3):
        raise ValueError(
            f"right_eyeball_center shape {right_eyeball_center.shape} must be ({n_frames}, 3)"
        )

    # Allocate output arrays
    left_gaze_azimuth = np.zeros(n_frames, dtype=np.float64)
    left_gaze_elevation = np.zeros(n_frames, dtype=np.float64)
    left_gaze_endpoint = np.zeros((n_frames, 3), dtype=np.float64)
    left_gaze_direction = np.zeros((n_frames, 3), dtype=np.float64)

    right_gaze_azimuth = np.zeros(n_frames, dtype=np.float64)
    right_gaze_elevation = np.zeros(n_frames, dtype=np.float64)
    right_gaze_endpoint = np.zeros((n_frames, 3), dtype=np.float64)
    right_gaze_direction = np.zeros((n_frames, 3), dtype=np.float64)

    for i in range(n_frames):
        # Get skull basis vectors for this frame
        basis_x = skull.basis_x[i]
        basis_y = skull.basis_y[i]
        basis_z = skull.basis_z[i]

        # Left eye
        gaze_skull_left = compute_gaze_direction_skull_frame(
            eyeball_angle_azimuth_rad=float(left_eye.eyeball_angle_azimuth_rad[i]),
            eyeball_angle_elevation_rad=float(left_eye.eyeball_angle_elevation_rad[i]),
            is_left_eye=True,
        )
        gaze_world_left = transform_to_world_frame(gaze_skull_left, basis_x, basis_y, basis_z)
        left_gaze_direction[i] = gaze_world_left
        left_gaze_azimuth[i], left_gaze_elevation[i] = compute_azimuth_elevation(gaze_world_left)
        left_gaze_endpoint[i] = left_eyeball_center[i] + gaze_world_left * GAZE_VECTOR_LENGTH_MM

        # Right eye
        gaze_skull_right = compute_gaze_direction_skull_frame(
            eyeball_angle_azimuth_rad=float(right_eye.eyeball_angle_azimuth_rad[i]),
            eyeball_angle_elevation_rad=float(right_eye.eyeball_angle_elevation_rad[i]),
            is_left_eye=False,
        )
        gaze_world_right = transform_to_world_frame(gaze_skull_right, basis_x, basis_y, basis_z)
        right_gaze_direction[i] = gaze_world_right
        right_gaze_azimuth[i], right_gaze_elevation[i] = compute_azimuth_elevation(gaze_world_right)
        right_gaze_endpoint[i] = right_eyeball_center[i] + gaze_world_right * GAZE_VECTOR_LENGTH_MM

    return GazeKinematics(
        timestamps=skull.timestamps.copy(),
        left_eyeball_center_xyz_mm=left_eyeball_center.copy(),
        left_gaze_azimuth_rad=left_gaze_azimuth,
        left_gaze_elevation_rad=left_gaze_elevation,
        left_gaze_endpoint_mm=left_gaze_endpoint,
        left_gaze_direction=left_gaze_direction,
        right_eyeball_center_xyz_mm=right_eyeball_center.copy(),
        right_gaze_azimuth_rad=right_gaze_azimuth,
        right_gaze_elevation_rad=right_gaze_elevation,
        right_gaze_endpoint_mm=right_gaze_endpoint,
        right_gaze_direction=right_gaze_direction,
    )