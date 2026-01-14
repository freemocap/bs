"""Abstract base class for rigid body kinematics with disk persistence."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from python_code.ferret_gaze.kinematics_core.quaternion_helper import Quaternion


class RigidBodyKinematicsABC(BaseModel, ABC):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    timestamps: NDArray[np.float64]  # (N,) seconds
    position_xyz_mm: NDArray[np.float64]  # (N, 3) position of rigid body origin in mm
    orientation_quaternions: list[Quaternion]  # (N,) list of quaternions representing orientation of the rigid body at each timestamp

    basis_vector_x_xyz_mm: NDArray[np.float64]  # (N, 3) x-axis basis vector in mm
    basis_vector_y_xyz_mm: NDArray[np.float64]  # (N, 3) y-axis basis vector in mm
    basis_vector_z_xyz_mm: NDArray[np.float64]  # (N, 3) z-axis basis vector in mm

    keypoint_trajectories: dict[str, NDArray[np.float64]]  # Dict of keypoint name to (N, 3) position arrays in mm, calculated by applying rigid body pose to the reference geometry keypoints. Keys must match those in reference_geometry.
    reference_geometry: dict[str, NDArray[np.float64]]  # Dict of keypoint name to (3,) position arrays in mm, defining the keypoint positions in the rigid body's reference frame (keys must match those in keypoint_trajectories)

    global_angular_velocity_pitch_rad_s: NDArray[np.float64]  # (N,) angular velocity around y-axis in radians/s
    global_angular_velocity_roll_rad_s: NDArray[np.float64]  # (N,) angular velocity around x-axis in radians/s
    global_angular_velocity_yaw_rad_s: NDArray[np.float64]  # (N,) angular velocity around z-axis in radians/s

    local_angular_velocity_pitch_rad_s: NDArray[np.float64]  # (N,) angular velocity around y-axis in radians/s, expressed in the rigid body's local frame
    local_angular_velocity_roll_rad_s: NDArray[np.float64]  # (N,) angular velocity around x-axis in radians/s, expressed in the rigid body's local frame
    local_angular_velocity_yaw_rad_s: NDArray[np.float64]  # (N,) angular velocity around z-axis in radians/s, expressed in the rigid body's local frame

    linear_velocity_xyz_mm_s: NDArray[np.float64]  # (N, 3) linear velocity of the rigid body's origin in mm/s (first frame padded with second frame's value)

    origin_markers: list[str]  # List of marker names that define the origin of the rigid body in the reference geometry. Must be a subset of the keys in reference_geometry.
    x_reference_markers: list[str]  # List of marker names that define the x-axis direction in the reference geometry. Must be a subset of the keys in reference_geometry. Will be exactly the same direction as the X-basis vector in the rigid body's local frame.
    y_ward_markers: list[str]  # List of marker names that define the y-axis direction in the reference geometry. Must be a subset of the keys in reference_geometry. Will be approximately the same direction as the Y-basis vector in the rigid body's local frame (not exact because of orthonormalization).

    @classmethod
    @abstractmethod
    def from_data(
        cls,
        timestamps: NDArray[np.float64],
        position_xyz_mm: NDArray[np.float64],
        orientation_quaternions: NDArray[np.float64],
        reference_geometry: dict[str, NDArray[np.float64]],
        origin_markers: list[str],
        x_reference_markers: list[str],
        y_ward_markers: list[str],
    ) -> "RigidBodyKinematicsABC":
        pass

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def keypoint_names(self) -> list[str]:
        return sorted(self.reference_geometry.keys())

    def save_to_disk(self, save_directory: Path, name: str) -> None:
        """
        Save kinematics data to disk.

        Creates two files:
            {name}_reference_geometry.json - Static reference geometry and marker configuration
            {name}_kinematics.csv - All time-varying kinematic data (one row per frame)

        Args:
            save_directory: Directory to save files to (will be created if it doesn't exist)
            name: Base name for the output files
        """
        save_directory.mkdir(parents=True, exist_ok=True)

        self._save_reference_geometry_json(
            filepath=save_directory / f"{name}_reference_geometry.json",
        )
        self._save_kinematics_csv(
            filepath=save_directory / f"{name}_kinematics.csv",
        )

    def _save_reference_geometry_json(self, filepath: Path) -> None:
        """Save static reference geometry and marker configuration to JSON."""
        data = {
            "reference_geometry": {
                keypoint_name: position.tolist()
                for keypoint_name, position in self.reference_geometry.items()
            },
            "origin_markers": self.origin_markers,
            "x_reference_markers": self.x_reference_markers,
            "y_ward_markers": self.y_ward_markers,
        }

        with open(filepath, "w") as f:
            json.dump(obj=data, fp=f, indent=2)

    def _save_kinematics_csv(self, filepath: Path) -> None:
        """
        Save all time-varying kinematic data to a tidy-format CSV file.

        Columns:
            frame: int — frame index
            timestamp_s: float — timestamp in seconds
            trajectory: str — what is being measured (e.g. position, orientation, angular_velocity, keypoint__{name})
            component: str — vector/quaternion component (x, y, z, w, roll, pitch, yaw)
            value: float — the measurement value
            unit: str — unit of measurement (mm, mm_s, rad_s, quaternion)
            reference_frame: str — coordinate frame (world or body)
        """
        rows: list[dict[str, int | float | str]] = []

        for frame_idx in range(self.n_frames):
            timestamp = self.timestamps[frame_idx]

            # Position (world frame)
            for component, value in [
                ("x", self.position_xyz_mm[frame_idx, 0]),
                ("y", self.position_xyz_mm[frame_idx, 1]),
                ("z", self.position_xyz_mm[frame_idx, 2]),
            ]:
                rows.append({
                    "frame": frame_idx,
                    "timestamp_s": timestamp,
                    "trajectory": "position",
                    "component": component,
                    "value": value,
                    "unit": "mm",
                    "reference_frame": "world",
                })

            # Orientation quaternion (world frame, represents world→body rotation)
            q = self.orientation_quaternions[frame_idx]
            for component, value in [("w", q.w), ("x", q.x), ("y", q.y), ("z", q.z)]:
                rows.append({
                    "frame": frame_idx,
                    "timestamp_s": timestamp,
                    "trajectory": "orientation",
                    "component": component,
                    "value": value,
                    "unit": "quaternion",
                    "reference_frame": "world",
                })

            # Basis vectors (world frame)
            for basis_name, basis_data in [
                ("basis_x", self.basis_vector_x_xyz_mm),
                ("basis_y", self.basis_vector_y_xyz_mm),
                ("basis_z", self.basis_vector_z_xyz_mm),
            ]:
                for comp_idx, component in enumerate(["x", "y", "z"]):
                    rows.append({
                        "frame": frame_idx,
                        "timestamp_s": timestamp,
                        "trajectory": basis_name,
                        "component": component,
                        "value": basis_data[frame_idx, comp_idx],
                        "unit": "mm",
                        "reference_frame": "world",
                    })

            # Linear velocity (world frame)
            for comp_idx, component in enumerate(["x", "y", "z"]):
                rows.append({
                    "frame": frame_idx,
                    "timestamp_s": timestamp,
                    "trajectory": "linear_velocity",
                    "component": component,
                    "value": self.linear_velocity_xyz_mm_s[frame_idx, comp_idx],
                    "unit": "mm_s",
                    "reference_frame": "world",
                })

            # Angular velocity (world frame)
            for component, value in [
                ("roll", self.global_angular_velocity_roll_rad_s[frame_idx]),
                ("pitch", self.global_angular_velocity_pitch_rad_s[frame_idx]),
                ("yaw", self.global_angular_velocity_yaw_rad_s[frame_idx]),
            ]:
                rows.append({
                    "frame": frame_idx,
                    "timestamp_s": timestamp,
                    "trajectory": "angular_velocity",
                    "component": component,
                    "value": value,
                    "unit": "rad_s",
                    "reference_frame": "world",
                })

            # Angular velocity (body frame)
            for component, value in [
                ("roll", self.local_angular_velocity_roll_rad_s[frame_idx]),
                ("pitch", self.local_angular_velocity_pitch_rad_s[frame_idx]),
                ("yaw", self.local_angular_velocity_yaw_rad_s[frame_idx]),
            ]:
                rows.append({
                    "frame": frame_idx,
                    "timestamp_s": timestamp,
                    "trajectory": "angular_velocity",
                    "component": component,
                    "value": value,
                    "unit": "rad_s",
                    "reference_frame": "body",
                })

            # Keypoint trajectories (world frame)
            for keypoint_name in self.keypoint_names:
                trajectory = self.keypoint_trajectories[keypoint_name]
                for comp_idx, component in enumerate(["x", "y", "z"]):
                    rows.append({
                        "frame": frame_idx,
                        "timestamp_s": timestamp,
                        "trajectory": f"keypoint__{keypoint_name}",
                        "component": component,
                        "value": trajectory[frame_idx, comp_idx],
                        "unit": "mm",
                        "reference_frame": "world",
                    })

        df = pd.DataFrame(data=rows)
        df.to_csv(path_or_buf=filepath, index=False)