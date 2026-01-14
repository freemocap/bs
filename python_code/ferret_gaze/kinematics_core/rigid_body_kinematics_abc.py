from abc import ABC, abstractmethod

import numpy as np
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
    ):

        pass
