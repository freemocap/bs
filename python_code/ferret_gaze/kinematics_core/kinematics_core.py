"""
Core kinematics data model - Pydantic version.

Conceptual hierarchy:
- Kinematics: Full time-varying dataset of a rigid body's motion (N frames)
- Pose: Single observation at one timestamp (horizontal slice)
- Trajectory: A quantity tracked across all timestamps (vertical slice)
- Timeseries: A single scalar component over time
"""
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from python_code.ferret_gaze.kinematics_core.quaternion_helper import Quaternion
from python_code.ferret_gaze.kinematics_core.reference_geometry_model import ReferenceGeometry


# =============================================================================
# TIMESERIES: Single scalar component over time
# =============================================================================


class Timeseries(BaseModel):
    """
    A single scalar quantity tracked over time.

    This is the most granular time-varying data: one value per timestamp.
    Examples: x-position, roll angle, speed.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    timestamps: NDArray[np.float64]  # (N,)
    values: NDArray[np.float64]  # (N,)

    @model_validator(mode="after")
    def validate_lengths(self) -> "Timeseries":
        if self.timestamps.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"timestamps length {self.timestamps.shape[0]} != "
                f"values length {self.values.shape[0]}"
            )
        return self

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> float:
        return float(self.values[idx])

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def mean_dt(self) -> float:
        return self.duration / (self.n_frames - 1) if self.n_frames > 1 else 0.0

    def differentiate(self) -> "Timeseries":
        """Compute time derivative using central differences."""
        n = len(self.values)
        derivative = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if i == 0:
                dt = self.timestamps[1] - self.timestamps[0]
                derivative[i] = (self.values[1] - self.values[0]) / dt
            elif i == n - 1:
                dt = self.timestamps[i] - self.timestamps[i - 1]
                derivative[i] = (self.values[i] - self.values[i - 1]) / dt
            else:
                dt = self.timestamps[i + 1] - self.timestamps[i - 1]
                derivative[i] = (self.values[i + 1] - self.values[i - 1]) / dt

        return Timeseries(
            name=f"d({self.name})/dt",
            timestamps=self.timestamps,
            values=derivative,
        )

    def interpolate(self, target_timestamps: NDArray[np.float64]) -> "Timeseries":
        """Linearly interpolate to new timestamps."""
        interpolated = np.interp(target_timestamps, self.timestamps, self.values)
        return Timeseries(
            name=self.name,
            timestamps=target_timestamps,
            values=interpolated,
        )


# =============================================================================
# TRAJECTORY: Multi-component quantity over time (vertical slice)
# =============================================================================


class Vec3Trajectory(BaseModel):
    """
    A 3D vector quantity tracked over time.

    Examples: position, velocity, angular velocity.
    Composed of three Timeseries (x, y, z components).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    timestamps: NDArray[np.float64]  # (N,)
    values: NDArray[np.float64]  # (N, 3)

    @model_validator(mode="after")
    def validate_shape(self) -> "Vec3Trajectory":
        expected = (len(self.timestamps), 3)
        if self.values.shape != expected:
            raise ValueError(f"values shape {self.values.shape} != {expected}")
        return self

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> NDArray[np.float64]:
        return self.values[idx]

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def x(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.x", timestamps=self.timestamps, values=self.values[:, 0])

    @property
    def y(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.y", timestamps=self.timestamps, values=self.values[:, 1])

    @property
    def z(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.z", timestamps=self.timestamps, values=self.values[:, 2])

    @property
    def magnitude(self) -> Timeseries:
        """Compute magnitude (norm) at each timestamp."""
        mags = np.linalg.norm(self.values, axis=1)
        return Timeseries(name=f"|{self.name}|", timestamps=self.timestamps, values=mags)

    def differentiate(self) -> "Vec3Trajectory":
        """Compute time derivative of each component."""
        dx = self.x.differentiate()
        dy = self.y.differentiate()
        dz = self.z.differentiate()
        return Vec3Trajectory(
            name=f"d({self.name})/dt",
            timestamps=self.timestamps,
            values=np.column_stack([dx.values, dy.values, dz.values]),
        )


class QuaternionTrajectory(BaseModel):
    """
    Orientation (as quaternions) tracked over time.

    Composed of four Timeseries (w, x, y, z components).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    timestamps: NDArray[np.float64]  # (N,)
    quaternions: list[Quaternion]  # (N,)

    @model_validator(mode="after")
    def validate_length(self) -> "QuaternionTrajectory":
        if len(self.quaternions) != len(self.timestamps):
            raise ValueError(
                f"quaternions length {len(self.quaternions)} != "
                f"timestamps length {len(self.timestamps)}"
            )
        return self

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int) -> Quaternion:
        return self.quaternions[idx]

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def w(self) -> Timeseries:
        values = np.array([q.w for q in self.quaternions])
        return Timeseries(name=f"{self.name}.w", timestamps=self.timestamps, values=values)

    @property
    def x(self) -> Timeseries:
        values = np.array([q.x for q in self.quaternions])
        return Timeseries(name=f"{self.name}.x", timestamps=self.timestamps, values=values)

    @property
    def y(self) -> Timeseries:
        values = np.array([q.y for q in self.quaternions])
        return Timeseries(name=f"{self.name}.y", timestamps=self.timestamps, values=values)

    @property
    def z(self) -> Timeseries:
        values = np.array([q.z for q in self.quaternions])
        return Timeseries(name=f"{self.name}.z", timestamps=self.timestamps, values=values)

    @property
    def roll(self) -> Timeseries:
        """Roll angle (rotation around X) over time."""
        values = np.array([q.to_euler_xyz()[0] for q in self.quaternions])
        return Timeseries(name=f"{self.name}.roll", timestamps=self.timestamps, values=values)

    @property
    def pitch(self) -> Timeseries:
        """Pitch angle (rotation around Y) over time."""
        values = np.array([q.to_euler_xyz()[1] for q in self.quaternions])
        return Timeseries(name=f"{self.name}.pitch", timestamps=self.timestamps, values=values)

    @property
    def yaw(self) -> Timeseries:
        """Yaw angle (rotation around Z) over time."""
        values = np.array([q.to_euler_xyz()[2] for q in self.quaternions])
        return Timeseries(name=f"{self.name}.yaw", timestamps=self.timestamps, values=values)

    def to_rotation_matrices(self) -> NDArray[np.float64]:
        """Convert all quaternions to rotation matrices. Returns (N, 3, 3)."""
        return np.array([q.to_rotation_matrix() for q in self.quaternions])


class AngularVelocityTrajectory(BaseModel):
    """
    Angular velocity tracked over time, with both global and local representations.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    timestamps: NDArray[np.float64]  # (N,)
    global_xyz: NDArray[np.float64]  # (N, 3) in world frame
    local_xyz: NDArray[np.float64]  # (N, 3) in body frame

    @model_validator(mode="after")
    def validate_shapes(self) -> "AngularVelocityTrajectory":
        n = len(self.timestamps)
        if self.global_xyz.shape != (n, 3):
            raise ValueError(f"global_xyz shape {self.global_xyz.shape} != ({n}, 3)")
        if self.local_xyz.shape != (n, 3):
            raise ValueError(f"local_xyz shape {self.local_xyz.shape} != ({n}, 3)")
        return self

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def global_roll(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.global_roll", timestamps=self.timestamps, values=self.global_xyz[:, 0])

    @property
    def global_pitch(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.global_pitch", timestamps=self.timestamps, values=self.global_xyz[:, 1])

    @property
    def global_yaw(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.global_yaw", timestamps=self.timestamps, values=self.global_xyz[:, 2])

    @property
    def local_roll(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.local_roll", timestamps=self.timestamps, values=self.local_xyz[:, 0])

    @property
    def local_pitch(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.local_pitch", timestamps=self.timestamps, values=self.local_xyz[:, 1])

    @property
    def local_yaw(self) -> Timeseries:
        return Timeseries(name=f"{self.name}.local_yaw", timestamps=self.timestamps, values=self.local_xyz[:, 2])

    @property
    def global_magnitude(self) -> Timeseries:
        mags = np.linalg.norm(self.global_xyz, axis=1)
        return Timeseries(name=f"|{self.name}|", timestamps=self.timestamps, values=mags)


# =============================================================================
# POSE: Single observation at one timestamp (horizontal slice)
# =============================================================================


class RigidBodyPose(BaseModel):
    """
    Complete state of a rigid body at a single instant in time.

    This is a "horizontal slice" of the kinematics data - one observation.
    Includes position, velocity, orientation, and angular velocity.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    reference_geometry: ReferenceGeometry
    timestamp: float

    # Position and velocity of the origin
    position: NDArray[np.float64]  # (3,) in mm
    velocity: NDArray[np.float64]  # (3,) in mm/s

    # Orientation and angular velocity
    orientation: Quaternion
    angular_velocity_global: NDArray[np.float64]  # (3,) rad/s in world frame
    angular_velocity_local: NDArray[np.float64]  # (3,) rad/s in body frame

    @field_validator("position",
                     "velocity",
                     "angular_velocity_global",
                     "angular_velocity_local")
    @classmethod
    def validate_vec3_shape(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if v.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {v.shape}")
        return v

    @property
    def basis_vectors(self) -> NDArray[np.float64]:
        """(3, 3) rotation matrix. Columns are body-frame basis vectors in world frame."""
        return self.orientation.to_rotation_matrix()

    @property
    def basis_x(self) -> NDArray[np.float64]:
        """Body X-axis direction in world frame."""
        return self.basis_vectors[:, 0]

    @property
    def basis_y(self) -> NDArray[np.float64]:
        """Body Y-axis direction in world frame."""
        return self.basis_vectors[:, 1]

    @property
    def basis_z(self) -> NDArray[np.float64]:
        """Body Z-axis direction in world frame."""
        return self.basis_vectors[:, 2]

    @property
    def keypoints(self) -> dict[str, NDArray[np.float64]]:
        """World-frame positions of all markers."""
        marker_positions = self.reference_geometry.get_marker_positions()
        return {
            name: self.position + self.orientation.rotate_vector(local_pos)
            for name, local_pos in marker_positions.items()
        }

    def get_keypoint(self, name: str) -> NDArray[np.float64]:
        """Get world-frame position of a specific marker."""
        if name not in self.reference_geometry.markers:
            raise KeyError(
                f"Keypoint '{name}' not found. "
                f"Available: {sorted(self.reference_geometry.markers.keys())}"
            )
        local_pos = self.reference_geometry.markers[name].to_array()
        return self.position + self.orientation.rotate_vector(local_pos)

    @property
    def speed(self) -> float:
        """Linear speed (magnitude of velocity)."""
        return float(np.linalg.norm(self.velocity))

    @property
    def angular_speed(self) -> float:
        """Angular speed (magnitude of angular velocity)."""
        return float(np.linalg.norm(self.angular_velocity_global))

    @property
    def euler_angles(self) -> tuple[float, float, float]:
        """(roll, pitch, yaw) in radians."""
        return self.orientation.to_euler_xyz()

    @property
    def homogeneous_transform(self) -> NDArray[np.float64]:
        """4x4 transformation matrix from body to world frame."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.basis_vectors
        T[:3, 3] = self.position
        return T


# =============================================================================
# KINEMATICS: Full time-varying dataset (the main container)
# =============================================================================


class RigidBodyKinematics(BaseModel):
    """
    Complete time-varying kinematics of a rigid body.

    This is the main container that holds all kinematic data and provides
    access via both horizontal timeslices (Pose) and vertical slices (Trajectory/timeseries).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    reference_geometry: ReferenceGeometry
    timestamps: NDArray[np.float64]  # (N,)

    # Position and velocity
    position_xyz: NDArray[np.float64]  # (N, 3) mm
    velocity_xyz: NDArray[np.float64]  # (N, 3) mm/s

    # Orientation
    orientations: list[Quaternion]  # (N,)

    # Angular velocity
    angular_velocity_global: NDArray[np.float64]  # (N, 3) rad/s world frame
    angular_velocity_local: NDArray[np.float64]  # (N, 3) rad/s body frame

    @model_validator(mode="after")
    def validate_shapes(self) -> "RigidBodyKinematics":
        n = len(self.timestamps)
        for name, arr, expected in [
            ("position_xyz", self.position_xyz, (n, 3)),
            ("velocity_xyz", self.velocity_xyz, (n, 3)),
            ("angular_velocity_global", self.angular_velocity_global, (n, 3)),
            ("angular_velocity_local", self.angular_velocity_local, (n, 3)),
        ]:
            if arr.shape != expected:
                raise ValueError(f"{name} shape {arr.shape} != {expected}")
        if len(self.orientations) != n:
            raise ValueError(f"orientations length {len(self.orientations)} != {n}")
        return self

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    # -------------------------------------------------------------------------
    # Horizontal slicing: get a Pose at a specific frame
    # -------------------------------------------------------------------------

    def get_pose_at_frame(self, idx: int) -> RigidBodyPose:
        """Get the pose at frame index."""
        return RigidBodyPose(
            reference_geometry=self.reference_geometry,
            timestamp=float(self.timestamps[idx]),
            position=self.position_xyz[idx],
            velocity=self.velocity_xyz[idx],
            orientation=self.orientations[idx],
            angular_velocity_global=self.angular_velocity_global[idx],
            angular_velocity_local=self.angular_velocity_local[idx],
        )

    def __getitem__(self, idx: int) -> RigidBodyPose:
        """Get the pose at frame index."""
        return self.get_pose_at_frame(idx)

    def __iter__(self) -> Iterator[RigidBodyPose]:
        """Iterate over all poses."""
        for i in range(self.n_frames):
            yield self.get_pose_at_frame(i)

    def __len__(self) -> int:
        return self.n_frames

    # -------------------------------------------------------------------------
    # Vertical slicing: get Trajectories
    # -------------------------------------------------------------------------

    @property
    def position_trajectory(self) -> Vec3Trajectory:
        """Position trajectory (x, y, z over time)."""
        return Vec3Trajectory(name="position", timestamps=self.timestamps, values=self.position_xyz)

    @property
    def velocity_trajectory(self) -> Vec3Trajectory:
        """Velocity trajectory (vx, vy, vz over time)."""
        return Vec3Trajectory(name="velocity", timestamps=self.timestamps, values=self.velocity_xyz)

    @property
    def orientation_trajectory(self) -> QuaternionTrajectory:
        """Orientation trajectory (quaternions over time)."""
        return QuaternionTrajectory(name="orientation", timestamps=self.timestamps, quaternions=self.orientations)

    @property
    def angular_velocity_trajectory(self) -> AngularVelocityTrajectory:
        """Angular velocity trajectory (global and local over time)."""
        return AngularVelocityTrajectory(
            name="angular_velocity",
            timestamps=self.timestamps,
            global_xyz=self.angular_velocity_global,
            local_xyz=self.angular_velocity_local,
        )

    def get_keypoint_trajectory(self, name: str) -> Vec3Trajectory:
        """Get the trajectory of a specific keypoint."""
        if name not in self.reference_geometry.markers:
            raise KeyError(
                f"Keypoint '{name}' not found. "
                f"Available: {sorted(self.reference_geometry.markers.keys())}"
            )

        local_pos = self.reference_geometry.markers[name].to_array()
        world_positions = np.zeros((self.n_frames, 3), dtype=np.float64)

        for i, q in enumerate(self.orientations):
            world_positions[i] = self.position_xyz[i] + q.rotate_vector(local_pos)

        return Vec3Trajectory(name=f"keypoint.{name}", timestamps=self.timestamps, values=world_positions)

    @property
    def keypoint_names(self) -> list[str]:
        """List of available keypoint names."""
        return list(self.reference_geometry.markers.keys())

    # -------------------------------------------------------------------------
    # Convenience accessors for individual timeseries
    # -------------------------------------------------------------------------

    @property
    def x(self) -> Timeseries:
        return self.position_trajectory.x

    @property
    def y(self) -> Timeseries:
        return self.position_trajectory.y

    @property
    def z(self) -> Timeseries:
        return self.position_trajectory.z

    @property
    def vx(self) -> Timeseries:
        return self.velocity_trajectory.x

    @property
    def vy(self) -> Timeseries:
        return self.velocity_trajectory.y

    @property
    def vz(self) -> Timeseries:
        return self.velocity_trajectory.z

    @property
    def speed(self) -> Timeseries:
        return self.velocity_trajectory.magnitude

    @property
    def roll(self) -> Timeseries:
        return self.orientation_trajectory.roll

    @property
    def pitch(self) -> Timeseries:
        return self.orientation_trajectory.pitch

    @property
    def yaw(self) -> Timeseries:
        return self.orientation_trajectory.yaw

    @property
    def angular_speed(self) -> Timeseries:
        return self.angular_velocity_trajectory.global_magnitude

    @property
    def global_angular_velocity_roll(self) -> Timeseries:
        return self.angular_velocity_trajectory.global_roll

    @property
    def global_angular_velocity_pitch(self) -> Timeseries:
        return self.angular_velocity_trajectory.global_pitch

    @property
    def global_angular_velocity_yaw(self) -> Timeseries:
        return self.angular_velocity_trajectory.global_yaw

    @property
    def local_angular_velocity_roll(self) -> Timeseries:
        return self.angular_velocity_trajectory.local_roll

    @property
    def local_angular_velocity_pitch(self) -> Timeseries:
        return self.angular_velocity_trajectory.local_pitch

    @property
    def local_angular_velocity_yaw(self) -> Timeseries:
        return self.angular_velocity_trajectory.local_yaw

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_pose_arrays(
        cls,
        reference_geometry: ReferenceGeometry,
        timestamps: NDArray[np.float64],
        position_xyz: NDArray[np.float64],
        quaternions_wxyz: NDArray[np.float64],
    ) -> "RigidBodyKinematics":
        """
        Construct from basic pose arrays, computing velocities automatically.

        Args:
            reference_geometry: The reference geometry
            timestamps: (N,) timestamps in seconds
            position_xyz: (N, 3) positions in mm
            quaternions_wxyz: (N, 4) quaternions as [w, x, y, z]
        """
        n_frames = len(timestamps)

        # Convert quaternions
        orientations = [
            Quaternion(
                w=float(quaternions_wxyz[i, 0]),
                x=float(quaternions_wxyz[i, 1]),
                y=float(quaternions_wxyz[i, 2]),
                z=float(quaternions_wxyz[i, 3]),
            )
            for i in range(n_frames)
        ]

        # Compute linear velocity
        velocity_xyz = _compute_velocity(position_xyz, timestamps)

        # Compute angular velocity
        angular_velocity_global, angular_velocity_local = _compute_angular_velocity(
            orientations, timestamps
        )

        return cls(
            reference_geometry=reference_geometry,
            timestamps=timestamps,
            position_xyz=position_xyz,
            velocity_xyz=velocity_xyz,
            orientations=orientations,
            angular_velocity_global=angular_velocity_global,
            angular_velocity_local=angular_velocity_local,
        )


# =============================================================================
# Helper functions for computing derived quantities
# =============================================================================


def _compute_velocity(
    position_xyz: NDArray[np.float64],
    timestamps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute linear velocity using forward differences, first frame padded."""
    n = len(timestamps)
    velocity = np.zeros((n, 3), dtype=np.float64)

    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        if dt > 1e-10:
            velocity[i] = (position_xyz[i] - position_xyz[i - 1]) / dt

    velocity[0] = velocity[1]
    return velocity


def _compute_angular_velocity(
    orientations: list[Quaternion],
    timestamps: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute global and local angular velocity from quaternion sequence."""
    n = len(orientations)

    global_xyz = np.zeros((n, 3), dtype=np.float64)
    local_xyz = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        if i == 0:
            dt = timestamps[1] - timestamps[0]
            q_curr = orientations[0]
            q_next = orientations[1]
        elif i == n - 1:
            dt = timestamps[i] - timestamps[i - 1]
            q_curr = orientations[i - 1]
            q_next = orientations[i]
        else:
            dt = timestamps[i + 1] - timestamps[i - 1]
            q_curr = orientations[i - 1]
            q_next = orientations[i + 1]

        if dt < 1e-10:
            continue

        # Relative rotation: q_rel = q_next * q_curr^-1
        q_rel = q_next * q_curr.inverse()
        axis, angle = q_rel.to_axis_angle()

        # Global angular velocity
        omega_global = axis * (angle / dt)
        global_xyz[i] = omega_global

        # Local angular velocity: transform into body frame
        R_curr = orientations[i].to_rotation_matrix()
        local_xyz[i] = R_curr.T @ omega_global

    return global_xyz, local_xyz