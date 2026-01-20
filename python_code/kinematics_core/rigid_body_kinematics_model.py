"""
Rigid Body Kinematics Model
===========================

This module provides RigidBodyKinematics, a complete representation of
time-varying rigid body motion including:

- Position trajectory (N, 3)
- Orientation trajectory as quaternions (N, 4)
- Derived quantities computed lazily: velocity, acceleration, angular velocity,
  angular acceleration, Euler angles
- Keypoint trajectories computed from reference geometry

All operations use vectorized numpy for efficient computation.
"""

from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from python_code.kinematics_core.keypoint_trajectories import KeypointTrajectories
from python_code.kinematics_core.quaternion_trajectory_model import QuaternionTrajectory
from python_code.kinematics_core.timeseries_model import Timeseries
from python_code.kinematics_core.vector3_trajectory_model import Vector3Trajectory
from python_code.kinematics_core.angular_velocity_trajectory_model import AngularVelocityTrajectory
from python_code.kinematics_core.angular_acceleration_trajectory_model import AngularAccelerationTrajectory
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.kinematics_core.quaternion_model import Quaternion


class RigidBodyKinematics(BaseModel):
    """
    Complete time-varying kinematics of a rigid body.

    Stores position and orientation trajectories, with derived quantities
    (velocity, acceleration, angular velocity, angular acceleration, Euler angles,
    keypoint positions) computed lazily on first access.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    reference_geometry: ReferenceGeometry
    timestamps: NDArray[np.float64]  # (N,)
    position_xyz: NDArray[np.float64]  # (N, 3) mm
    quaternions_wxyz: NDArray[np.float64]  # (N, 4) [w, x, y, z]

    @classmethod
    def from_pose_arrays(
        cls,
        name: str,
        reference_geometry: ReferenceGeometry,
        timestamps: NDArray[np.float64],
        position_xyz: NDArray[np.float64],
        quaternions_wxyz: NDArray[np.float64],
    ) -> "RigidBodyKinematics":
        """
        Construct from basic pose arrays.

        Derived quantities (velocities, accelerations, angular velocities,
        angular accelerations) are computed lazily on first access.

        Args:
            name: Identifier for this rigid body
            reference_geometry: Reference geometry defining keypoint positions
            timestamps: (N,) array of timestamps in seconds
            position_xyz: (N, 3) array of positions in mm
            quaternions_wxyz: (N, 4) array of quaternions [w, x, y, z]

        Returns:
            RigidBodyKinematics instance
        """
        # Normalize quaternions
        norms = np.linalg.norm(quaternions_wxyz, axis=1, keepdims=True)
        quaternions_wxyz = quaternions_wxyz / np.maximum(norms, 1e-10)

        return cls(
            name=name,
            reference_geometry=reference_geometry,
            timestamps=timestamps,
            position_xyz=position_xyz,
            quaternions_wxyz=quaternions_wxyz,
        )

    @model_validator(mode="after")
    def validate_shapes(self) -> "RigidBodyKinematics":
        n = len(self.timestamps)
        if self.position_xyz.shape != (n, 3):
            raise ValueError(f"position_xyz shape {self.position_xyz.shape} != ({n}, 3)")
        if self.quaternions_wxyz.shape != (n, 4):
            raise ValueError(f"quaternions_wxyz shape {self.quaternions_wxyz.shape} != ({n}, 4)")
        return self

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    # =========================================================================
    # LAZY COMPUTED PROPERTIES - VELOCITY AND ACCELERATION
    # =========================================================================

    @cached_property
    def velocity_xyz(self) -> NDArray[np.float64]:
        """Linear velocity (N, 3) in mm/s - computed lazily."""
        return _compute_velocity_vectorized(self.position_xyz, self.timestamps)

    @cached_property
    def acceleration_xyz(self) -> NDArray[np.float64]:
        """Linear acceleration (N, 3) in mm/s² - computed lazily."""
        return _compute_acceleration_vectorized(self.velocity_xyz, self.timestamps)

    @cached_property
    def _angular_velocity_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Angular velocity (global, local) - computed lazily."""
        return _compute_angular_velocity_vectorized(self.quaternions_wxyz, self.timestamps)

    @property
    def angular_velocity_global(self) -> NDArray[np.float64]:
        """Global angular velocity (N, 3) in rad/s."""
        return self._angular_velocity_arrays[0]

    @property
    def angular_velocity_local(self) -> NDArray[np.float64]:
        """Local angular velocity (N, 3) in rad/s."""
        return self._angular_velocity_arrays[1]

    @cached_property
    def _angular_acceleration_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Angular acceleration (global, local) - computed lazily.

        Global angular acceleration is the time derivative of global angular velocity.
        Local angular acceleration is R^T @ alpha_global, where R is the rotation matrix.

        Note: The local angular acceleration is NOT simply the derivative of local angular
        velocity. The relationship is:
            d(omega_local)/dt = R^T @ alpha_global

        This is because the body frame is rotating, so when we differentiate
        omega_local = R^T @ omega_global, the R^T term also changes. However, the
        cross-product term (omega × omega) vanishes, so we end up with alpha_local = R^T @ alpha_global.
        """
        return _compute_angular_acceleration_vectorized(
            angular_velocity_global=self.angular_velocity_global,
            quaternions_wxyz=self.quaternions_wxyz,
            timestamps=self.timestamps,
        )

    @property
    def angular_acceleration_global(self) -> NDArray[np.float64]:
        """Global angular acceleration (N, 3) in rad/s²."""
        return self._angular_acceleration_arrays[0]

    @property
    def angular_acceleration_local(self) -> NDArray[np.float64]:
        """Local angular acceleration (N, 3) in rad/s²."""
        return self._angular_acceleration_arrays[1]

    @cached_property
    def _euler_angles(self) -> NDArray[np.float64]:
        """Euler angles (N, 3) as [roll, pitch, yaw] - computed lazily."""
        return _quaternions_to_euler_vectorized(self.quaternions_wxyz)

    @cached_property
    def keypoint_trajectories(self) -> KeypointTrajectories:
        """Keypoint trajectories - computed lazily."""
        local_positions = self.reference_geometry.keypoint_local_positions_array

        # Rotate all local positions by all quaternions (vectorized)
        rotated = _rotate_vectors_by_quaternions_batch(
            self.quaternions_wxyz, local_positions
        )

        # Add body origin translation
        world_keypoints = rotated + self.position_xyz[:, np.newaxis, :]

        return KeypointTrajectories(
            keypoint_names=tuple(self.reference_geometry.keypoints.keys()),
            timestamps=self.timestamps,
            trajectories_fr_id_xyz=world_keypoints,
        )

    # =========================================================================
    # QUATERNION OPERATIONS (Vectorized)
    # =========================================================================

    def rotate_vector(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rotate a single vector by all quaternions. Returns (N, 3)."""
        return _rotate_vector_by_quaternions(self.quaternions_wxyz, v)

    def rotate_vectors(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rotate multiple vectors by all quaternions. Returns (N, M, 3)."""
        return _rotate_vectors_by_quaternions_batch(self.quaternions_wxyz, vectors)

    # =========================================================================
    # TRAJECTORY ACCESSORS
    # =========================================================================

    @property
    def position_trajectory(self) -> Vector3Trajectory:
        return Vector3Trajectory(
            name="position",
            timestamps=self.timestamps,
            values=self.position_xyz,
        )

    @property
    def velocity_trajectory(self) -> Vector3Trajectory:
        return Vector3Trajectory(
            name="velocity",
            timestamps=self.timestamps,
            values=self.velocity_xyz,
        )

    @property
    def acceleration_trajectory(self) -> Vector3Trajectory:
        return Vector3Trajectory(
            name="acceleration",
            timestamps=self.timestamps,
            values=self.acceleration_xyz,
        )

    @property
    def angular_velocity_trajectory(self) -> AngularVelocityTrajectory:
        return AngularVelocityTrajectory(
            name="angular_velocity",
            timestamps=self.timestamps,
            global_xyz=self.angular_velocity_global,
            local_xyz=self.angular_velocity_local,
        )

    @property
    def angular_acceleration_trajectory(self) -> AngularAccelerationTrajectory:
        return AngularAccelerationTrajectory(
            name="angular_acceleration",
            timestamps=self.timestamps,
            global_xyz=self.angular_acceleration_global,
            local_xyz=self.angular_acceleration_local,
        )

    # =========================================================================
    # INDIVIDUAL TIMESERIES ACCESSORS
    # =========================================================================

    @property
    def x(self) -> Timeseries:
        return Timeseries(name="x", timestamps=self.timestamps, values=self.position_xyz[:, 0])

    @property
    def y(self) -> Timeseries:
        return Timeseries(name="y", timestamps=self.timestamps, values=self.position_xyz[:, 1])

    @property
    def z(self) -> Timeseries:
        return Timeseries(name="z", timestamps=self.timestamps, values=self.position_xyz[:, 2])

    @property
    def vx(self) -> Timeseries:
        return Timeseries(name="vx", timestamps=self.timestamps, values=self.velocity_xyz[:, 0])

    @property
    def vy(self) -> Timeseries:
        return Timeseries(name="vy", timestamps=self.timestamps, values=self.velocity_xyz[:, 1])

    @property
    def vz(self) -> Timeseries:
        return Timeseries(name="vz", timestamps=self.timestamps, values=self.velocity_xyz[:, 2])

    @property
    def ax(self) -> Timeseries:
        return Timeseries(name="ax", timestamps=self.timestamps, values=self.acceleration_xyz[:, 0])

    @property
    def ay(self) -> Timeseries:
        return Timeseries(name="ay", timestamps=self.timestamps, values=self.acceleration_xyz[:, 1])

    @property
    def az(self) -> Timeseries:
        return Timeseries(name="az", timestamps=self.timestamps, values=self.acceleration_xyz[:, 2])

    @property
    def speed(self) -> Timeseries:
        mags = np.linalg.norm(self.velocity_xyz, axis=1)
        return Timeseries(name="speed", timestamps=self.timestamps, values=mags)

    @property
    def acceleration_magnitude(self) -> Timeseries:
        mags = np.linalg.norm(self.acceleration_xyz, axis=1)
        return Timeseries(name="acceleration_magnitude", timestamps=self.timestamps, values=mags)

    @property
    def roll(self) -> Timeseries:
        return Timeseries(name="roll", timestamps=self.timestamps, values=self._euler_angles[:, 0])

    @property
    def pitch(self) -> Timeseries:
        return Timeseries(name="pitch", timestamps=self.timestamps, values=self._euler_angles[:, 1])

    @property
    def yaw(self) -> Timeseries:
        return Timeseries(name="yaw", timestamps=self.timestamps, values=self._euler_angles[:, 2])

    @property
    def angular_speed(self) -> Timeseries:
        mags = np.linalg.norm(self.angular_velocity_global, axis=1)
        return Timeseries(name="angular_speed", timestamps=self.timestamps, values=mags)

    @property
    def angular_acceleration_magnitude(self) -> Timeseries:
        mags = np.linalg.norm(self.angular_acceleration_global, axis=1)
        return Timeseries(name="angular_acceleration_magnitude", timestamps=self.timestamps, values=mags)

    @property
    def keypoint_names(self) -> list[str]:
        return list(self.reference_geometry.keypoints.keys())

    # =========================================================================
    # FRAME ACCESS (creates Quaternion objects only when needed)
    # =========================================================================

    def get_quaternion(self, idx: int) -> Quaternion:
        """Get Quaternion object for a single frame (created on demand)."""
        q = self.quaternions_wxyz[idx]
        return Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))

    def __len__(self) -> int:
        return self.n_frames

    @cached_property
    def orientations(self) -> QuaternionTrajectory:
        """Orientation trajectory as a QuaternionTrajectory object."""
        return QuaternionTrajectory.from_wxyz_array(
            name=f"{self.name}_orientation",
            timestamps=self.timestamps,
            quaternions_wxyz=self.quaternions_wxyz,
        )


# =============================================================================
# VECTORIZED HELPER FUNCTIONS
# =============================================================================


def _compute_velocity_vectorized(
    position_xyz: NDArray[np.float64],
    timestamps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute linear velocity using forward differences (vectorized)."""
    n = len(timestamps)
    if n < 2:
        raise ValueError(f"Need at least 2 frames, got {n}")

    dt = np.diff(timestamps)
    if np.any(dt <= 1e-10):
        raise ValueError("Timestamps must be strictly increasing")

    # Position differences
    dp = np.diff(position_xyz, axis=0)

    # Velocity: v[i] = dp[i-1] / dt[i-1] for i > 0
    velocity = np.zeros((n, 3), dtype=np.float64)
    velocity[1:] = dp / dt[:, np.newaxis]
    velocity[0] = velocity[1]  # Copy first from second

    return velocity


def _compute_acceleration_vectorized(
    velocity_xyz: NDArray[np.float64],
    timestamps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute linear acceleration using central differences (vectorized).

    Uses central differences for interior points, forward/backward for endpoints.
    This provides better accuracy than pure forward differences.

    Args:
        velocity_xyz: (N, 3) array of velocities in mm/s
        timestamps: (N,) array of timestamps in seconds

    Returns:
        (N, 3) array of accelerations in mm/s²
    """
    n = len(timestamps)
    if n < 2:
        raise ValueError(f"Need at least 2 frames, got {n}")

    dt = np.diff(timestamps)
    if np.any(dt <= 1e-10):
        raise ValueError("Timestamps must be strictly increasing")

    acceleration = np.zeros((n, 3), dtype=np.float64)

    # Forward difference for first frame
    acceleration[0] = (velocity_xyz[1] - velocity_xyz[0]) / dt[0]

    # Central differences for interior frames
    if n > 2:
        dt_central = timestamps[2:] - timestamps[:-2]  # (n-2,)
        dv_central = velocity_xyz[2:] - velocity_xyz[:-2]  # (n-2, 3)
        acceleration[1:-1] = dv_central / dt_central[:, np.newaxis]

    # Backward difference for last frame
    acceleration[-1] = (velocity_xyz[-1] - velocity_xyz[-2]) / dt[-1]

    return acceleration


def _compute_angular_velocity_vectorized(
    quaternions_wxyz: NDArray[np.float64],
    timestamps: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute angular velocity from quaternions (fully vectorized)."""
    n = len(timestamps)
    if n < 2:
        raise ValueError(f"Need at least 2 frames, got {n}")

    dt = np.diff(timestamps)
    if np.any(dt <= 1e-10):
        raise ValueError("Timestamps must be strictly increasing")

    # Index arrays for finite differences
    curr_idx = np.concatenate([[0], np.arange(0, n - 2), [n - 2]])
    next_idx = np.concatenate([[1], np.arange(2, n), [n - 1]])

    # Time deltas
    time_deltas = np.empty(n, dtype=np.float64)
    time_deltas[0] = timestamps[1] - timestamps[0]
    time_deltas[1:-1] = timestamps[2:] - timestamps[:-2]
    time_deltas[-1] = timestamps[-1] - timestamps[-2]

    # Get quaternion pairs
    q_curr = quaternions_wxyz[curr_idx]
    q_next = quaternions_wxyz[next_idx]

    # Relative quaternion: q_rel = q_next * conj(q_curr)
    q_curr_conj = q_curr * np.array([1, -1, -1, -1])
    q_rel = _batch_quat_multiply(q_next, q_curr_conj)

    # Extract axis-angle
    axes, angles = _batch_quat_to_axis_angle(q_rel)

    # Global angular velocity
    omega_global = axes * (angles / time_deltas)[:, np.newaxis]

    # Local angular velocity: R^T @ omega
    R = _batch_quat_to_rotation_matrix(quaternions_wxyz)
    omega_local = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), omega_global)

    return omega_global, omega_local


def _compute_angular_acceleration_vectorized(
    angular_velocity_global: NDArray[np.float64],
    quaternions_wxyz: NDArray[np.float64],
    timestamps: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute angular acceleration from angular velocity (fully vectorized).

    The global angular acceleration is simply the time derivative of the global
    angular velocity.

    The local angular acceleration is obtained by transforming the global angular
    acceleration to the body frame: alpha_local = R^T @ alpha_global

    Mathematical justification:
    - omega_local = R^T @ omega_global
    - Taking the time derivative:
      d(omega_local)/dt = d(R^T)/dt @ omega_global + R^T @ d(omega_global)/dt
    - Since d(R^T)/dt = -R^T @ [omega_global]× where [.]× is the skew-symmetric matrix
    - And [omega_global]× @ omega_global = omega_global × omega_global = 0
    - We get: alpha_local = R^T @ alpha_global

    Args:
        angular_velocity_global: (N, 3) global angular velocity in rad/s
        quaternions_wxyz: (N, 4) quaternions [w, x, y, z]
        timestamps: (N,) timestamps in seconds

    Returns:
        Tuple of (alpha_global, alpha_local) each (N, 3) in rad/s²
    """
    n = len(timestamps)
    if n < 2:
        raise ValueError(f"Need at least 2 frames, got {n}")

    dt = np.diff(timestamps)
    if np.any(dt <= 1e-10):
        raise ValueError("Timestamps must be strictly increasing")

    # Compute global angular acceleration using central differences
    alpha_global = np.zeros((n, 3), dtype=np.float64)

    # Forward difference for first frame
    alpha_global[0] = (angular_velocity_global[1] - angular_velocity_global[0]) / dt[0]

    # Central differences for interior frames
    if n > 2:
        dt_central = timestamps[2:] - timestamps[:-2]
        domega_central = angular_velocity_global[2:] - angular_velocity_global[:-2]
        alpha_global[1:-1] = domega_central / dt_central[:, np.newaxis]

    # Backward difference for last frame
    alpha_global[-1] = (angular_velocity_global[-1] - angular_velocity_global[-2]) / dt[-1]

    # Transform to local frame: alpha_local = R^T @ alpha_global
    R = _batch_quat_to_rotation_matrix(quaternions_wxyz)
    alpha_local = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), alpha_global)

    return alpha_global, alpha_local


def _quaternions_to_euler_vectorized(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternions to Euler angles (vectorized)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # Yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.column_stack([roll, pitch, yaw])


def _rotate_vector_by_quaternions(
    q: NDArray[np.float64],
    v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Rotate a single vector by N quaternions.

    Args:
        q: (N, 4) quaternions [w, x, y, z]
        v: (3,) vector to rotate

    Returns:
        (N, 3) rotated vectors
    """
    w = q[:, 0]
    u = q[:, 1:4]

    uv = np.cross(u, v)
    uuv = np.cross(u, uv)

    return v + 2.0 * w[:, np.newaxis] * uv + 2.0 * uuv


def _rotate_vectors_by_quaternions_batch(
    q: NDArray[np.float64],
    vectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Rotate M vectors by N quaternions.

    Args:
        q: (N, 4) quaternions [w, x, y, z]
        vectors: (M, 3) vectors to rotate

    Returns:
        (N, M, 3) rotated vectors
    """
    n_frames = len(q)
    n_vectors = len(vectors)

    w = q[:, 0]  # (N,)
    u = q[:, 1:4]  # (N, 3)

    # Broadcast
    v_broadcast = np.broadcast_to(vectors, (n_frames, n_vectors, 3))
    u_broadcast = u[:, np.newaxis, :]  # (N, 1, 3)

    uv = np.cross(u_broadcast, v_broadcast)
    uuv = np.cross(u_broadcast, uv)

    return v_broadcast + 2.0 * w[:, np.newaxis, np.newaxis] * uv + 2.0 * uuv


def _batch_quat_multiply(
    q_a: NDArray[np.float64],
    q_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batch quaternion multiplication."""
    w1, x1, y1, z1 = q_a[:, 0], q_a[:, 1], q_a[:, 2], q_a[:, 3]
    w2, x2, y2, z2 = q_b[:, 0], q_b[:, 1], q_b[:, 2], q_b[:, 3]

    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _batch_quat_to_axis_angle(
    q: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Batch convert quaternions to axis-angle."""
    w = q[:, 0]
    xyz = q[:, 1:4]

    w_clamped = np.clip(w, -1.0, 1.0)
    angles = 2.0 * np.arccos(np.abs(w_clamped))
    sin_half = np.sqrt(1.0 - w_clamped ** 2)

    axes = np.zeros_like(xyz)
    valid = sin_half > 1e-10
    axes[valid] = xyz[valid] / sin_half[valid, np.newaxis]
    axes[~valid] = [1.0, 0.0, 0.0]
    axes[w < 0] *= -1

    return axes, angles


def _batch_quat_to_rotation_matrix(
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batch convert quaternions to rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    n = len(q)
    R = np.empty((n, 3, 3), dtype=np.float64)

    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)

    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)

    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)

    return R