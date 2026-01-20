"""
Quaternion Trajectory Model
===========================

Vectorized quaternion trajectory representation for efficient kinematics computation.

This module provides QuaternionTrajectory, which stores orientation data as a (N, 4)
numpy array and performs all operations using vectorized numpy operations. Quaternion
objects are only created on-demand when individual frames are accessed.
"""

from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from python_code.kinematics_core.keypoint_trajectories import KeypointTrajectories
from python_code.kinematics_core.timeseries_model import Timeseries
from python_code.kinematics_core.quaternion_model import Quaternion


class QuaternionTrajectory(BaseModel):
    """
    Orientation (as quaternions) tracked over time.

    Stores quaternions as a (N, 4) numpy array with [w, x, y, z] ordering.
    All operations use vectorized numpy for efficiency. Individual Quaternion
    objects are created on-demand when accessed via indexing.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    timestamps: NDArray[np.float64]  # (N,)
    quaternions_wxyz: NDArray[np.float64]  # (N, 4) as [w, x, y, z]

    @model_validator(mode="after")
    def validate_lengths(self) -> "QuaternionTrajectory":
        n_timestamps = len(self.timestamps)
        if self.quaternions_wxyz.shape != (n_timestamps, 4):
            raise ValueError(
                f"quaternions_wxyz shape {self.quaternions_wxyz.shape} != "
                f"expected ({n_timestamps}, 4)"
            )
        return self

    @classmethod
    def from_wxyz_array(
        cls,
        name: str,
        timestamps: NDArray[np.float64],
        quaternions_wxyz: NDArray[np.float64],
    ) -> "QuaternionTrajectory":
        """
        Create from a (N, 4) array of quaternions as [w, x, y, z].

        Args:
            name: Identifier for this trajectory
            timestamps: (N,) array of timestamps
            quaternions_wxyz: (N, 4) array of quaternions [w, x, y, z]

        Returns:
            QuaternionTrajectory with normalized quaternions
        """
        if quaternions_wxyz.ndim != 2 or quaternions_wxyz.shape[1] != 4:
            raise ValueError(
                f"quaternions_wxyz must have shape (N, 4), got {quaternions_wxyz.shape}"
            )
        # Normalize quaternions (vectorized)
        norms = np.linalg.norm(quaternions_wxyz, axis=1, keepdims=True)
        quaternions_wxyz = quaternions_wxyz / norms

        return cls(
            name=name,
            timestamps=timestamps,
            quaternions_wxyz=quaternions_wxyz,
        )

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, index: int) -> Quaternion:
        """Get a single Quaternion object (created on demand)."""
        q = self.quaternions_wxyz[index]
        return Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @cached_property
    def quaternions(self) -> list[Quaternion]:
        """
        List of Quaternion objects - created lazily when accessed.

        Prefer using quaternions_wxyz directly for performance.
        """
        return [
            Quaternion(
                w=float(self.quaternions_wxyz[i, 0]),
                x=float(self.quaternions_wxyz[i, 1]),
                y=float(self.quaternions_wxyz[i, 2]),
                z=float(self.quaternions_wxyz[i, 3]),
            )
            for i in range(len(self.quaternions_wxyz))
        ]

    # -------------------------------------------------------------------------
    # Component accessors (vectorized)
    # -------------------------------------------------------------------------

    @property
    def w(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.w",
            timestamps=self.timestamps,
            values=self.quaternions_wxyz[:, 0],
        )

    @property
    def x(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.x",
            timestamps=self.timestamps,
            values=self.quaternions_wxyz[:, 1],
        )

    @property
    def y(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.y",
            timestamps=self.timestamps,
            values=self.quaternions_wxyz[:, 2],
        )

    @property
    def z(self) -> Timeseries:
        return Timeseries(
            name=f"{self.name}.z",
            timestamps=self.timestamps,
            values=self.quaternions_wxyz[:, 3],
        )

    # -------------------------------------------------------------------------
    # Vectorized vector rotation
    # -------------------------------------------------------------------------

    def rotate_vector(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Rotate a single vector by all quaternions in the trajectory.

        Uses vectorized rotation formula:
            v' = v + 2*w*(u × v) + 2*(u × (u × v))
        where u = [x, y, z] (vector part) and w is scalar part.

        Args:
            v: (3,) vector to rotate

        Returns:
            (N, 3) array of rotated vectors, one per frame
        """
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (3,):
            raise ValueError(f"Vector must have shape (3,), got {v.shape}")

        w = self.quaternions_wxyz[:, 0]  # (N,)
        u = self.quaternions_wxyz[:, 1:4]  # (N, 3)

        # Compute cross products vectorized
        uv = np.cross(u, v)  # (N, 3)
        uuv = np.cross(u, uv)  # (N, 3)

        return v + 2.0 * w[:, np.newaxis] * uv + 2.0 * uuv

    def rotate_vectors(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Rotate multiple vectors by all quaternions in the trajectory.

        Args:
            vectors: (M, 3) array of vectors to rotate

        Returns:
            (N, M, 3) array where result[n, m] is vectors[m] rotated by quaternion[n]
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError(f"vectors must have shape (M, 3), got {vectors.shape}")

        n_frames = self.n_frames
        n_vectors = len(vectors)

        w = self.quaternions_wxyz[:, 0]  # (N,)
        u = self.quaternions_wxyz[:, 1:4]  # (N, 3)

        # Broadcast vectors to (N, M, 3)
        v_broadcast = np.broadcast_to(vectors, (n_frames, n_vectors, 3))

        # Broadcast u to (N, M, 3) for cross product
        u_broadcast = u[:, np.newaxis, :]  # (N, 1, 3) -> broadcasts to (N, M, 3)

        # Compute cross products
        uv = np.cross(u_broadcast, v_broadcast)  # (N, M, 3)
        uuv = np.cross(u_broadcast, uv)  # (N, M, 3)

        return v_broadcast + 2.0 * w[:, np.newaxis, np.newaxis] * uv + 2.0 * uuv

    def compute_keypoint_trajectories(
        self,
        keypoint_names: tuple[str, ...],
        local_positions: NDArray[np.float64],
    ) -> KeypointTrajectories:
        """
        Pre-compute rotated trajectories for multiple keypoints.

        Args:
            keypoint_names: Names for each keypoint
            local_positions: (M, 3) local positions to rotate

        Returns:
            KeypointTrajectories with (N, M, 3) rotated positions
        """
        rotated = self.rotate_vectors(local_positions)
        return KeypointTrajectories(
            keypoint_names=keypoint_names,
            timestamps=self.timestamps,
            trajectories_fr_id_xyz=rotated,
        )

    # -------------------------------------------------------------------------
    # Euler angles (vectorized)
    # -------------------------------------------------------------------------

    @cached_property
    def _euler_xyz_array(self) -> NDArray[np.float64]:
        """
        Batch convert all quaternions to Euler angles (cached).

        Returns:
            (N, 3) array of [roll, pitch, yaw] in radians
        """
        w = self.quaternions_wxyz[:, 0]
        x = self.quaternions_wxyz[:, 1]
        y = self.quaternions_wxyz[:, 2]
        z = self.quaternions_wxyz[:, 3]

        # Roll (rotation around X)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (rotation around Y)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        # Yaw (rotation around Z)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.column_stack([roll, pitch, yaw])

    def to_euler_xyz_array(self) -> NDArray[np.float64]:
        """Get Euler angles as (N, 3) array."""
        return self._euler_xyz_array

    @property
    def roll(self) -> Timeseries:
        """Roll angle (rotation around X) over time."""
        return Timeseries(
            name=f"{self.name}.roll",
            timestamps=self.timestamps,
            values=self._euler_xyz_array[:, 0],
        )

    @property
    def pitch(self) -> Timeseries:
        """Pitch angle (rotation around Y) over time."""
        return Timeseries(
            name=f"{self.name}.pitch",
            timestamps=self.timestamps,
            values=self._euler_xyz_array[:, 1],
        )

    @property
    def yaw(self) -> Timeseries:
        """Yaw angle (rotation around Z) over time."""
        return Timeseries(
            name=f"{self.name}.yaw",
            timestamps=self.timestamps,
            values=self._euler_xyz_array[:, 2],
        )

    # -------------------------------------------------------------------------
    # Rotation matrices (vectorized)
    # -------------------------------------------------------------------------

    def to_rotation_matrices(self) -> NDArray[np.float64]:
        """
        Batch convert all quaternions to rotation matrices.

        Returns:
            (N, 3, 3) array of rotation matrices
        """
        w = self.quaternions_wxyz[:, 0]
        x = self.quaternions_wxyz[:, 1]
        y = self.quaternions_wxyz[:, 2]
        z = self.quaternions_wxyz[:, 3]

        # Pre-compute repeated terms
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        n_frames = len(self.quaternions_wxyz)
        R = np.empty((n_frames, 3, 3), dtype=np.float64)

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

    # -------------------------------------------------------------------------
    # Angular velocity (vectorized)
    # -------------------------------------------------------------------------

    def compute_angular_velocity(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute angular velocity using finite differences.

        Returns:
            Tuple of (global_angular_velocity, local_angular_velocity)
            Each is (N, 3) array in rad/s

        Raises:
            ValueError: If timestamps are not strictly increasing
        """
        n_frames = self.n_frames
        if n_frames < 2:
            raise ValueError(f"Need at least 2 frames, got {n_frames}")

        # Validate timestamps
        dt = np.diff(self.timestamps)
        invalid = np.where(dt <= 1e-10)[0]
        if len(invalid) > 0:
            raise ValueError(
                f"Invalid timestamp difference at frame {invalid[0]} -> {invalid[0] + 1}: "
                f"dt = {dt[invalid[0]]:.2e} seconds."
            )

        # Index arrays for finite differences
        # Frame 0: forward (curr=0, next=1)
        # Frame i: central (curr=i-1, next=i+1)
        # Frame N-1: backward (curr=N-2, next=N-1)
        curr_idx = np.concatenate([[0], np.arange(0, n_frames - 2), [n_frames - 2]])
        next_idx = np.concatenate([[1], np.arange(2, n_frames), [n_frames - 1]])

        # Time deltas
        time_deltas = np.empty(n_frames, dtype=np.float64)
        time_deltas[0] = self.timestamps[1] - self.timestamps[0]
        time_deltas[1:-1] = self.timestamps[2:] - self.timestamps[:-2]
        time_deltas[-1] = self.timestamps[-1] - self.timestamps[-2]

        # Get quaternion pairs
        q_curr = self.quaternions_wxyz[curr_idx]  # (N, 4)
        q_next = self.quaternions_wxyz[next_idx]  # (N, 4)

        # Relative quaternion: q_rel = q_next * conj(q_curr)
        q_curr_conj = q_curr * np.array([1, -1, -1, -1])
        q_rel = _batch_quat_multiply(q_next, q_curr_conj)

        # Extract axis-angle
        axes, angles = _batch_quat_to_axis_angle(q_rel)

        # Global angular velocity
        omega_global = axes * (angles / time_deltas)[:, np.newaxis]

        # Local angular velocity: R^T @ omega
        R = self.to_rotation_matrices()
        omega_local = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), omega_global)

        return omega_global, omega_local


# =============================================================================
# Batch quaternion helper functions
# =============================================================================


def _batch_quat_multiply(
    q_a: NDArray[np.float64],
    q_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Batch quaternion multiplication (Hamilton product)."""
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
    """Batch convert quaternions to axis-angle representation."""
    w = q[:, 0]
    xyz = q[:, 1:4]

    w_clamped = np.clip(w, -1.0, 1.0)
    angles = 2.0 * np.arccos(np.abs(w_clamped))
    sin_half = np.sqrt(1.0 - w_clamped ** 2)

    axes = np.zeros_like(xyz)
    valid = sin_half > 1e-10
    axes[valid] = xyz[valid] / sin_half[valid, np.newaxis]
    axes[~valid] = [1.0, 0.0, 0.0]

    # Flip axis if w < 0
    axes[w < 0] *= -1

    return axes, angles