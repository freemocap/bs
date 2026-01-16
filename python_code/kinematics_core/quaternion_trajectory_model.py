"""Quaternion trajectory model with vectorized batch operations."""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from python_code.kinematics_core.keypoint_trajectories import KeypointTrajectories
from python_code.kinematics_core.timeseries_model import Timeseries
from python_code.kinematics_core.quaternion_model import Quaternion


class QuaternionTrajectory(BaseModel):
    """
    Orientation (as quaternions) tracked over time.

    Stores both individual Quaternion objects for convenience and a (N, 4) array
    for fast vectorized operations.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    timestamps: NDArray[np.float64]  # (N,)
    quaternions: list[Quaternion]  # (N,)
    quaternions_wxyz: NDArray[np.float64]  # (N, 4) as [w, x, y, z]

    @model_validator(mode="after")
    def validate_lengths(self) -> "QuaternionTrajectory":
        number_of_timestamps = len(self.timestamps)
        if len(self.quaternions) != number_of_timestamps:
            raise ValueError(
                f"quaternions length {len(self.quaternions)} != "
                f"timestamps length {number_of_timestamps}"
            )
        if self.quaternions_wxyz.shape != (number_of_timestamps, 4):
            raise ValueError(
                f"quaternions_wxyz shape {self.quaternions_wxyz.shape} != "
                f"expected ({number_of_timestamps}, 4)"
            )
        return self

    @classmethod
    def from_quaternion_list(
        cls,
        name: str,
        timestamps: NDArray[np.float64],
        quaternions: list[Quaternion],
    ) -> "QuaternionTrajectory":
        """
        Create from a list of Quaternion objects.

        Builds the quaternions_wxyz automatically.
        """
        quaternions_wxyz = np.array(
            [[quaternion.w, quaternion.x, quaternion.y, quaternion.z] for quaternion in quaternions],
            dtype=np.float64,
        )
        return cls(
            name=name,
            timestamps=timestamps,
            quaternions=quaternions,
            quaternions_wxyz=quaternions_wxyz,
        )

    @classmethod
    def from_wxyz_array(
        cls,
        name: str,
        timestamps: NDArray[np.float64],
        quaternions_wxyz: NDArray[np.float64],
    ) -> "QuaternionTrajectory":
        """
        Create from a (N, 4) array of quaternions as [w, x, y, z].

        This is the fast path - builds Quaternion objects from the array.
        """
        if quaternions_wxyz.ndim != 2 or quaternions_wxyz.shape[1] != 4:
            raise ValueError(
                f"quaternions_wxyz must have shape (N, 4), got {quaternions_wxyz.shape}"
            )
        quaternions = [
            Quaternion(
                w=float(quaternions_wxyz[i, 0]),
                x=float(quaternions_wxyz[i, 1]),
                y=float(quaternions_wxyz[i, 2]),
                z=float(quaternions_wxyz[i, 3]),
            )
            for i in range(len(quaternions_wxyz))
        ]
        return cls(
            name=name,
            timestamps=timestamps,
            quaternions=quaternions,
            quaternions_wxyz=quaternions_wxyz,
        )

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, index: int) -> Quaternion:
        return self.quaternions[index]

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

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
        # uv = u × v for each quaternion
        uv = np.cross(u, v)  # (N, 3)
        # uuv = u × uv for each quaternion
        uuv = np.cross(u, uv)  # (N, 3)

        # v' = v + 2*w*uv + 2*uuv
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
        # uv[n, m] = u[n] × v[m]
        uv = np.cross(u_broadcast, v_broadcast)  # (N, M, 3)
        # uuv[n, m] = u[n] × uv[n, m]
        uuv = np.cross(u_broadcast, uv)  # (N, M, 3)

        # v' = v + 2*w*uv + 2*uuv
        # w needs shape (N, 1, 1) to broadcast correctly
        return v_broadcast + 2.0 * w[:, np.newaxis, np.newaxis] * uv + 2.0 * uuv

    def compute_keypoint_trajectories(
        self,
        keypoint_names: tuple[str, ...],
        local_positions: NDArray[np.float64],
    ) -> KeypointTrajectories:
        """
        Pre-compute rotated trajectories for multiple keypoints.

        This is the most efficient way to compute keypoint trajectories when you
        have multiple keypoints. All rotations are computed in a single vectorized
        operation.

        Args:
            keypoint_names: Tuple of M keypoint names (for lookup, immutable)
            local_positions: (M, 3) array of keypoint positions in body frame

        Returns:
            KeypointTrajectories container with pre-computed (N, M, 3) rotated positions.
            Access individual trajectories by name: result["keypoint_name"]

        Example:
            keypoint_names, local_pos = reference_geometry.get_marker_array()
            trajectories = orientations.compute_keypoint_trajectories(
                tuple(keypoint_names), local_pos
            )
            nose_trajectory = trajectories["nose"]  # (N, 3) array
        """
        local_positions = np.asarray(local_positions, dtype=np.float64)
        if local_positions.ndim != 2 or local_positions.shape[1] != 3:
            raise ValueError(
                f"local_positions must have shape (M, 3), got {local_positions.shape}"
            )
        if len(keypoint_names) != len(local_positions):
            raise ValueError(
                f"keypoint_names length ({len(keypoint_names)}) must match "
                f"local_positions length ({len(local_positions)})"
            )

        rotated = self.rotate_vectors(local_positions)  # (N, M, 3)

        return KeypointTrajectories(
            keypoint_names=keypoint_names,
            timestamps=self.timestamps,
            trajectories_fr_id_xyz=rotated,
        )

    # -------------------------------------------------------------------------
    # Euler angles (vectorized)
    # -------------------------------------------------------------------------

    def to_euler_xyz_array(self) -> NDArray[np.float64]:
        """
        Batch convert all quaternions to euler angles.

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

    @property
    def roll(self) -> Timeseries:
        """Roll angle (rotation around X) over time."""
        euler_angles = self.to_euler_xyz_array()
        return Timeseries(
            name=f"{self.name}.roll",
            timestamps=self.timestamps,
            values=euler_angles[:, 0],
        )

    @property
    def pitch(self) -> Timeseries:
        """Pitch angle (rotation around Y) over time."""
        euler_angles = self.to_euler_xyz_array()
        return Timeseries(
            name=f"{self.name}.pitch",
            timestamps=self.timestamps,
            values=euler_angles[:, 1],
        )

    @property
    def yaw(self) -> Timeseries:
        """Yaw angle (rotation around Z) over time."""
        euler_angles = self.to_euler_xyz_array()
        return Timeseries(
            name=f"{self.name}.yaw",
            timestamps=self.timestamps,
            values=euler_angles[:, 2],
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

        number_of_frames = len(self.quaternions_wxyz)
        rotation_matrices = np.empty((number_of_frames, 3, 3), dtype=np.float64)

        rotation_matrices[:, 0, 0] = 1 - 2 * (yy + zz)
        rotation_matrices[:, 0, 1] = 2 * (xy - wz)
        rotation_matrices[:, 0, 2] = 2 * (xz + wy)

        rotation_matrices[:, 1, 0] = 2 * (xy + wz)
        rotation_matrices[:, 1, 1] = 1 - 2 * (xx + zz)
        rotation_matrices[:, 1, 2] = 2 * (yz - wx)

        rotation_matrices[:, 2, 0] = 2 * (xz - wy)
        rotation_matrices[:, 2, 1] = 2 * (yz + wx)
        rotation_matrices[:, 2, 2] = 1 - 2 * (xx + yy)

        return rotation_matrices

    # -------------------------------------------------------------------------
    # Angular velocity (vectorized)
    # -------------------------------------------------------------------------

    def compute_angular_velocity(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute angular velocity using finite differences.

        Uses central differences for interior frames, forward/backward for endpoints.

        Returns:
            Tuple of (global_angular_velocity, local_angular_velocity)
            Each is (N, 3) array in rad/s with components [roll, pitch, yaw]

        Raises:
            ValueError: If timestamps are not strictly increasing
        """
        number_of_frames = self.n_frames
        if number_of_frames < 2:
            raise ValueError(
                f"Need at least 2 frames to compute angular velocity, got {number_of_frames}"
            )

        # Validate timestamps
        time_differences = np.diff(self.timestamps)
        invalid_indices = np.where(time_differences <= 1e-10)[0]
        if len(invalid_indices) > 0:
            first_invalid = invalid_indices[0]
            raise ValueError(
                f"Invalid timestamp difference at frame {first_invalid} -> {first_invalid + 1}: "
                f"dt = {time_differences[first_invalid]:.2e} seconds. "
                f"Timestamps must be strictly increasing."
            )

        # Build index arrays for "current" and "next" quaternions at each frame
        # Frame 0: forward difference (current=0, next=1)
        # Frame i (interior): central difference (current=i-1, next=i+1)
        # Frame N-1: backward difference (current=N-2, next=N-1)
        current_indices = np.concatenate([
            [0],
            np.arange(0, number_of_frames - 2),
            [number_of_frames - 2],
        ])
        next_indices = np.concatenate([
            [1],
            np.arange(2, number_of_frames),
            [number_of_frames - 1],
        ])

        # Build time deltas for each frame
        time_deltas = np.empty(number_of_frames, dtype=np.float64)
        time_deltas[0] = self.timestamps[1] - self.timestamps[0]
        time_deltas[1:-1] = self.timestamps[2:] - self.timestamps[:-2]
        time_deltas[-1] = self.timestamps[-1] - self.timestamps[-2]

        # Get quaternion arrays for current and next
        quaternions_current = self.quaternions_wxyz[current_indices]
        quaternions_next = self.quaternions_wxyz[next_indices]

        # Compute relative quaternion: q_relative = q_next * q_current^-1
        # For unit quaternions, inverse is conjugate: [w, -x, -y, -z]
        quaternions_current_conjugate = quaternions_current * np.array([1, -1, -1, -1])
        quaternions_relative = _batch_quaternion_multiply(
            quaternions_next,
            quaternions_current_conjugate,
        )

        # Extract axis-angle from relative quaternions
        axes, angles = _batch_quaternion_to_axis_angle(quaternions_relative)

        # Global angular velocity: omega = axis * (angle / dt)
        global_angular_velocity = axes * (angles / time_deltas)[:, np.newaxis]

        # Local angular velocity: R^T @ omega for each frame
        rotation_matrices = self.to_rotation_matrices()
        local_angular_velocity = np.einsum(
            "nij,nj->ni",
            rotation_matrices.transpose(0, 2, 1),
            global_angular_velocity,
        )

        return global_angular_velocity, local_angular_velocity


# =============================================================================
# Batch quaternion helper functions
# =============================================================================


def _batch_quaternion_multiply(
    quaternions_a: NDArray[np.float64],
    quaternions_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Batch quaternion multiplication (Hamilton product).

    Args:
        quaternions_a: (N, 4) array of quaternions [w, x, y, z]
        quaternions_b: (N, 4) array of quaternions [w, x, y, z]

    Returns:
        (N, 4) array of product quaternions
    """
    w1, x1, y1, z1 = quaternions_a[:, 0], quaternions_a[:, 1], quaternions_a[:, 2], quaternions_a[:, 3]
    w2, x2, y2, z2 = quaternions_b[:, 0], quaternions_b[:, 1], quaternions_b[:, 2], quaternions_b[:, 3]

    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _batch_quaternion_to_axis_angle(
    quaternions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Batch convert quaternions to axis-angle representation.

    Args:
        quaternions: (N, 4) array of quaternions [w, x, y, z]

    Returns:
        Tuple of (axes, angles) where axes is (N, 3) and angles is (N,)
    """
    w = quaternions[:, 0]
    xyz = quaternions[:, 1:4]

    # Clamp w to [-1, 1] for numerical stability
    w_clamped = np.clip(w, -1.0, 1.0)

    # angle = 2 * arccos(|w|)
    angles = 2.0 * np.arccos(np.abs(w_clamped))

    # sin(angle/2) = sqrt(1 - w^2)
    sin_half_angle = np.sqrt(1.0 - w_clamped ** 2)

    # axis = xyz / sin(angle/2), with fallback for small angles
    axes = np.zeros_like(xyz)
    valid_mask = sin_half_angle > 1e-10
    axes[valid_mask] = xyz[valid_mask] / sin_half_angle[valid_mask, np.newaxis]
    axes[~valid_mask] = [1.0, 0.0, 0.0]  # arbitrary axis for zero rotation

    # Flip axis if w < 0 (to stay in positive hemisphere)
    negative_w_mask = w < 0
    axes[negative_w_mask] *= -1

    return axes, angles