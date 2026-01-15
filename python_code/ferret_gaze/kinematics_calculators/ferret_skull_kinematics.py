"""
Ferret Skull Kinematics Analysis

Computes skull position, orientation (Euler angles), angular velocity in both
world frame and skull-local frame, and orthonormal basis vectors of the skull
coordinate frame. Also computes eye socket kinematics as sub-components.

COORDINATE FRAME CONVENTIONS
============================

Skull Local Frame (body-fixed, defined in reference_geometry.py):
    Origin: Midpoint of left_eye and right_eye markers
    +X: Forward (toward nose / rostral)
    +Y: Left (toward left_eye / lateral), orthogonalized to be perpendicular to X
    +Z: Up (dorsal), computed as Y × X

    This is a right-handed anatomical coordinate system.

World Frame:
    Fixed global reference frame from motion capture system.

Quaternion Convention:
    Quaternions represent the rotation from LOCAL frame to WORLD frame.
    q.rotate_vector(v_local) → v_world

Euler Angles (XYZ extrinsic / ZYX intrinsic):
    Roll:  Rotation around X (head tilt left/right)
    Pitch: Rotation around Y (head nod up/down)
    Yaw:   Rotation around Z (head turn left/right)

Angular Velocity:
    World frame: ω vector expressed in world coordinate axes [ωx, ωy, ωz]
    Local frame: ω vector expressed in skull coordinate axes [roll_rate, pitch_rate, yaw_rate]

    The LOCAL angular velocity is what the vestibular system senses, since the
    semicircular canals are fixed in the skull. For VOR analysis:
    - Horizontal canals sense yaw_rate (rotation around skull Z-axis)
    - This should anti-correlate with horizontal eye velocity (VOR gain ≈ -1)

Basis Vectors:
    basis_x/y/z store the skull's local axis directions expressed in world coordinates.
    e.g., basis_x[frame] tells you "which direction does the skull's nose point in world coords?"

EYE SOCKET COORDINATE FRAMES
============================

The eye sockets have fixed positions and orientations relative to the skull.
Each socket has a right-handed coordinate system designed for eye tracking:

Right Eye Socket Frame:
    Origin: right_eye marker position (fixed in skull local coords)
    +X: Outward (lateral)      → skull -Y direction
    +Y: Toward nose (medial)   → skull +X direction
    +Z: Dorsal (up)            → skull +Z direction

Left Eye Socket Frame:
    Origin: left_eye marker position (fixed in skull local coords)
    +X: Outward (lateral)      → skull +Y direction
    +Y: Away from nose         → skull -X direction
    +Z: Dorsal (up)            → skull +Z direction

Both frames are right-handed (X × Y = Z).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_code.ferret_gaze.kinematics_core.quaternion_model import Quaternion


@dataclass
class EyeSocketKinematics:
    """
    Eye socket kinematics for a single eye (left or right).

    The eye socket frame is fixed relative to the skull. This stores
    the socket's position and orientation in the world frame for each frame.
    """

    eye_name: str  # "left" or "right"
    timestamps: NDArray[np.float64]  # (N,) seconds
    position_mm: NDArray[np.float64]  # (N, 3) socket center in world frame
    orientation_quaternions: list[Quaternion]  # N quaternions (socket local → world)
    basis_x: NDArray[np.float64]  # (N, 3) socket +X (outward) in world coords
    basis_y: NDArray[np.float64]  # (N, 3) socket +Y in world coords
    basis_z: NDArray[np.float64]  # (N, 3) socket +Z (dorsal) in world coords
    position_in_skull_mm: NDArray[np.float64]  # (3,) fixed position in skull local frame

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.timestamps)

    def __post_init__(self) -> None:
        """Validate all array shapes are consistent."""
        n = self.n_frames

        if self.eye_name not in ("left", "right"):
            raise ValueError(f"eye_name must be 'left' or 'right', got '{self.eye_name}'")
        if self.timestamps.ndim != 1:
            raise ValueError(f"timestamps must be 1D, got shape {self.timestamps.shape}")
        if self.position_mm.shape != (n, 3):
            raise ValueError(f"position_mm shape {self.position_mm.shape} must be ({n}, 3)")
        if len(self.orientation_quaternions) != n:
            raise ValueError(
                f"orientation_quaternions length {len(self.orientation_quaternions)} "
                f"must match timestamps length {n}"
            )
        if self.basis_x.shape != (n, 3):
            raise ValueError(f"basis_x shape {self.basis_x.shape} must be ({n}, 3)")
        if self.basis_y.shape != (n, 3):
            raise ValueError(f"basis_y shape {self.basis_y.shape} must be ({n}, 3)")
        if self.basis_z.shape != (n, 3):
            raise ValueError(f"basis_z shape {self.basis_z.shape} must be ({n}, 3)")
        if self.position_in_skull_mm.shape != (3,):
            raise ValueError(f"position_in_skull_mm shape {self.position_in_skull_mm.shape} must be (3,)")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert EyeSocketKinematics to a pandas DataFrame."""
        return pd.DataFrame({
            "frame": np.arange(self.n_frames),
            "timestamp_s": self.timestamps,
            "eye": self.eye_name,
            "position_x_mm": self.position_mm[:, 0],
            "position_y_mm": self.position_mm[:, 1],
            "position_z_mm": self.position_mm[:, 2],
            "quaternion_w": [q.w for q in self.orientation_quaternions],
            "quaternion_x": [q.x for q in self.orientation_quaternions],
            "quaternion_y": [q.y for q in self.orientation_quaternions],
            "quaternion_z": [q.z for q in self.orientation_quaternions],
            "basis_x_x": self.basis_x[:, 0],
            "basis_x_y": self.basis_x[:, 1],
            "basis_x_z": self.basis_x[:, 2],
            "basis_y_x": self.basis_y[:, 0],
            "basis_y_y": self.basis_y[:, 1],
            "basis_y_z": self.basis_y[:, 2],
            "basis_z_x": self.basis_z[:, 0],
            "basis_z_y": self.basis_z[:, 1],
            "basis_z_z": self.basis_z[:, 2],
        })


@dataclass
class SkullKinematics:
    """
    Skull kinematics data with position, orientation, angular velocity,
    and eye socket sub-components.

    All data is stored for N frames. Quaternions represent rotation from
    skull-local frame to world frame.
    """

    timestamps: NDArray[np.float64]  # (N,) seconds
    position_mm: NDArray[np.float64]  # (N, 3) skull origin position in world frame [mm]
    orientation_quaternions: list[Quaternion]  # N quaternions (local → world rotation)
    euler_angles_deg: NDArray[np.float64]  # (N, 3) [roll, pitch, yaw] in world frame [deg]
    angular_velocity_world_deg_s: NDArray[np.float64]  # (N, 3) [ωx, ωy, ωz] world frame [deg/s]
    angular_velocity_local_deg_s: NDArray[np.float64]  # (N, 3) [roll_rate, pitch_rate, yaw_rate] local frame [deg/s]
    basis_x: NDArray[np.float64]  # (N, 3) skull +X direction (forward) in world coords
    basis_y: NDArray[np.float64]  # (N, 3) skull +Y direction (left) in world coords
    basis_z: NDArray[np.float64]  # (N, 3) skull +Z direction (up) in world coords
    left_eye_socket: EyeSocketKinematics  # left eye socket kinematics
    right_eye_socket: EyeSocketKinematics  # right eye socket kinematics

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.timestamps)

    @property
    def head_yaw_deg(self) -> NDArray[np.float64]:
        """Head yaw angle in degrees (horizontal head direction)."""
        return self.euler_angles_deg[:, 2]

    @property
    def head_pitch_deg(self) -> NDArray[np.float64]:
        """Head pitch angle in degrees (head nod up/down)."""
        return self.euler_angles_deg[:, 1]

    @property
    def head_roll_deg(self) -> NDArray[np.float64]:
        """Head roll angle in degrees (head tilt left/right)."""
        return self.euler_angles_deg[:, 0]

    @property
    def head_yaw_velocity_deg_s(self) -> NDArray[np.float64]:
        """
        Head yaw velocity in degrees/second (local frame).

        This is what the horizontal semicircular canals sense.
        For VOR analysis, this should anti-correlate with horizontal eye velocity.
        """
        return self.angular_velocity_local_deg_s[:, 2]

    @property
    def head_pitch_velocity_deg_s(self) -> NDArray[np.float64]:
        """Head pitch velocity in degrees/second (local frame)."""
        return self.angular_velocity_local_deg_s[:, 1]

    @property
    def head_roll_velocity_deg_s(self) -> NDArray[np.float64]:
        """Head roll velocity in degrees/second (local frame)."""
        return self.angular_velocity_local_deg_s[:, 0]

    def __post_init__(self) -> None:
        """Validate all array shapes are consistent."""
        n = self.n_frames

        if self.timestamps.ndim != 1:
            raise ValueError(f"timestamps must be 1D, got shape {self.timestamps.shape}")
        if self.position_mm.shape != (n, 3):
            raise ValueError(f"position_mm shape {self.position_mm.shape} must be ({n}, 3)")
        if len(self.orientation_quaternions) != n:
            raise ValueError(
                f"orientation_quaternions length {len(self.orientation_quaternions)} "
                f"must match timestamps length {n}"
            )
        if self.euler_angles_deg.shape != (n, 3):
            raise ValueError(f"euler_angles_deg shape {self.euler_angles_deg.shape} must be ({n}, 3)")
        if self.angular_velocity_world_deg_s.shape != (n, 3):
            raise ValueError(
                f"angular_velocity_world_deg_s shape {self.angular_velocity_world_deg_s.shape} "
                f"must be ({n}, 3)"
            )
        if self.angular_velocity_local_deg_s.shape != (n, 3):
            raise ValueError(
                f"angular_velocity_local_deg_s shape {self.angular_velocity_local_deg_s.shape} "
                f"must be ({n}, 3)"
            )
        if self.basis_x.shape != (n, 3):
            raise ValueError(f"basis_x shape {self.basis_x.shape} must be ({n}, 3)")
        if self.basis_y.shape != (n, 3):
            raise ValueError(f"basis_y shape {self.basis_y.shape} must be ({n}, 3)")
        if self.basis_z.shape != (n, 3):
            raise ValueError(f"basis_z shape {self.basis_z.shape} must be ({n}, 3)")
        if self.left_eye_socket.n_frames != n:
            raise ValueError(
                f"left_eye_socket has {self.left_eye_socket.n_frames} frames, expected {n}"
            )
        if self.right_eye_socket.n_frames != n:
            raise ValueError(
                f"right_eye_socket has {self.right_eye_socket.n_frames} frames, expected {n}"
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert SkullKinematics to a pandas DataFrame (skull data only)."""
        return pd.DataFrame({
            "frame": np.arange(self.n_frames),
            "timestamp_s": self.timestamps,
            "position_x_mm": self.position_mm[:, 0],
            "position_y_mm": self.position_mm[:, 1],
            "position_z_mm": self.position_mm[:, 2],
            "quaternion_w": [q.w for q in self.orientation_quaternions],
            "quaternion_x": [q.x for q in self.orientation_quaternions],
            "quaternion_y": [q.y for q in self.orientation_quaternions],
            "quaternion_z": [q.z for q in self.orientation_quaternions],
            "roll_deg": self.euler_angles_deg[:, 0],
            "pitch_deg": self.euler_angles_deg[:, 1],
            "yaw_deg": self.euler_angles_deg[:, 2],
            "omega_world_x_deg_s": self.angular_velocity_world_deg_s[:, 0],
            "omega_world_y_deg_s": self.angular_velocity_world_deg_s[:, 1],
            "omega_world_z_deg_s": self.angular_velocity_world_deg_s[:, 2],
            "roll_rate_local_deg_s": self.angular_velocity_local_deg_s[:, 0],
            "pitch_rate_local_deg_s": self.angular_velocity_local_deg_s[:, 1],
            "yaw_rate_local_deg_s": self.angular_velocity_local_deg_s[:, 2],
            "basis_x_x": self.basis_x[:, 0],
            "basis_x_y": self.basis_x[:, 1],
            "basis_x_z": self.basis_x[:, 2],
            "basis_y_x": self.basis_y[:, 0],
            "basis_y_y": self.basis_y[:, 1],
            "basis_y_z": self.basis_y[:, 2],
            "basis_z_x": self.basis_z[:, 0],
            "basis_z_y": self.basis_z[:, 1],
            "basis_z_z": self.basis_z[:, 2],
        })

    @staticmethod
    def _compute_angular_velocity_world_rad_s(
        q_prev: Quaternion,
        q_curr: Quaternion,
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Compute angular velocity in world frame from consecutive quaternions.

        Uses finite difference: ω ≈ axis-angle of delta rotation / dt.

        Args:
            q_prev: Quaternion at time t
            q_curr: Quaternion at time t + dt
            dt: Time step in seconds (must be positive)

        Returns:
            Angular velocity vector in world frame [rad/s], shape (3,)
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        delta_q = q_curr * q_prev.inverse()
        if delta_q.w < 0:
            delta_q = Quaternion(w=-delta_q.w, x=-delta_q.x, y=-delta_q.y, z=-delta_q.z)

        axis, angle = delta_q.to_axis_angle()
        return (angle / dt) * axis

    @staticmethod
    def _transform_omega_world_to_local(
        omega_world: NDArray[np.float64],
        q: Quaternion,
    ) -> NDArray[np.float64]:
        """
        Transform angular velocity from world frame to local (skull) frame.

        Args:
            omega_world: Angular velocity in world frame, shape (3,)
            q: Orientation quaternion (local → world)

        Returns:
            Angular velocity in skull-local frame, shape (3,)
        """
        if omega_world.shape != (3,):
            raise ValueError(f"omega_world must have shape (3,), got {omega_world.shape}")
        return q.inverse().rotate_vector(omega_world)

    @staticmethod
    def _rotation_matrix_to_quaternion(R: NDArray[np.float64]) -> Quaternion:
        """Convert a 3x3 rotation matrix to a quaternion using Shepperd's method."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        q = Quaternion(w=w, x=x, y=y, z=z).normalized()
        if q.w < 0:
            q = Quaternion(w=-q.w, x=-q.x, y=-q.y, z=-q.z)
        return q

    @staticmethod
    def _get_socket_to_skull_rotation(eye_name: str) -> NDArray[np.float64]:
        """
        Get rotation matrix from eye socket frame to skull frame.

        Args:
            eye_name: "left" or "right"

        Returns:
            (3, 3) rotation matrix R where R @ v_socket = v_skull
        """
        if eye_name == "right":
            # Right socket: +X=skull -Y, +Y=skull +X, +Z=skull +Z
            return np.array([
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        elif eye_name == "left":
            # Left socket: +X=skull +Y, +Y=skull -X, +Z=skull +Z
            return np.array([
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        else:
            raise ValueError(f"eye_name must be 'left' or 'right', got '{eye_name}'")

    @classmethod
    def _compute_eye_socket_kinematics(
        cls,
        eye_name: str,
        eye_position_in_skull: NDArray[np.float64],
        timestamps: NDArray[np.float64],
        skull_positions_mm: NDArray[np.float64],
        skull_quaternions: list[Quaternion],
    ) -> EyeSocketKinematics:
        """
        Compute eye socket kinematics from skull pose data.

        Args:
            eye_name: "left" or "right"
            eye_position_in_skull: (3,) eye socket center in skull local coords [mm]
            timestamps: (N,) timestamps in seconds
            skull_positions_mm: (N, 3) skull origin positions in mm
            skull_quaternions: N skull orientation quaternions

        Returns:
            EyeSocketKinematics for the specified eye
        """
        n_frames = len(timestamps)

        R_socket_to_skull = cls._get_socket_to_skull_rotation(eye_name)
        q_socket_to_skull = cls._rotation_matrix_to_quaternion(R_socket_to_skull)

        position_mm = np.zeros((n_frames, 3), dtype=np.float64)
        orientation_quaternions: list[Quaternion] = []
        basis_x = np.zeros((n_frames, 3), dtype=np.float64)
        basis_y = np.zeros((n_frames, 3), dtype=np.float64)
        basis_z = np.zeros((n_frames, 3), dtype=np.float64)

        socket_local_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        socket_local_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        socket_local_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for i, q_skull in enumerate(skull_quaternions):
            position_mm[i] = q_skull.rotate_vector(eye_position_in_skull) + skull_positions_mm[i]

            q_socket_world = q_skull * q_socket_to_skull
            orientation_quaternions.append(q_socket_world)

            basis_x[i] = q_socket_world.rotate_vector(socket_local_x)
            basis_y[i] = q_socket_world.rotate_vector(socket_local_y)
            basis_z[i] = q_socket_world.rotate_vector(socket_local_z)

        return EyeSocketKinematics(
            eye_name=eye_name,
            timestamps=timestamps.copy(),
            position_mm=position_mm,
            orientation_quaternions=orientation_quaternions,
            basis_x=basis_x,
            basis_y=basis_y,
            basis_z=basis_z,
            position_in_skull_mm=eye_position_in_skull.copy(),
        )

    @classmethod
    def from_pose_data(
        cls,
        timestamps: NDArray[np.float64],
        positions_mm: NDArray[np.float64],
        quaternions: list[Quaternion],
        left_eye_position_in_skull: NDArray[np.float64],
        right_eye_position_in_skull: NDArray[np.float64],
    ) -> "SkullKinematics":
        """
        Compute skull kinematics from raw pose data.

        Args:
            timestamps: (N,) timestamps in seconds
            positions_mm: (N, 3) skull origin positions in mm (world frame)
            quaternions: N Quaternion objects (local → world rotation)
            left_eye_position_in_skull: (3,) left eye position in skull local coords [mm]
            right_eye_position_in_skull: (3,) right eye position in skull local coords [mm]

        Returns:
            SkullKinematics with all computed quantities including eye sockets
        """
        n_frames = len(timestamps)

        # Validate inputs
        if timestamps.ndim != 1:
            raise ValueError(f"timestamps must be 1D, got shape {timestamps.shape}")
        if positions_mm.shape != (n_frames, 3):
            raise ValueError(f"positions_mm shape {positions_mm.shape} must be ({n_frames}, 3)")
        if len(quaternions) != n_frames:
            raise ValueError(
                f"quaternions length {len(quaternions)} must match timestamps length {n_frames}"
            )
        if left_eye_position_in_skull.shape != (3,):
            raise ValueError(f"left_eye_position_in_skull must have shape (3,), got {left_eye_position_in_skull.shape}")
        if right_eye_position_in_skull.shape != (3,):
            raise ValueError(f"right_eye_position_in_skull must have shape (3,), got {right_eye_position_in_skull.shape}")

        # Pre-allocate arrays
        euler_angles_deg = np.zeros((n_frames, 3), dtype=np.float64)
        angular_velocity_world_deg_s = np.zeros((n_frames, 3), dtype=np.float64)
        angular_velocity_local_deg_s = np.zeros((n_frames, 3), dtype=np.float64)
        basis_x = np.zeros((n_frames, 3), dtype=np.float64)
        basis_y = np.zeros((n_frames, 3), dtype=np.float64)
        basis_z = np.zeros((n_frames, 3), dtype=np.float64)

        local_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        local_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        local_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for i, q in enumerate(quaternions):
            roll, pitch, yaw = q.to_euler_xyz()
            euler_angles_deg[i] = np.rad2deg([roll, pitch, yaw])

            basis_x[i] = q.rotate_vector(local_x)
            basis_y[i] = q.rotate_vector(local_y)
            basis_z[i] = q.rotate_vector(local_z)

        for i in range(1, n_frames):
            dt = timestamps[i] - timestamps[i - 1]
            if dt > 0:
                omega_world_rad = cls._compute_angular_velocity_world_rad_s(
                    q_prev=quaternions[i - 1],
                    q_curr=quaternions[i],
                    dt=dt,
                )
                omega_local_rad = cls._transform_omega_world_to_local(
                    omega_world=omega_world_rad,
                    q=quaternions[i],
                )
                angular_velocity_world_deg_s[i] = np.rad2deg(omega_world_rad)
                angular_velocity_local_deg_s[i] = np.rad2deg(omega_local_rad)

        if n_frames > 1:
            angular_velocity_world_deg_s[0] = angular_velocity_world_deg_s[1]
            angular_velocity_local_deg_s[0] = angular_velocity_local_deg_s[1]

        # Compute eye socket kinematics
        left_eye_socket = cls._compute_eye_socket_kinematics(
            eye_name="left",
            eye_position_in_skull=left_eye_position_in_skull,
            timestamps=timestamps,
            skull_positions_mm=positions_mm,
            skull_quaternions=quaternions,
        )
        right_eye_socket = cls._compute_eye_socket_kinematics(
            eye_name="right",
            eye_position_in_skull=right_eye_position_in_skull,
            timestamps=timestamps,
            skull_positions_mm=positions_mm,
            skull_quaternions=quaternions,
        )

        return cls(
            timestamps=timestamps,
            position_mm=positions_mm,
            orientation_quaternions=quaternions,
            euler_angles_deg=euler_angles_deg,
            angular_velocity_world_deg_s=angular_velocity_world_deg_s,
            angular_velocity_local_deg_s=angular_velocity_local_deg_s,
            basis_x=basis_x,
            basis_y=basis_y,
            basis_z=basis_z,
            left_eye_socket=left_eye_socket,
            right_eye_socket=right_eye_socket,
        )

    @classmethod
    def load_from_pose_csv(
        cls,
        csv_path: Path,
        left_eye_position_in_skull: NDArray[np.float64],
        right_eye_position_in_skull: NDArray[np.float64],
    ) -> "SkullKinematics":
        """
        Load skull pose from CSV and compute all kinematics.

        Expected CSV columns:
            - timestamp
            - rotation_r{0-2}_c{0-2} (rotation matrix)
            - quaternion_w, quaternion_x, quaternion_y, quaternion_z
            - translation_x/y/z OR tx/ty/tz OR x/y/z

        Args:
            csv_path: Path to pose CSV file (e.g., skull_pose_data.csv)
            left_eye_position_in_skull: (3,) left eye position in skull local coords [mm]
            right_eye_position_in_skull: (3,) right eye position in skull local coords [mm]

        Returns:
            SkullKinematics with all computed quantities
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Skull pose CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        n_frames = len(df)

        if n_frames == 0:
            raise ValueError(f"Empty CSV file: {csv_path}")

        timestamps = df["timestamp"].values.astype(np.float64)
        positions_mm = np.zeros((n_frames, 3), dtype=np.float64)
        quaternions: list[Quaternion] = []

        for i, (_, row) in enumerate(df.iterrows()):
            R = np.array(
                [
                    [row["rotation_r0_c0"], row["rotation_r0_c1"], row["rotation_r0_c2"]],
                    [row["rotation_r1_c0"], row["rotation_r1_c1"], row["rotation_r1_c2"]],
                    [row["rotation_r2_c0"], row["rotation_r2_c1"], row["rotation_r2_c2"]],
                ],
                dtype=np.float64,
            )
            quaternion = Quaternion(
                w=float(row["quaternion_w"]),
                x=float(row["quaternion_x"]),
                y=float(row["quaternion_y"]),
                z=float(row["quaternion_z"]),
            )
            if not np.allclose(quaternion.to_rotation_matrix(), R, atol=1e-6):
                raise ValueError(f"Rotation matrix does not match quaternion at frame {i}")
            quaternions.append(quaternion)

            if "translation_x" in row.index:
                positions_mm[i] = [row["translation_x"], row["translation_y"], row["translation_z"]]
            elif "tx" in row.index:
                positions_mm[i] = [row["tx"], row["ty"], row["tz"]]
            elif "x" in row.index:
                positions_mm[i] = [row["x"], row["y"], row["z"]]
            else:
                raise ValueError(f"No recognized translation columns in CSV: {csv_path}")

        return cls.from_pose_data(
            timestamps=timestamps,
            positions_mm=positions_mm,
            quaternions=quaternions,
            left_eye_position_in_skull=left_eye_position_in_skull,
            right_eye_position_in_skull=right_eye_position_in_skull,
        )


if __name__ == "__main__":
    skull_pose_csv = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output\skull_and_spine_pose_data.csv"
    )

    # Eye positions from reference geometry (indices: left_eye=1, right_eye=2)
    # These would come from your saved reference_geometry.npy
    left_eye_in_skull = np.array([10.0, 8.0, 5.0], dtype=np.float64)  # placeholder
    right_eye_in_skull = np.array([10.0, -8.0, 5.0], dtype=np.float64)  # placeholder

    print(f"Loading skull pose data from {skull_pose_csv}...")
    kinematics = SkullKinematics.load_from_pose_csv(
        csv_path=skull_pose_csv,
        left_eye_position_in_skull=left_eye_in_skull,
        right_eye_position_in_skull=right_eye_in_skull,
    )
    print(f"  Loaded {kinematics.n_frames} frames")

    # Save skull kinematics CSV
    output_dir = skull_pose_csv.parent
    kinematics.to_dataframe().to_csv(output_dir / "skull_kinematics.csv", index=False)
    kinematics.left_eye_socket.to_dataframe().to_csv(output_dir / "left_eye_socket_kinematics.csv", index=False)
    kinematics.right_eye_socket.to_dataframe().to_csv(output_dir / "right_eye_socket_kinematics.csv", index=False)

    print(f"Saved kinematics to {output_dir}")
    print(f"\nSummary:")
    print(f"  Duration: {kinematics.timestamps[-1] - kinematics.timestamps[0]:.2f} s")
    print(f"  Head yaw range: {kinematics.head_yaw_deg.min():.1f}° to {kinematics.head_yaw_deg.max():.1f}°")
    print(f"  Max yaw velocity: {np.abs(kinematics.head_yaw_velocity_deg_s).max():.1f} °/s")