"""
Ferret Skull Kinematics Analysis

Computes skull position, orientation (Euler angles), angular velocity
in both world frame and skull-local frame (roll, pitch, yaw rates in degrees/second),
and orthonormal basis vectors of the skull coordinate frame.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_code.ferret_gaze.quaternion_helper import Quaternion


@dataclass
class SkullKinematics:
    """Skull kinematics data."""

    timestamps: NDArray[np.float64]  # (N,) seconds
    position: NDArray[np.float64]  # (N, 3) mm, world frame
    orientation_quaternions: list[Quaternion]  # list of N Quaternion objects defining skull orientation
    euler_angles_deg: NDArray[np.float64]  # (N, 3) degrees [roll, pitch, yaw]
    angular_velocity_world_deg_s: NDArray[np.float64]  # (N, 3) deg/s, world frame [x, y, z]
    angular_velocity_local_deg_s: NDArray[np.float64]  # (N, 3) deg/s, skull-local [roll_rate, pitch_rate, yaw_rate]
    # Basis vectors transformed to world frame (origin at skull position, unit length)
    basis_x: NDArray[np.float64]  # (N, 3) skull's x-axis direction in world frame
    basis_y: NDArray[np.float64]  # (N, 3) skull's y-axis direction in world frame
    basis_z: NDArray[np.float64]  # (N, 3) skull's z-axis direction in world frame

    def to_dataframe(self) -> pd.DataFrame:
        """Convert SkullKinematics to a pandas DataFrame."""
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "position_x_mm": self.position[:, 0],
            "position_y_mm": self.position[:, 1],
            "position_z_mm": self.position[:, 2],
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
            "omega_local_roll_deg_s": self.angular_velocity_local_deg_s[:, 0],
            "omega_local_pitch_deg_s": self.angular_velocity_local_deg_s[:, 1],
            "omega_local_yaw_deg_s": self.angular_velocity_local_deg_s[:, 2],
            # Basis vectors (unit vectors in world frame, rotated by skull orientation)
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


def compute_omega_world(q1: Quaternion, q2: Quaternion, dt: float) -> NDArray[np.float64]:
    """Compute angular velocity in world frame from two quaternions."""
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    delta_q = q2 * q1.inverse()
    if delta_q.w < 0:
        delta_q = Quaternion(w=-delta_q.w, x=-delta_q.x, y=-delta_q.y, z=-delta_q.z)

    axis, angle = delta_q.to_axis_angle()
    return (angle / dt) * axis


def omega_world_to_local(omega_world: NDArray[np.float64], q: Quaternion) -> NDArray[np.float64]:
    """Transform angular velocity from world frame to local (body-attached) frame."""
    return q.inverse().rotate_vector(omega_world)


def load_skull_pose(csv_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], list[Quaternion]]:
    """Load skull 6DoF pose from CSV.

    Returns:
        timestamps: (N,) array of timestamps in seconds
        positions: (N, 3) array of positions in mm
        quaternions: list of N Quaternion objects
    """
    df = pd.read_csv(csv_path)
    n_frames = len(df)

    timestamps = df["timestamp"].values.astype(np.float64)
    positions = np.zeros((n_frames, 3), dtype=np.float64)
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
        quaternion = Quaternion(w=row["quaternion_w"],
                                x=row["quaternion_x"],
                                y=row["quaternion_y"],
                                z=row["quaternion_z"])
        if not np.allclose(quaternion.to_rotation_matrix(), R, atol=1e-6):
            raise ValueError(f"Rotation matrix does not match quaternion at frame {i}")
        quaternions.append(quaternion)

        if "translation_x" in row.index:
            positions[i] = [row["translation_x"], row["translation_y"], row["translation_z"]]
        elif "tx" in row.index:
            positions[i] = [row["tx"], row["ty"], row["tz"]]
        elif "x" in row.index:
            positions[i] = [row["x"], row["y"], row["z"]]
        else:
            raise ValueError("No recognized translation columns in CSV")

    return timestamps, positions, quaternions


def compute_skull_kinematics(
    timestamps: NDArray[np.float64],
    positions: NDArray[np.float64],
    quaternions: list[Quaternion],
) -> SkullKinematics:
    """Compute skull kinematics from pose data.

    Args:
        timestamps: (N,) array of timestamps in seconds
        positions: (N, 3) array of skull positions in mm
        quaternions: list of N Quaternion objects representing skull orientation

    Returns:
        SkullKinematics with position, orientation, angular velocity (world and local), and basis vectors
    """
    n_frames = len(timestamps)
    if len(positions) != n_frames or len(quaternions) != n_frames:
        raise ValueError("timestamps, positions, and quaternions must have the same length")

    euler_angles_deg = np.zeros((n_frames, 3), dtype=np.float64)
    angular_velocity_world_deg_s = np.zeros((n_frames, 3), dtype=np.float64)
    angular_velocity_local_deg_s = np.zeros((n_frames, 3), dtype=np.float64)

    # Canonical basis vectors
    canonical_basis = np.eye(3, dtype=np.float64)  # rows are x_hat, y_hat, z_hat
    basis_x = np.zeros((n_frames, 3), dtype=np.float64)
    basis_y = np.zeros((n_frames, 3), dtype=np.float64)
    basis_z = np.zeros((n_frames, 3), dtype=np.float64)

    for frame_number, q in enumerate(quaternions):
        roll, pitch, yaw = q.to_euler_xyz()
        euler_angles_deg[frame_number] = np.rad2deg([roll, pitch, yaw])

        # Rotate canonical basis vectors by skull orientation
        basis_x[frame_number] = q.rotate_vector(canonical_basis[0])
        basis_y[frame_number] = q.rotate_vector(canonical_basis[1])
        basis_z[frame_number] = q.rotate_vector(canonical_basis[2])

    # Compute angular velocity using finite differences
    for frame_number in range(1, n_frames):
        dt = timestamps[frame_number] - timestamps[frame_number - 1]
        if dt > 0:
            omega_world = compute_omega_world(quaternions[frame_number - 1], quaternions[frame_number], dt)
            omega_local = omega_world_to_local(omega_world, quaternions[frame_number])
            angular_velocity_world_deg_s[frame_number] = np.rad2deg(omega_world)
            angular_velocity_local_deg_s[frame_number] = np.rad2deg(omega_local)

    # First frame: copy from second frame
    if n_frames > 1:
        angular_velocity_world_deg_s[0] = angular_velocity_world_deg_s[1]
        angular_velocity_local_deg_s[0] = angular_velocity_local_deg_s[1]

    return SkullKinematics(
        timestamps=timestamps,
        position=positions,
        orientation_quaternions=quaternions,
        euler_angles_deg=euler_angles_deg,
        angular_velocity_world_deg_s=angular_velocity_world_deg_s,
        angular_velocity_local_deg_s=angular_velocity_local_deg_s,
        basis_x=basis_x,
        basis_y=basis_y,
        basis_z=basis_z,
    )

if __name__ == "__main__":
    from python_code.ferret_gaze.visualization.ferret_gaze_rerun import load_trajectory_data, run_visualization

    # Paths - edit these
    skull_pose_csv = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output\rotation_translation_data.csv")
    trajectory_csv = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output\tidy_trajectory_data.csv")

    print(f"Loading skull pose data from {skull_pose_csv}...")
    timestamps, positions, quaternions = load_skull_pose(skull_pose_csv)
    print(f"  Loaded {len(timestamps)} frames")

    print("Computing skull kinematics...")
    hk = compute_skull_kinematics(
        timestamps=timestamps,
        positions=positions,
        quaternions=quaternions,
    )

    # Save CSV
    output_path = skull_pose_csv.parent / "skull_kinematics.csv"
    df = hk.to_dataframe()
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Load trajectory and run visualization
    trajectory_data = load_trajectory_data(trajectory_csv)
    run_visualization(hk=hk, trajectory_data=trajectory_data)
