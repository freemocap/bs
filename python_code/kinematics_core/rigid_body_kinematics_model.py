from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from python_code.kinematics_core.keypoint_trajectories import KeypointTrajectories
from python_code.kinematics_core.timeseries_model import Timeseries
from python_code.kinematics_core.vector3_trajectory_model import Vector3Trajectory
from python_code.kinematics_core.quaternion_trajectory_model import QuaternionTrajectory
from python_code.kinematics_core.angular_velocity_trajectory_model import AngularVelocityTrajectory
from python_code.kinematics_core.rigid_body_state_model import RigidBodyState
from python_code.kinematics_core.derivative_helpers import compute_velocity
from python_code.kinematics_core.kinematics_serialization import save_kinematics, \
    kinematics_to_tidy_dataframe
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry


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
    name: str
    reference_geometry: ReferenceGeometry
    timestamps: NDArray[np.float64]  # (N,)

    # Position and velocity
    position_xyz: NDArray[np.float64]  # (N, 3) mm
    velocity_xyz: NDArray[np.float64]  # (N, 3) mm/s

    # Orientation
    orientations: QuaternionTrajectory
    # Angular velocity
    angular_velocity_global: NDArray[np.float64]  # (N, 3) rad/s world frame
    angular_velocity_local: NDArray[np.float64]  # (N, 3) rad/s body frame

    keypoint_trajectories: KeypointTrajectories

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
        Construct from basic pose arrays, computing velocities automatically.

        Args:
            name: Name of the rigid body
            reference_geometry: The reference geometry
            timestamps: (N,) timestamps in seconds
            position_xyz: (N, 3) positions in mm
            quaternions_wxyz: (N, 4) quaternions as [w, x, y, z]
        """
        # Convert quaternions
        orientations = QuaternionTrajectory.from_wxyz_array(
            name=f"{name}_orientation",
            timestamps=timestamps,
            quaternions_wxyz=quaternions_wxyz,
        )

        # Compute linear velocity
        velocity_xyz = compute_velocity(position_xyz, timestamps)

        # Compute angular velocity
        angular_velocity_global, angular_velocity_local = orientations.compute_angular_velocity()

        # Compute keypoint trajectories in WORLD frame
        # Step 1: Rotate local positions by orientation (N, M, 3)
        local_positions = reference_geometry.marker_local_positions_array
        rotated_keypoints = orientations.rotate_vectors(local_positions)

        # Step 2: Add body origin translation to get world positions
        # (N, M, 3) + (N, 1, 3) -> (N, M, 3)
        world_keypoints = rotated_keypoints + position_xyz[:, np.newaxis, :]

        keypoint_trajectories = KeypointTrajectories(
            keypoint_names=tuple(reference_geometry.markers.keys()),
            timestamps=timestamps,
            trajectories_fr_id_xyz=world_keypoints,
        )

        return cls(
            name=name,
            reference_geometry=reference_geometry,
            timestamps=timestamps,
            position_xyz=position_xyz,
            velocity_xyz=velocity_xyz,
            orientations=orientations,
            angular_velocity_global=angular_velocity_global,
            angular_velocity_local=angular_velocity_local,
            keypoint_trajectories=keypoint_trajectories,
        )

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

    def get_state_at_frame(self, idx: int) -> RigidBodyState:
        """Get the pose at frame index."""
        return RigidBodyState(
            reference_geometry=self.reference_geometry,
            timestamp=float(self.timestamps[idx]),
            position=self.position_xyz[idx],
            velocity=self.velocity_xyz[idx],
            orientation=self.orientations[idx],
            angular_velocity_global=self.angular_velocity_global[idx],
            angular_velocity_local=self.angular_velocity_local[idx],
        )

    def __getitem__(self, idx: int) -> RigidBodyState:
        """Get the pose at frame index."""
        return self.get_state_at_frame(idx)

    def __iter__(self) -> Iterator[RigidBodyState]:
        """Iterate over all poses."""
        for i in range(self.n_frames):
            yield self.get_state_at_frame(i)

    def __len__(self) -> int:
        return self.n_frames

    # -------------------------------------------------------------------------
    # Vertical slicing: get Trajectories
    # -------------------------------------------------------------------------

    @property
    def position_trajectory(self) -> Vector3Trajectory:
        """Position trajectory (x, y, z over time)."""
        return Vector3Trajectory(name="position", timestamps=self.timestamps, values=self.position_xyz)

    @property
    def velocity_trajectory(self) -> Vector3Trajectory:
        """Velocity trajectory (vx, vy, vz over time)."""
        return Vector3Trajectory(name="velocity", timestamps=self.timestamps, values=self.velocity_xyz)


    @property
    def angular_velocity_trajectory(self) -> AngularVelocityTrajectory:
        """Angular velocity trajectory (global and local over time)."""
        return AngularVelocityTrajectory(
            name="angular_velocity",
            timestamps=self.timestamps,
            global_xyz=self.angular_velocity_global,
            local_xyz=self.angular_velocity_local,
        )


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
        return self.orientations.roll

    @property
    def pitch(self) -> Timeseries:
        return self.orientations.pitch

    @property
    def yaw(self) -> Timeseries:
        return self.orientations.yaw

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



    def save_to_disk(
            self,
            output_directory: "Path",
            include_keypoints: bool = True,
    ) -> tuple["Path", "Path"]:
        """
        Save kinematics data to disk.

        Creates two files:
            {name}_reference_geometry.json - Static reference geometry
            {name}_kinematics.csv - Tidy-format kinematic data

        Args:
            output_directory: Directory to save files to (created if needed)
            include_keypoints: If True, CSV includes keypoint trajectory rows

        Returns:
            Tuple of (reference_geometry_path, kinematics_csv_path)
        """
        return save_kinematics(
            kinematics=self,
            output_directory=output_directory,
        )

    def to_dataframe(
            self,
            include_keypoints: bool = True,
    ) -> "pl.DataFrame":
        """
        Export kinematics to a tidy-format polars DataFrame.

        Tidy format: one row per (frame, trajectory, component) observation.

        Args:
            include_keypoints: If True, include keypoint trajectory rows

        Returns:
            Tidy-format polars DataFrame
        """
        return kinematics_to_tidy_dataframe(kinematics=self)

    def resample(self, target_timestamps: NDArray[np.float64]) -> "RigidBodyKinematics":
        from python_code.kinematics_core.resample_helpers import resample_kinematics
        return resample_kinematics(kinematics=self, target_timestamps=target_timestamps)

    def resample_to_uniform_rate(self, target_fps: float) -> "RigidBodyKinematics":
        from python_code.kinematics_core.resample_helpers import resample_kinematics_to_uniform_rate
        return resample_kinematics_to_uniform_rate(kinematics=self, target_fps=target_fps)