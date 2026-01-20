"""
Ferret Eye Kinematics Models
============================

COORDINATE SYSTEM (at rest, right-handed):
    Origin: Eye center [0, 0, 0]
    +Z: Rest gaze direction (anterior, toward pupil) - "north pole"
    +Y: Superior (up)
    +X: Subject's left (computed via Y × Z)

This means +X points to the subject's LEFT for BOTH eyes.
The anatomical interpretation differs:
    - RIGHT eye: +X = medial (toward nose)
    - LEFT eye: +X = lateral (away from nose)

Use the anatomical accessors (adduction_angle, etc.) for consistent
anatomical meaning across eyes.

At rest position:
    - Pupil at [0, 0, +eye_radius]
    - Quaternion = [1, 0, 0, 0] (identity)
    - Gaze direction = [0, 0, 1]
"""
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python_code.ferret_gaze.eye_kinematics.ferret_eyeball_reference_geometry import (
    create_eyeball_reference_geometry,
    NUM_PUPIL_POINTS,
    FERRET_EYE_RADIUS_MM,
    DEFAULT_FERRET_EYE_PUPIL_RADIUS_MM,
    DEFAULT_FERRET_EYE_PUPIL_ECCENTRICITY,
)
from python_code.kinematics_core.angular_velocity_trajectory_model import AngularVelocityTrajectory
from python_code.kinematics_core.quaternion_trajectory_model import QuaternionTrajectory
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.kinematics_core.timeseries_model import Timeseries
from python_code.kinematics_core.vector3_trajectory_model import Vector3Trajectory


class SocketLandmarks(BaseModel):
    """
    Anatomical landmarks of the eye socket/opening.

    These are FIXED relative to the skull - they do NOT rotate with the eyeball.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    timestamps: NDArray[np.float64]
    tear_duct_mm: NDArray[np.float64]
    outer_eye_mm: NDArray[np.float64]

    @model_validator(mode="after")
    def validate_shapes(self) -> "SocketLandmarks":
        n_frames = len(self.timestamps)
        if self.tear_duct_mm.shape != (n_frames, 3):
            raise ValueError(f"tear_duct_mm shape {self.tear_duct_mm.shape} != ({n_frames}, 3)")
        if self.outer_eye_mm.shape != (n_frames, 3):
            raise ValueError(f"outer_eye_mm shape {self.outer_eye_mm.shape} != ({n_frames}, 3)")
        return self

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def tear_duct_trajectory(self) -> Vector3Trajectory:
        return Vector3Trajectory(name="tear_duct", timestamps=self.timestamps, values=self.tear_duct_mm)

    @property
    def outer_eye_trajectory(self) -> Vector3Trajectory:
        return Vector3Trajectory(name="outer_eye", timestamps=self.timestamps, values=self.outer_eye_mm)

    @property
    def eye_opening_width_mm(self) -> Timeseries:
        widths = np.linalg.norm(self.outer_eye_mm - self.tear_duct_mm, axis=1)
        return Timeseries(name="eye_opening_width", timestamps=self.timestamps, values=widths)

    @property
    def midpoint_mm(self) -> NDArray[np.float64]:
        return (self.tear_duct_mm + self.outer_eye_mm) / 2.0

    def get_mean_positions(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.mean(self.tear_duct_mm, axis=0), np.mean(self.outer_eye_mm, axis=0)


class FerretEyeKinematics(BaseModel):
    """
    Complete eye kinematics with eyeball (rigid body) and socket landmarks.

    COORDINATE SYSTEM (right-handed, Z+ = gaze):
        Origin: Eye center [0, 0, 0]
        +Z: Rest gaze direction (pupil at [0, 0, R])
        +Y: Superior (up)
        +X: Subject's left
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str = Field(min_length=1)
    eyeball: RigidBodyKinematics
    socket_landmarks: SocketLandmarks

    @model_validator(mode="after")
    def validate_timestamps_match(self) -> "FerretEyeKinematics":
        if self.eyeball.n_frames != self.socket_landmarks.n_frames:
            raise ValueError(
                f"Eyeball frames ({self.eyeball.n_frames}) != "
                f"socket landmark frames ({self.socket_landmarks.n_frames})"
            )
        if not np.allclose(self.eyeball.timestamps, self.socket_landmarks.timestamps, rtol=1e-9):
            raise ValueError("Eyeball and socket landmark timestamps don't match")
        return self

    @classmethod
    def from_pose_data(
        cls,
        eye_name: Literal["left_eye", "right_eye"],
        eye_data_csv_path: str,
        timestamps: NDArray[np.float64],
        quaternions_wxyz: NDArray[np.float64],
        tear_duct_mm: NDArray[np.float64],
        outer_eye_mm: NDArray[np.float64],
        rest_gaze_direction_camera: NDArray[np.float64],
        camera_to_eye_rotation: NDArray[np.float64],
        eyeball_radius_mm: float = FERRET_EYE_RADIUS_MM,
        pupil_radius_mm: float = DEFAULT_FERRET_EYE_PUPIL_RADIUS_MM,
        pupil_eccentricity: float = DEFAULT_FERRET_EYE_PUPIL_ECCENTRICITY,
    ) -> "FerretEyeKinematics":
        eyeball_geometry = create_eyeball_reference_geometry(
            eye_radius_mm=eyeball_radius_mm,
            default_pupil_radius_mm=pupil_radius_mm,
            default_pupil_eccentricity=pupil_eccentricity,
        )

        n_frames = len(timestamps)
        position_xyz = np.zeros((n_frames, 3), dtype=np.float64)

        eyeball = RigidBodyKinematics.from_pose_arrays(
            name=eye_name,
            reference_geometry=eyeball_geometry,
            timestamps=timestamps,
            position_xyz=position_xyz,
            quaternions_wxyz=quaternions_wxyz,
        )

        socket_landmarks = SocketLandmarks(
            timestamps=timestamps,
            tear_duct_mm=tear_duct_mm,
            outer_eye_mm=outer_eye_mm,
        )

        return cls(name=eye_name, eyeball=eyeball, socket_landmarks=socket_landmarks)

    @classmethod
    def load_from_directory(cls, eye_name: str, input_directory: str | Path) -> "FerretEyeKinematics":
        from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import (
            load_ferret_eye_kinematics_from_directory,
        )
        return load_ferret_eye_kinematics_from_directory(
            input_directory=Path(input_directory),
            eye_name=eye_name,
        )

    @classmethod
    def load_from_data_paths(
        cls,
        eyeball_kinematics_path: str | Path,
        reference_geometry_path: str | Path,
    ) -> "FerretEyeKinematics":
        from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import (
            load_ferret_eye_kinematics,
        )
        return load_ferret_eye_kinematics(
            kinematics_csv_path=Path(eyeball_kinematics_path),
            reference_geometry_path=Path(reference_geometry_path),
        )

    @classmethod
    def calculate_from_trajectories(
        cls,
        eye_name: Literal["left_eye", "right_eye"],
        eye_trajectories_csv_path: str | Path,
        eye_camera_distance_mm: float,
    ) -> "FerretEyeKinematics":
        from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import (
            process_ferret_eye_data,
        )
        (timestamps, quaternions_wxyz, tear_duct_mm, outer_eye_mm,
         rest_gaze_direction_camera, camera_to_eye_rotation) = process_ferret_eye_data(
            eye_name=eye_name,
            eye_trajectories_csv_path=Path(eye_trajectories_csv_path),
            eye_camera_distance_mm=eye_camera_distance_mm,
        )
        return cls.from_pose_data(
            eye_name=eye_name,
            eye_data_csv_path=str(eye_trajectories_csv_path),
            timestamps=timestamps,
            quaternions_wxyz=quaternions_wxyz,
            tear_duct_mm=tear_duct_mm,
            outer_eye_mm=outer_eye_mm,
            rest_gaze_direction_camera=rest_gaze_direction_camera,
            camera_to_eye_rotation=camera_to_eye_rotation,
        )

    def save_to_disk(self, output_directory: str | Path) -> None:
        from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import (
            save_ferret_eye_kinematics,
        )
        save_ferret_eye_kinematics(kinematics=self, output_directory=Path(output_directory))

    # =========================================================================
    # Convenience properties
    # =========================================================================

    @property
    def timestamps(self) -> NDArray[np.float64]:
        return self.eyeball.timestamps

    @property
    def n_frames(self) -> int:
        return self.eyeball.n_frames

    @property
    def duration_seconds(self) -> float:
        return self.eyeball.duration

    @property
    def eye_side(self) -> Literal["left", "right"]:
        return "right" if "right" in self.name.lower() else "left"

    # =========================================================================
    # Eyeball kinematics accessors
    # =========================================================================

    @property
    def orientations(self) -> QuaternionTrajectory:
        return self.eyeball.orientations

    @property
    def quaternions_wxyz(self) -> NDArray[np.float64]:
        return self.eyeball.quaternions_wxyz

    @property
    def angular_velocity_trajectory(self) -> AngularVelocityTrajectory:
        return self.eyeball.angular_velocity_trajectory

    @property
    def angular_velocity_global(self) -> NDArray[np.float64]:
        return self.eyeball.angular_velocity_global

    @property
    def angular_velocity_local(self) -> NDArray[np.float64]:
        return self.eyeball.angular_velocity_local

    @property
    def angular_speed(self) -> Timeseries:
        return self.eyeball.angular_speed

    @property
    def roll(self) -> Timeseries:
        return self.eyeball.roll

    @property
    def pitch(self) -> Timeseries:
        return self.eyeball.pitch

    @property
    def yaw(self) -> Timeseries:
        return self.eyeball.yaw

    # =========================================================================
    # Gaze direction (Z+ = gaze convention)
    # =========================================================================

    @property
    def gaze_directions(self) -> NDArray[np.float64]:
        """
        Gaze direction (unit vector) over time.

        This is the +Z axis of the eyeball rotated by the current orientation.
        At rest, gaze = [0, 0, 1] (north pole).

        Returns:
            (N, 3) array of unit vectors
        """
        rest_gaze = np.array([0.0, 0.0, 1.0])
        return self.eyeball.orientations.rotate_vector(rest_gaze)

    @property
    def azimuth_radians(self) -> NDArray[np.float64]:
        """
        Azimuth angle (horizontal gaze direction) in radians.

        Positive = looking toward +X (subject's left)
        Zero = looking straight ahead (+Z)
        """
        gaze = self.gaze_directions
        return np.arctan2(gaze[:, 0], gaze[:, 2])

    @property
    def elevation_radians(self) -> NDArray[np.float64]:
        """
        Elevation angle (vertical gaze direction) in radians.

        Positive = looking up (+Y)
        Zero = looking straight ahead
        """
        gaze = self.gaze_directions
        horizontal = np.sqrt(gaze[:, 0] ** 2 + gaze[:, 2] ** 2)
        return np.arctan2(gaze[:, 1], horizontal)

    @property
    def azimuth_degrees(self) -> NDArray[np.float64]:
        return np.degrees(self.azimuth_radians)

    @property
    def elevation_degrees(self) -> NDArray[np.float64]:
        return np.degrees(self.elevation_radians)

    # =========================================================================
    # Anatomical Rotation Angles
    # =========================================================================
    #
    # Coordinate System: +Z = gaze, +Y = up, +X = subject's left
    #
    # Anatomical mapping:
    #   - Horizontal gaze (adduction/abduction): rotation in XZ plane
    #     - +X = subject's left
    #     - RIGHT eye: +X = medial, so positive azimuth = adduction
    #     - LEFT eye: +X = lateral, so positive azimuth = abduction
    #   - Vertical gaze (elevation/depression): rotation toward Y
    #   - Torsion (intorsion/extorsion): rotation around Z (gaze axis)
    #
    # =========================================================================

    @property
    def _anatomical_horizontal_sign(self) -> float:
        """
        Sign multiplier for adduction angle.

        +X points to subject's left.
        - RIGHT eye: +X = medial → positive azimuth = adduction → sign = +1
        - LEFT eye: +X = lateral → positive azimuth = abduction → sign = -1
        """
        return 1.0 if self.eye_side == "right" else -1.0

    @property
    def _anatomical_torsion_sign(self) -> float:
        """
        Sign multiplier for torsion (extorsion positive convention).

        Torsion = rotation around +Z (gaze axis).
        Right-hand rule: +rotation tilts +Y toward -X (top of eye tilts toward subject's right).

        - RIGHT eye: top tilts right = lateral = EXTORSION → sign = +1
        - LEFT eye: top tilts right = medial = INTORSION → sign = -1
        """
        return 1.0 if self.eye_side == "right" else -1.0

    @property
    def adduction_angle(self) -> Timeseries:
        """
        Horizontal gaze angle with consistent anatomical meaning.

        Positive = Adduction (gaze toward nose/medial)
        Negative = Abduction (gaze away from nose/lateral)
        """
        return Timeseries(
            name="adduction",
            timestamps=self.timestamps,
            values=self._anatomical_horizontal_sign * self.azimuth_radians,
        )

    @property
    def elevation_angle(self) -> Timeseries:
        """
        Vertical gaze angle.

        Positive = Elevation (gaze up)
        Negative = Depression (gaze down)
        """
        return Timeseries(
            name="elevation",
            timestamps=self.timestamps,
            values=self.elevation_radians,
        )

    @property
    def torsion_angle(self) -> Timeseries:
        """
        Torsional rotation around the gaze axis.

        Positive = Extorsion (top of eye rotates away from nose)
        Negative = Intorsion (top of eye rotates toward nose)

        NOTE: Torsion calculations may not be fully reliable.
        """
        return Timeseries(
            name="torsion",
            timestamps=self.timestamps,
            values=self._anatomical_torsion_sign * self.yaw.values,
        )

    @property
    def intorsion_angle(self) -> Timeseries:
        """Torsion with intorsion-positive convention."""
        return Timeseries(
            name="intorsion",
            timestamps=self.timestamps,
            values=-self._anatomical_torsion_sign * self.yaw.values,
        )

    # =========================================================================
    # Anatomical Angular Velocity
    # =========================================================================

    @property
    def adduction_velocity(self) -> Timeseries:
        """Angular velocity of adduction (rad/s)."""
        return Timeseries(
            name="adduction_velocity",
            timestamps=self.timestamps,
            values=self._anatomical_horizontal_sign * self.angular_velocity_global[:, 1],
        )

    @property
    def elevation_velocity(self) -> Timeseries:
        """Angular velocity of elevation (rad/s).

        Positive = looking up faster (increasing elevation).
        Negative = looking down faster (decreasing elevation).

        Note: Negation is required because positive ωx (rotation around X-axis)
        causes gaze to rotate toward -Y (downward), decreasing elevation.
        """
        return Timeseries(
            name="elevation_velocity",
            timestamps=self.timestamps,
            values=-self.angular_velocity_global[:, 0],
        )

    @property
    def torsion_velocity(self) -> Timeseries:
        """Angular velocity of torsion (rad/s)."""
        return Timeseries(
            name="torsion_velocity",
            timestamps=self.timestamps,
            values=self._anatomical_torsion_sign * self.angular_velocity_global[:, 2],
        )

    # =========================================================================
    # Pupil position accessors
    # =========================================================================

    @property
    def pupil_center_trajectory(self) -> NDArray[np.float64]:
        return self.eyeball.keypoint_trajectories["pupil_center"]

    def get_pupil_point_trajectory(self, point_index: int) -> NDArray[np.float64]:
        if not 1 <= point_index <= NUM_PUPIL_POINTS:
            raise ValueError(f"point_index must be 1-{NUM_PUPIL_POINTS}, got {point_index}")
        return self.eyeball.keypoint_trajectories[f"p{point_index}"]

    @property
    def pupil_points_trajectories(self) -> NDArray[np.float64]:
        points = [self.eyeball.keypoint_trajectories[f"p{i + 1}"] for i in range(NUM_PUPIL_POINTS)]
        return np.stack(points, axis=1)

    # =========================================================================
    # Socket landmark accessors
    # =========================================================================

    @property
    def tear_duct_mm(self) -> NDArray[np.float64]:
        return self.socket_landmarks.tear_duct_mm

    @property
    def outer_eye_mm(self) -> NDArray[np.float64]:
        return self.socket_landmarks.outer_eye_mm