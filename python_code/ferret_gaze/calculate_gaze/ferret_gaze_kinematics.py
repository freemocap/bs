from pathlib import Path
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python_code.ferret_gaze.eye_kinematics.ferret_eyeball_reference_geometry import NUM_PUPIL_POINTS
from python_code.kinematics_core.angular_velocity_trajectory_model import AngularVelocityTrajectory
from python_code.kinematics_core.angular_acceleration_trajectory_model import AngularAccelerationTrajectory
from python_code.kinematics_core.quaternion_trajectory_model import QuaternionTrajectory
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.kinematics_core.timeseries_model import Timeseries


class FerretGazeKinematics(BaseModel):
    """
    COORDINATE SYSTEM:
        Origin: Eye center [0, 0, 0]
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str = Field(min_length=1)
    kinematics: RigidBodyKinematics
    tracked_pupil_center_world_mm: NDArray[np.float64] | None = None  # (N, 3)
    tracked_pupil_points_world_mm: NDArray[np.float64] | None = None  # (N, 8, 3)

    @model_validator(mode="after")
    def validate_tracked_pupil_shapes(self) -> "FerretGazeKinematics":
        n_frames = self.n_frames
        if self.tracked_pupil_center_world_mm is not None:
            if self.tracked_pupil_center_world_mm.shape != (n_frames, 3):
                raise ValueError(
                    f"tracked_pupil_center_world_mm shape {self.tracked_pupil_center_world_mm.shape} "
                    f"!= ({n_frames}, 3)"
                )
        if self.tracked_pupil_points_world_mm is not None:
            if self.tracked_pupil_points_world_mm.shape != (n_frames, NUM_PUPIL_POINTS, 3):
                raise ValueError(
                    f"tracked_pupil_points_world_mm shape {self.tracked_pupil_points_world_mm.shape} "
                    f"!= ({n_frames}, {NUM_PUPIL_POINTS}, 3)"
                )
        return self

    @classmethod
    def from_rigid_body_kinematics(
        cls,
        rigid_body_kinematics: RigidBodyKinematics,
        tracked_pupil_center_world_mm: NDArray[np.float64] | None = None,
        tracked_pupil_points_world_mm: NDArray[np.float64] | None = None,
    ) -> "FerretGazeKinematics":
        return cls(
            name=rigid_body_kinematics.name,
            kinematics=rigid_body_kinematics,
            tracked_pupil_center_world_mm=tracked_pupil_center_world_mm,
            tracked_pupil_points_world_mm=tracked_pupil_points_world_mm,
        )


    @classmethod
    def load_from_directory(cls, eye_name: str, input_directory: str | Path) -> "FerretGazeKinematics":
        """Load FerretEyeKinematics from disk."""
        if eye_name not in  ['left_gaze', 'right_gaze']:
            raise ValueError(
                f"Unexpected eye_name '{eye_name}'. Expected 'left_gaze' or 'right_gaze'."
            )
        reference_geometry_path = Path(input_directory) / f"{eye_name}_reference_geometry.json"
        kinematics_csv_path = Path(input_directory) / f"{eye_name}_kinematics.csv"

        return cls.load_from_data_paths(
            gaze_kinematics_path=kinematics_csv_path,
            reference_geometry_path=reference_geometry_path
        )

    @classmethod
    def load_from_data_paths(
        cls,
        gaze_kinematics_path: str | Path,
        reference_geometry_path: str | Path,
    ) -> "FerretGazeKinematics":
        eye_name = "left_gaze" if "left" in Path(gaze_kinematics_path).name else "right_gaze"

        kinematics = RigidBodyKinematics.load_from_disk(
            kinematics_csv_path=Path(gaze_kinematics_path),
            reference_geometry_json_path=Path(reference_geometry_path),
        )

        return FerretGazeKinematics.from_rigid_body_kinematics(kinematics)


    def _build_gaze_kinematics_dataframe(self) -> "pl.DataFrame":
        from python_code.kinematics_core.kinematics_serialization import (
            kinematics_to_tidy_dataframe,
            _build_vector_chunk,
        )
        import polars as pl

        # Build base dataframe from RigidBodyKinematics
        df = kinematics_to_tidy_dataframe(self.kinematics)

        # Append horizontal and vertical degree trajectories
        frame_indices = np.arange(self.n_frames, dtype=np.int64)
        extra_chunks = [
            _build_vector_chunk(
                frame_indices=frame_indices,
                timestamps=self.timestamps,
                values=self.horizontal_degrees[:, np.newaxis],
                trajectory_name="gaze_angle",
                component_names=["horizontal"],
                units="degrees",
            ),
            _build_vector_chunk(
                frame_indices=frame_indices,
                timestamps=self.timestamps,
                values=self.vertical_degrees[:, np.newaxis],
                trajectory_name="gaze_angle",
                component_names=["vertical"],
                units="degrees",
            ),
        ]

        return pl.concat([df] + extra_chunks).sort(by="frame")

    def _build_gaze_kinematics_parquet_dataframe(self) -> "pl.DataFrame":
        """
        Build the dataframe written to the parquet output.

        Includes everything in the CSV dataframe plus the actual tracked pupil
        positions in world space (real per-frame detections, not idealized
        geometry, already projected to world by the caller - see
        project_tracked_pupil_to_world()). These are parquet-only, not written
        to the CSV.
        """
        from python_code.kinematics_core.kinematics_serialization import _build_vector_chunk
        import polars as pl

        df = self._build_gaze_kinematics_dataframe()

        frame_indices = np.arange(self.n_frames, dtype=np.int64)
        tracked_pupil_chunks = []

        if self.tracked_pupil_center_world_mm is not None:
            tracked_pupil_chunks.append(_build_vector_chunk(
                frame_indices=frame_indices,
                timestamps=self.timestamps,
                values=self.tracked_pupil_center_world_mm,
                trajectory_name="tracked_pupil__pupil_center",
                component_names=["x", "y", "z"],
                units="mm",
            ))
        if self.tracked_pupil_points_world_mm is not None:
            for point_index in range(NUM_PUPIL_POINTS):
                tracked_pupil_chunks.append(_build_vector_chunk(
                    frame_indices=frame_indices,
                    timestamps=self.timestamps,
                    values=self.tracked_pupil_points_world_mm[:, point_index, :],
                    trajectory_name=f"tracked_pupil__p{point_index + 1}",
                    component_names=["x", "y", "z"],
                    units="mm",
                ))

        if not tracked_pupil_chunks:
            return df

        return pl.concat([df] + tracked_pupil_chunks).sort(by="frame")

    def save_to_disk(self, output_directory: str | Path) -> None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        # Save reference geometry JSON (same as base save_to_disk)
        reference_geometry_path = output_directory / f"{self.name}_reference_geometry.json"
        self.kinematics.reference_geometry.to_json_file(path=reference_geometry_path)

        # Save kinematics CSV and parquet, built independently: the parquet
        # additionally carries the tracked pupil world-space trajectories,
        # which are not written to the CSV.
        kinematics_csv_path = output_directory / f"{self.name}_kinematics.csv"
        kinematics_parquet_path = output_directory / f"{self.name}_kinematics.parquet"
        self._build_gaze_kinematics_dataframe().write_csv(file=kinematics_csv_path)
        self._build_gaze_kinematics_parquet_dataframe().write_parquet(file=kinematics_parquet_path)


    # =========================================================================
    # Convenience properties
    # =========================================================================

    @property
    def timestamps(self) -> NDArray[np.float64]:
        return self.kinematics.timestamps

    @property
    def framerate_hz(self) -> float:
        return self.kinematics.framerate_hz

    @property
    def n_frames(self) -> int:
        return self.kinematics.n_frames

    @property
    def duration_seconds(self) -> float:
        return self.kinematics.duration

    @property
    def eye_side(self) -> Literal["left", "right"]:
        return "right" if "right" in self.name.lower() else "left"

    # =========================================================================
    # gaze_kinematics kinematics accessors
    # =========================================================================

    @property
    def orientations(self) -> QuaternionTrajectory:
        return self.kinematics.orientations

    @property
    def quaternions_wxyz(self) -> NDArray[np.float64]:
        return self.kinematics.quaternions_wxyz

    @property
    def angular_velocity_trajectory(self) -> AngularVelocityTrajectory:
        return self.kinematics.angular_velocity_trajectory

    @property
    def angular_velocity_global(self) -> NDArray[np.float64]:
        return self.kinematics.angular_velocity_global

    @property
    def angular_velocity_local(self) -> NDArray[np.float64]:
        return self.kinematics.angular_velocity_local

    @property
    def angular_speed(self) -> Timeseries:
        return self.kinematics.angular_speed

    # =========================================================================
    # Angular Acceleration accessors
    # =========================================================================

    @property
    def angular_acceleration_trajectory(self) -> AngularAccelerationTrajectory:
        return self.kinematics.angular_acceleration_trajectory

    @property
    def angular_acceleration_global(self) -> NDArray[np.float64]:
        return self.kinematics.angular_acceleration_global

    @property
    def angular_acceleration_local(self) -> NDArray[np.float64]:
        return self.kinematics.angular_acceleration_local

    @property
    def angular_acceleration_magnitude(self) -> Timeseries:
        return self.kinematics.angular_acceleration_magnitude

    # =========================================================================
    # Gaze direction (Eye and Skull Movements)
    # =========================================================================

    @property
    def gaze_directions(self) -> NDArray[np.float64]:
        """
        Gaze direction (unit vector) over time.

        This is the +Z axis of the gaze_kinematics rotated by the current orientation.
        At rest, gaze = [0, 0, 1] (north pole).

        Returns:
            (N, 3) array of unit vectors
        """
        rest_gaze = np.array([0.0, 0.0, 1.0])
        return self.kinematics.orientations.rotate_vector(rest_gaze)

    @property
    def _anatomical_horizontal_sign(self) -> float:
        """
        Sign multiplier for Y component

        +Y points to subject's left (right hand coordinate system).
        - LEFT eye: +Y = subject left → gaze pointed out of head → sign = +1 → Rest gaze aligned more towards +Y
        - RIGHT eye: +Y = subject left → gaze pointed into head → sign = -1 → Rest gaze aligned more towards -Y
        """
        return 1.0 if self.eye_side == "left" else -1.0

    @property
    def horizontal_radians(self) -> NDArray[np.float64]:
        """
        Azimuth angle (horizontal gaze direction) in radians.

        Positive = looking more +X (subject's left)
        Zero = looking straight ahead (+Y for left eye/-Y for right eye)
        """
        gaze = self.gaze_directions
        return np.arctan2(gaze[:, 0], gaze[:, 1] * self._anatomical_horizontal_sign)

    @property
    def vertical_radians(self) -> NDArray[np.float64]:
        """
        Elevation angle (vertical gaze direction) in radians.

        Positive = looking up (+Z)
        Zero = looking along skull (XY) plane
        """
        gaze = self.gaze_directions
        horizontal = np.sqrt(gaze[:, 0] ** 2 + gaze[:, 1] ** 2)
        return np.arctan2(gaze[:, 2], horizontal)

    @property
    def horizontal_degrees(self) -> NDArray[np.float64]:
        return np.degrees(self.horizontal_radians)

    @property
    def vertical_degrees(self) -> NDArray[np.float64]:
        return np.degrees(self.vertical_radians)

    # =========================================================================
    # Gaze velocity (numerical derivative of azimuth / elevation)
    # =========================================================================

    @property
    def horizontal_velocity(self) -> Timeseries:
        """
        Rate of change of azimuth angle (rad/s).

        Computed via central differences (np.gradient). Positive = gaze
        moving in the positive azimuth direction.
        """
        return Timeseries(
            name="horizontal_velocity",
            timestamps=self.timestamps,
            values=np.gradient(self.horizontal_radians, self.timestamps),
        )

    @property
    def vertical_velocity(self) -> Timeseries:
        """
        Rate of change of elevation angle (rad/s).

        Computed via central differences (np.gradient). Positive = gaze
        moving upward.
        """
        return Timeseries(
            name="vertical_velocity",
            timestamps=self.timestamps,
            values=np.gradient(self.vertical_radians, self.timestamps),
        )

    # =========================================================================
    # Gaze acceleration (numerical derivative of velocity)
    # =========================================================================

    @property
    def horizontal_acceleration(self) -> Timeseries:
        """
        Rate of change of horizontal gaze velocity (rad/s²).

        Computed via central differences applied to horizontal_velocity.
        Two numerical differentiations amplify noise — consider smoothing
        the source data if this is used for saccade detection.
        """
        return Timeseries(
            name="horizontal_acceleration",
            timestamps=self.timestamps,
            values=np.gradient(self.horizontal_velocity.values, self.timestamps),
        )

    @property
    def vertical_acceleration(self) -> Timeseries:
        """
        Rate of change of vertical gaze velocity (rad/s²).

        Computed via central differences applied to vertical_velocity.
        Two numerical differentiations amplify noise — consider smoothing
        the source data if this is used for saccade detection.
        """
        return Timeseries(
            name="vertical_acceleration",
            timestamps=self.timestamps,
            values=np.gradient(self.vertical_velocity.values, self.timestamps),
        )


