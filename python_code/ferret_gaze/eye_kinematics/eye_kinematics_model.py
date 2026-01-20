"""
Refactored Ferret Eye Kinematics Model
======================================

Key architectural distinction:
- EYEBALL: A rigid body that rotates within the eye socket.
  Markers: eyeball_center, pupil_center, p1-p8 (rotate with eyeball)

- SOCKET LANDMARKS: Fixed relative to the skull/socket.
  Markers: tear_duct, outer_eye (do NOT rotate with eyeball)

The eyeball is modeled as a RigidBodyKinematics object with:
- Position always at [0,0,0] (eye-centered frame)
- Orientation that varies over time (where the eye is looking)
- Angular velocity computed from orientation changes

Socket landmarks are tracked separately and represent the anatomical
boundaries of the eye opening, which move with the head but NOT with
eye rotation.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python_code.kinematics_core.angular_velocity_trajectory_model import AngularVelocityTrajectory
from python_code.kinematics_core.quaternion_trajectory_model import QuaternionTrajectory
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry, MarkerPosition, \
    CoordinateFrameDefinition, AxisType, AxisDefinition
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.kinematics_core.timeseries_model import Timeseries
from python_code.kinematics_core.vector3_trajectory_model import Vector3Trajectory

# =============================================================================
# CONSTANTS
# =============================================================================

PUPIL_KEYPOINT_NAMES: tuple[str, ...] = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")
NUM_PUPIL_POINTS: int = 8


# =============================================================================
# EYEBALL REFERENCE GEOMETRY
# =============================================================================

def create_eyeball_reference_geometry(
        eye_radius_mm: float = 3.5,
        pupil_radius_mm: float = 0.5,
        pupil_eccentricity: float = 0.8,
) -> ReferenceGeometry:
    """
    Create reference geometry for the EYEBALL ONLY.

    This does NOT include tear_duct or outer_eye - those belong to the socket frame.

    Coordinate system (at rest):
        Origin: Eyeball center [0, 0, 0]
        +X: Rest gaze direction (pupil_center is at [eye_radius, 0, 0])
        +Y: Subject's left (perpendicular to gaze, in horizontal plane)
        +Z: Superior (up)

    Note: This is a right-handed coordinate system. The anatomical
    interpretation of +Y depends on which eye (medial for right, lateral for left).

    Markers:
        - eyeball_center: always [0, 0, 0]
        - pupil_center: [eye_radius, 0, 0] at rest
        - p1-p8: pupil boundary points in ellipse around pupil_center

    Args:
        eye_radius_mm: Radius of eyeball
        pupil_radius_mm: Semi-major axis of pupil ellipse (horizontal extent)
        pupil_eccentricity: Ratio of minor/major axis (1.0 = circle)

    Returns:
        ReferenceGeometry for the eyeball rigid body
    """
    R = eye_radius_mm
    a = pupil_radius_mm  # Semi-major (horizontal, along Y in tangent plane)
    b = pupil_radius_mm * pupil_eccentricity  # Semi-minor (vertical, along Z)

    markers: dict[str, MarkerPosition] = {}

    # Eyeball center (origin)
    markers["eyeball_center"] = MarkerPosition(x=0.0, y=0.0, z=0.0)

    # Pupil center (at +X on sphere surface)
    markers["pupil_center"] = MarkerPosition(x=R, y=0.0, z=0.0)

    # Pupil boundary points p1-p8 in ellipse around pupil center
    for i in range(NUM_PUPIL_POINTS):
        phi = 2 * np.pi * i / NUM_PUPIL_POINTS
        y_tangent = a * np.cos(phi)
        z_tangent = b * np.sin(phi)
        tangent_point = np.array([R, y_tangent, z_tangent])
        direction = tangent_point / np.linalg.norm(tangent_point)
        sphere_point = R * direction
        markers[f"p{i + 1}"] = MarkerPosition(
            x=float(sphere_point[0]),
            y=float(sphere_point[1]),
            z=float(sphere_point[2]),
        )

    coordinate_frame = CoordinateFrameDefinition(
        origin_markers=["eyeball_center"],
        x_axis=AxisDefinition(markers=["pupil_center"], type=AxisType.EXACT),
        y_axis=AxisDefinition(markers=["p1"], type=AxisType.APPROXIMATE),  # p1 is medial
    )

    # Display edges for visualization
    display_edges: list[tuple[str, str]] = [
        ("eyeball_center", "pupil_center"),
    ]
    for i in range(NUM_PUPIL_POINTS):
        next_i = (i + 1) % NUM_PUPIL_POINTS
        display_edges.append((f"p{i + 1}", f"p{next_i + 1}"))

    return ReferenceGeometry(
        units="mm",
        coordinate_frame=coordinate_frame,
        markers=markers,
        display_edges=display_edges,
    )


# =============================================================================
# SOCKET LANDMARKS (Fixed in skull frame)
# =============================================================================

class SocketLandmarks(BaseModel):
    """
    Anatomical landmarks of the eye socket/opening.

    These are FIXED relative to the skull - they do NOT rotate when the
    eyeball rotates. They may have small variations due to tracking noise
    or head motion, but their motion is independent of eye orientation.

    In the eye-centered coordinate frame:
        - tear_duct: Medial corner of the eye opening (toward nose)
            - For RIGHT eye: in +Y direction (since +Y = subject's left = medial)
            - For LEFT eye: in -Y direction (since +Y = subject's left = lateral)
        - outer_eye: Lateral corner of the eye opening (away from nose)
            - For RIGHT eye: in -Y direction
            - For LEFT eye: in +Y direction
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    timestamps: NDArray[np.float64]  # (N,)
    tear_duct_mm: NDArray[np.float64]  # (N, 3)
    outer_eye_mm: NDArray[np.float64]  # (N, 3)

    @model_validator(mode="after")
    def validate_shapes(self) -> "SocketLandmarks":
        n = len(self.timestamps)
        if self.tear_duct_mm.shape != (n, 3):
            raise ValueError(f"tear_duct_mm shape {self.tear_duct_mm.shape} != ({n}, 3)")
        if self.outer_eye_mm.shape != (n, 3):
            raise ValueError(f"outer_eye_mm shape {self.outer_eye_mm.shape} != ({n}, 3)")
        return self

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def tear_duct_trajectory(self) -> Vector3Trajectory:
        """Tear duct position over time."""
        return Vector3Trajectory(
            name="tear_duct",
            timestamps=self.timestamps,
            values=self.tear_duct_mm,
        )

    @property
    def outer_eye_trajectory(self) -> Vector3Trajectory:
        """Outer eye corner position over time."""
        return Vector3Trajectory(
            name="outer_eye",
            timestamps=self.timestamps,
            values=self.outer_eye_mm,
        )

    @property
    def eye_opening_width_mm(self) -> Timeseries:
        """Distance between tear duct and outer eye over time."""
        widths = np.linalg.norm(self.outer_eye_mm - self.tear_duct_mm, axis=1)
        return Timeseries(
            name="eye_opening_width",
            timestamps=self.timestamps,
            values=widths,
        )

    @property
    def midpoint_mm(self) -> NDArray[np.float64]:
        """Midpoint between landmarks (N, 3)."""
        return (self.tear_duct_mm + self.outer_eye_mm) / 2.0

    def get_mean_positions(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return mean positions of tear_duct and outer_eye."""
        return (
            np.mean(self.tear_duct_mm, axis=0),
            np.mean(self.outer_eye_mm, axis=0),
        )


# =============================================================================
# FERRET EYE KINEMATICS (Composite Model)
# =============================================================================

class FerretEyeKinematics(BaseModel):
    """
    Complete eye kinematics with clear separation of:

    1. EYEBALL KINEMATICS (RigidBodyKinematics):
       - Position: Always [0,0,0] (eye-centered frame)
       - Orientation: Quaternions representing where the eye is looking
       - Angular velocity: Computed from orientation changes
       - Keypoints: pupil_center, p1-p8, eyeball_center (rotate with eye)

    2. SOCKET LANDMARKS (SocketLandmarks):
       - tear_duct, outer_eye: Fixed in skull frame, don't rotate with eye
       - May have tracking noise but are independent of eye orientation

    COORDINATE SYSTEM (Right-handed, same world orientation for both eyes):
        Origin: Eye center (FIXED at [0, 0, 0] for ALL frames)
        +X: Rest gaze direction (anterior)
        +Y: Subject's left (medial for RIGHT eye, lateral for LEFT eye)
        +Z: Superior (up)

    NOTE: +Y always points to the subject's left to maintain right-handed
    coordinates for both eyes. Use anatomical accessors (adduction_angle,
    torsion_angle, etc.) for consistent anatomical meaning across eyes.

    At rest position:
        - Pupil at [+eye_radius, 0, 0]
        - Quaternion = [1, 0, 0, 0] (identity)
        - Gaze direction = [1, 0, 0]
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    # Metadata
    name: str = Field(min_length=1)
    source_path: str
    eye_side: Literal["left", "right"]

    # The eyeball as a rigid body
    # Contains: position (always [0,0,0]), orientation, angular velocity,
    #           and keypoint trajectories for pupil markers
    eyeball: RigidBodyKinematics

    # Socket landmarks (don't rotate with eyeball)
    socket_landmarks: SocketLandmarks

    # Calibration info (for camera-to-eye transformation)
    rest_gaze_direction_camera: NDArray[np.float64]  # (3,)
    camera_to_eye_rotation: NDArray[np.float64]  # (3, 3)

    @model_validator(mode="after")
    def validate_timestamps_match(self) -> "FerretEyeKinematics":
        """Ensure eyeball and socket landmarks have matching timestamps."""
        if self.eyeball.n_frames != self.socket_landmarks.n_frames:
            raise ValueError(
                f"Eyeball frames ({self.eyeball.n_frames}) != "
                f"socket landmark frames ({self.socket_landmarks.n_frames})"
            )
        # Check timestamps are close (allow small floating point differences)
        if not np.allclose(self.eyeball.timestamps, self.socket_landmarks.timestamps, rtol=1e-9):
            raise ValueError("Eyeball and socket landmark timestamps don't match")
        return self

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

    # =========================================================================
    # Eyeball kinematics accessors
    # =========================================================================

    @property
    def orientations(self) -> QuaternionTrajectory:
        """Eyeball orientation over time."""
        return self.eyeball.orientations

    @property
    def quaternions_wxyz(self) -> NDArray[np.float64]:
        """Eyeball orientation as (N, 4) array of [w, x, y, z]."""
        return self.eyeball.quaternions_wxyz

    @property
    def angular_velocity_trajectory(self) -> AngularVelocityTrajectory:
        """Eyeball angular velocity over time."""
        return self.eyeball.angular_velocity_trajectory

    @property
    def angular_velocity_global(self) -> NDArray[np.float64]:
        """Global angular velocity (N, 3) in rad/s."""
        return self.eyeball.angular_velocity_global

    @property
    def angular_velocity_local(self) -> NDArray[np.float64]:
        """Local (body-frame) angular velocity (N, 3) in rad/s."""
        return self.eyeball.angular_velocity_local

    @property
    def angular_speed(self) -> Timeseries:
        """Magnitude of angular velocity over time."""
        return self.eyeball.angular_speed

    @property
    def roll(self) -> Timeseries:
        """Roll angle (torsion) over time."""
        return self.eyeball.roll

    @property
    def pitch(self) -> Timeseries:
        """Pitch angle (elevation) over time."""
        return self.eyeball.pitch

    @property
    def yaw(self) -> Timeseries:
        """Yaw angle (azimuth) over time."""
        return self.eyeball.yaw

    # =========================================================================
    # Gaze direction
    # =========================================================================

    @property
    def gaze_directions(self) -> NDArray[np.float64]:
        """
        Gaze direction (unit vector) over time.

        This is the +X axis of the eyeball rotated by the current orientation.
        At rest, gaze = [1, 0, 0].

        Returns:
            (N, 3) array of unit vectors
        """
        # Gaze is the rotated +X axis
        rest_gaze = np.array([1.0, 0.0, 0.0])
        return self.eyeball.orientations.rotate_vector(rest_gaze)

    @property
    def azimuth_radians(self) -> NDArray[np.float64]:
        """
        Azimuth angle (horizontal gaze direction) in radians.

        Positive = looking toward +Y (medial for right eye)
        Zero = looking straight ahead (+X)
        """
        gaze = self.gaze_directions
        return np.arctan2(gaze[:, 1], gaze[:, 0])

    @property
    def elevation_radians(self) -> NDArray[np.float64]:
        """
        Elevation angle (vertical gaze direction) in radians.

        Positive = looking up (+Z)
        Zero = looking straight ahead
        """
        gaze = self.gaze_directions
        horizontal = np.sqrt(gaze[:, 0] ** 2 + gaze[:, 1] ** 2)
        return np.arctan2(gaze[:, 2], horizontal)

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
    # These provide rotation components with consistent ANATOMICAL meaning
    # for both left and right eyes, despite the underlying coordinate systems
    # having +Y point in opposite anatomical directions (to stay right-handed).
    #
    # Underlying Coordinate System (RIGHT-HANDED for both eyes):
    #   +X = Anterior (gaze direction at rest)
    #   +Y = Subject's left (medial for RIGHT eye, lateral for LEFT eye)
    #   +Z = Superior (up)
    #
    # The anatomical accessors flip signs as needed so that:
    #   - Adduction is ALWAYS positive when gaze moves toward nose
    #   - Elevation is ALWAYS positive when gaze moves up
    #   - Extorsion is ALWAYS positive when top of eye tilts away from nose
    #
    # =========================================================================

    @property
    def _anatomical_horizontal_sign(self) -> float:
        """
        Sign multiplier for horizontal (adduction/abduction) angles.

        For right eye: +Y = medial, so +yaw = adduction → sign = +1
        For left eye: +Y = lateral, so +yaw = abduction → sign = -1
        """
        return 1.0 if self.eye_side == "right" else -1.0

    @property
    def _anatomical_torsion_sign(self) -> float:
        """
        Sign multiplier for torsion angles.

        Using right-hand rule for +roll (rotation around +X/gaze):
        - Thumb along +X (forward)
        - Fingers curl from +Y toward +Z
        - The TOP of eye (at +Z) moves toward -Y (subject's right)

        For RIGHT eye: -Y = lateral → +roll tilts top laterally = EXTORSION
        For LEFT eye: -Y = medial → +roll tilts top medially = INTORSION

        We want +torsion_angle = extorsion for both eyes, so:
        - Right eye: sign = +1 (no flip needed)
        - Left eye: sign = -1 (flip to make extorsion positive)
        """
        return 1.0 if self.eye_side == "right" else -1.0

    @property
    def adduction_angle(self) -> Timeseries:
        """
        Horizontal gaze angle with consistent anatomical meaning.

        Positive = Adduction (gaze toward nose/medial)
        Negative = Abduction (gaze away from nose/lateral)

        This has the SAME anatomical meaning for both left and right eyes.
        """
        return Timeseries(
            name="adduction",
            timestamps=self.timestamps,
            values=self._anatomical_horizontal_sign * self.yaw.values,
        )

    @property
    def elevation_angle(self) -> Timeseries:
        """
        Vertical gaze angle (rotation around horizontal axis).

        Positive = Elevation (gaze up/superior)
        Negative = Depression (gaze down/inferior)

        No sign flip needed - elevation is the same for both eyes.
        """
        return Timeseries(
            name="elevation",
            timestamps=self.timestamps,
            values=self.pitch.values,
        )

    @property
    def torsion_angle(self) -> Timeseries:
        """
        Torsional rotation around the gaze/anterior axis.

        Positive = Extorsion (top of eye rotates away from nose/laterally)
        Negative = Intorsion (top of eye rotates toward nose/medially)

        This has the SAME anatomical meaning for both left and right eyes.
        """
        return Timeseries(
            name="torsion",
            timestamps=self.timestamps,
            values=self._anatomical_torsion_sign * self.roll.values,
        )

    @property
    def intorsion_angle(self) -> Timeseries:
        """
        Torsional rotation, with intorsion as positive.

        Positive = Intorsion (top of eye rotates toward nose/medially)
        Negative = Extorsion (top of eye rotates away from nose/laterally)

        This is just -torsion_angle, provided for convenience since some
        literature uses intorsion-positive convention.
        """
        return Timeseries(
            name="intorsion",
            timestamps=self.timestamps,
            values=-self._anatomical_torsion_sign * self.roll.values,
        )

    # =========================================================================
    # Anatomical Angular Velocity
    # =========================================================================

    @property
    def adduction_velocity(self) -> Timeseries:
        """
        Angular velocity of adduction/abduction (rad/s).

        Positive = adducting (gaze rotating toward nose)
        Negative = abducting (gaze rotating away from nose)

        Consistent anatomical meaning for both eyes.
        """
        return Timeseries(
            name="adduction_velocity",
            timestamps=self.timestamps,
            values=self._anatomical_horizontal_sign * self.angular_velocity_global[:, 2],
        )

    @property
    def elevation_velocity(self) -> Timeseries:
        """
        Angular velocity of elevation/depression (rad/s).

        Positive = elevating (gaze rotating upward)
        Negative = depressing (gaze rotating downward)

        No sign flip needed.
        """
        return Timeseries(
            name="elevation_velocity",
            timestamps=self.timestamps,
            values=self.angular_velocity_global[:, 1],
        )

    @property
    def torsion_velocity(self) -> Timeseries:
        """
        Angular velocity of torsion (rad/s).

        Positive = extorting (top of eye rotating away from nose)
        Negative = intorting (top of eye rotating toward nose)

        Consistent anatomical meaning for both eyes.
        """
        return Timeseries(
            name="torsion_velocity",
            timestamps=self.timestamps,
            values=self._anatomical_torsion_sign * self.angular_velocity_global[:, 0],
        )

    # =========================================================================
    # Pupil position accessors
    # =========================================================================

    @property
    def pupil_center_trajectory(self) -> NDArray[np.float64]:
        """Pupil center position over time (N, 3)."""
        return self.eyeball.keypoint_trajectories["pupil_center"]

    def get_pupil_point_trajectory(self, point_index: int) -> NDArray[np.float64]:
        """Get trajectory for pupil boundary point p1-p8."""
        if not 1 <= point_index <= NUM_PUPIL_POINTS:
            raise ValueError(f"point_index must be 1-{NUM_PUPIL_POINTS}, got {point_index}")
        return self.eyeball.keypoint_trajectories[f"p{point_index}"]

    @property
    def pupil_points_trajectories(self) -> NDArray[np.float64]:
        """All pupil boundary points over time (N, 8, 3)."""
        points = [
            self.eyeball.keypoint_trajectories[f"p{i + 1}"]
            for i in range(NUM_PUPIL_POINTS)
        ]
        return np.stack(points, axis=1)

    # =========================================================================
    # Socket landmark accessors
    # =========================================================================

    @property
    def tear_duct_mm(self) -> NDArray[np.float64]:
        """Tear duct position over time (N, 3)."""
        return self.socket_landmarks.tear_duct_mm

    @property
    def outer_eye_mm(self) -> NDArray[np.float64]:
        """Outer eye corner position over time (N, 3)."""
        return self.socket_landmarks.outer_eye_mm

    # =========================================================================
    # Factory method
    # =========================================================================

    @classmethod
    def from_pose_data(
            cls,
            name: str,
            source_path: str,
            eye_side: Literal["left", "right"],
            timestamps: NDArray[np.float64],
            quaternions_wxyz: NDArray[np.float64],
            tear_duct_mm: NDArray[np.float64],
            outer_eye_mm: NDArray[np.float64],
            rest_gaze_direction_camera: NDArray[np.float64],
            camera_to_eye_rotation: NDArray[np.float64],
            eye_radius_mm: float = 3.5,
            pupil_radius_mm: float = 0.5,
            pupil_eccentricity: float = 0.8,
    ) -> "FerretEyeKinematics":
        """
        Construct FerretEyeKinematics from basic data arrays.

        This is the primary factory method. It:
        1. Creates the eyeball reference geometry
        2. Builds RigidBodyKinematics for the eyeball (computing angular velocities)
        3. Wraps socket landmarks

        Args:
            name: Identifier for this recording
            source_path: Path to source data file
            eye_side: "left" or "right"
            timestamps: (N,) array of timestamps in seconds
            quaternions_wxyz: (N, 4) array of eyeball orientations [w, x, y, z]
            tear_duct_mm: (N, 3) array of tear duct positions
            outer_eye_mm: (N, 3) array of outer eye positions
            rest_gaze_direction_camera: (3,) rest gaze in camera frame
            camera_to_eye_rotation: (3, 3) rotation matrix
            eye_radius_mm: Eyeball radius
            pupil_radius_mm: Pupil ellipse semi-major axis
            pupil_eccentricity: Pupil ellipse eccentricity (minor/major)

        Returns:
            FerretEyeKinematics instance with computed angular velocities
        """
        # Create eyeball reference geometry
        eyeball_geometry = create_eyeball_reference_geometry(
            eye_radius_mm=eye_radius_mm,
            pupil_radius_mm=pupil_radius_mm,
            pupil_eccentricity=pupil_eccentricity,
        )

        # Eyeball position is always at origin (eye-centered frame)
        n_frames = len(timestamps)
        position_xyz = np.zeros((n_frames, 3), dtype=np.float64)

        # Create RigidBodyKinematics for the eyeball
        # This computes velocities, angular velocities, and keypoint trajectories
        eyeball = RigidBodyKinematics.from_pose_arrays(
            name=f"{name}_eyeball",
            reference_geometry=eyeball_geometry,
            timestamps=timestamps,
            position_xyz=position_xyz,
            quaternions_wxyz=quaternions_wxyz,
        )

        # Create socket landmarks
        socket_landmarks = SocketLandmarks(
            timestamps=timestamps,
            tear_duct_mm=tear_duct_mm,
            outer_eye_mm=outer_eye_mm,
        )

        return cls(
            name=name,
            source_path=source_path,
            eye_side=eye_side,
            eyeball=eyeball,
            socket_landmarks=socket_landmarks,
            rest_gaze_direction_camera=rest_gaze_direction_camera,
            camera_to_eye_rotation=camera_to_eye_rotation,
        )