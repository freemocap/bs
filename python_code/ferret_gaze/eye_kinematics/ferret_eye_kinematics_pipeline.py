"""
Ferret Eye Kinematics Pipeline
==============================

Loads eye tracking data from CSV, computes 3D eye orientation, and produces
RigidBodyKinematics-compatible output.

Data Format (Input CSV):
    Columns: frame, timestamp, keypoint, x, y, processing_level, ...
    Keypoints: p1-p8 (pupil boundary), tear_duct, outer_eye

Output Includes:
    - Eye orientation (quaternions, angles)
    - All tracked points in eye-centered coordinates:
        - pupil_center (mean of p1-p8)
        - pupil_points_p1 through pupil_points_p8
        - tear_duct
        - outer_eye
        - eye_center (always [0,0,0] by construction)

Output Coordinate System (Eyeball-Centered, Median pupil position-Aligned):
    - Origin: Eye center (fixed at [0, 0, 0] for ALL frames)
    - +X: Rest gaze direction (where eye looks most often, median pupil position across frames)
    - +Y: medial/lateral, towards the nose for the right eye, towards the ear for the left eye
    - +Z: upwards, completing the right-handed system
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# CONSTANTS
# =============================================================================

PUPIL_KEYPOINT_NAMES: tuple[str, ...] = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")
NUM_PUPIL_POINTS: int = 8
FERRET_EYE_DIAMETER_MM: float = 7.0  # Average eye diameter for ferrets, in mm


# =============================================================================
# CAMERA PARAMETERS (Pydantic)
# =============================================================================

class PupilCoreCameraParameters(BaseModel):
    """Intrinsic parameters of the Pupil Core eye camera."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    focal_length_mm: float = Field(gt=0, description="Physical focal length in mm")
    sensor_width_mm: float = Field(gt=0, description="Active sensor width in mm")
    image_width_pixels: int = Field(gt=0, description="Image width in pixels")
    image_height_pixels: int = Field(gt=0, description="Image height in pixels")

    @property
    def focal_length_pixels(self) -> float:
        """Focal length in pixel units."""
        return self.focal_length_mm * (self.image_width_pixels / self.sensor_width_mm)

    @property
    def principal_point_x(self) -> float:
        """X coordinate of optical center."""
        return self.image_width_pixels / 2.0

    @property
    def principal_point_y(self) -> float:
        """Y coordinate of optical center."""
        return self.image_height_pixels / 2.0

    @classmethod
    def at_400x400(cls) -> "PupilCoreCameraParameters":
        return cls(
            focal_length_mm=1.7,
            sensor_width_mm=2.4,
            image_width_pixels=400,
            image_height_pixels=400,
        )

    @classmethod
    def at_192x192(cls) -> "PupilCoreCameraParameters":
        return cls(
            focal_length_mm=1.7,
            sensor_width_mm=1.15,
            image_width_pixels=192,
            image_height_pixels=192,
        )


class EyeGeometryParameters(BaseModel):
    """Physical geometry of the ferret eye."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eye_diameter_mm: float = Field(default=7.0, gt=0, description="Eye diameter in mm")

    @property
    def eye_radius_mm(self) -> float:
        return self.eye_diameter_mm / 2.0


class PixelCoordinate(BaseModel):
    """A 2D pixel coordinate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(description="Horizontal pixel (0 = left edge)")
    y: float = Field(description="Vertical pixel (0 = top edge)")

    @field_validator("x", "y")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Pixel coordinate must be finite, got {v}")
        return v

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y], dtype=np.float64)


class RawEyeFrameData(BaseModel):
    """
    Raw tracking data for a single frame.

    Contains all tracked points: pupil boundary (p1-p8), tear_duct, outer_eye.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_number: int = Field(ge=0)
    timestamp_seconds: float

    # Individual pupil boundary points (p1-p8)
    pupil_points: tuple[PixelCoordinate, ...] = Field(
        description="Pupil boundary points p1-p8, in order"
    )

    # Computed pupil center
    pupil_center: PixelCoordinate

    # Anatomical landmarks
    tear_duct: PixelCoordinate
    outer_eye: PixelCoordinate

    @field_validator("timestamp_seconds")
    @classmethod
    def check_timestamp_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError(f"Timestamp must be finite, got {v}")
        return v

    @field_validator("pupil_points")
    @classmethod
    def check_pupil_points_count(cls, v: tuple[PixelCoordinate, ...]) -> tuple[PixelCoordinate, ...]:
        if len(v) != NUM_PUPIL_POINTS:
            raise ValueError(f"Expected {NUM_PUPIL_POINTS} pupil points, got {len(v)}")
        return v

    @property
    def landmark_distance_pixels(self) -> float:
        """Distance between tear duct and outer eye in pixels."""
        dx = self.outer_eye.x - self.tear_duct.x
        dy = self.outer_eye.y - self.tear_duct.y
        return float(np.sqrt(dx * dx + dy * dy))

    @property
    def eye_center_pixel(self) -> PixelCoordinate:
        """Eye center (midpoint of landmarks)."""
        return PixelCoordinate(
            x=(self.tear_duct.x + self.outer_eye.x) / 2.0,
            y=(self.tear_duct.y + self.outer_eye.y) / 2.0,
        )


class CameraCenteredEyeballState(BaseModel):
    """
    Computed 3D geometry for a single frame, in camera coordinates.

    Camera coordinate system:
        Origin: Camera nodal point
        +X: Right in image
        +Y: Down in image
        +Z: Into scene (toward eye)


    Contains ALL tracked points transformed to 3D.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    frame_number: int = Field(ge=0)
    timestamp_seconds: float
    eye_distance_mm: float = Field(gt=0)

    reference_frame: Literal['camera', 'eye', 'skull', 'world']
    # Core geometry
    eyeball_center_mm: NDArray[np.float64] = Field(description="(3,)")
    pupil_center_mm: NDArray[np.float64] = Field(description="(3,)")
    gaze_direction: NDArray[np.float64] = Field(description="(3,) unit vector")

    # All tracked points in camera frame
    pupil_points_mm: NDArray[np.float64] = Field(description="(8, 3) p1-p8 on eye surface")
    tear_duct_mm: NDArray[np.float64] = Field(description="(3,)")
    outer_eye_mm: NDArray[np.float64] = Field(description="(3,)")

    @field_validator("eye_center_mm", "pupil_center_mm", "gaze_direction",
                     "tear_duct_mm", "outer_eye_mm", mode="before")
    @classmethod
    def convert_vec3(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Values must be finite")
        return arr

    @field_validator("pupil_points_mm", mode="before")
    @classmethod
    def convert_pupil_points(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (NUM_PUPIL_POINTS, 3):
            raise ValueError(f"Expected shape ({NUM_PUPIL_POINTS}, 3), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Values must be finite")
        return arr

    @model_validator(mode="after")
    def validate_geometry(self) -> "CameraCenteredEyeballState":
        # Check gaze is unit vector
        gaze_mag = float(np.linalg.norm(self.gaze_direction))
        if abs(gaze_mag - 1.0) > 1e-6:
            raise ValueError(f"Gaze must be unit vector, magnitude={gaze_mag}")

        # Check points are in front of camera
        if self.eyeball_center_mm[2] <= 0:
            raise ValueError(f"Eye center must be in front of camera (z>0)")
        if self.pupil_center_mm[2] <= 0:
            raise ValueError(f"Pupil must be in front of camera (z>0)")

        return self

class EyeballState(BaseModel):
    """
    Computed 3D geometry for a single frame, in eye-centered coordinates.

    Eye-centered coordinate system:
        Origin: Eye center (ALWAYS [0,0,0], explicitly defined for compatibilty with RigidBodyKinematics)
        +X: Rest gaze direction (where eye looks most often, median pupil position across frames)
        +Y: medial/lateral, towards the nose for the right eye, towards the ear for the left eye
        +Z: upwards, completing the right-handed system
    Contains ALL tracked points transformed to eye-centered frame.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    frame_number: int = Field(ge=0)
    timestamp_seconds: float

    # Orientation
    gaze_direction: NDArray[np.float64] = Field(description="(3,) unit vector")
    azimuth_radians: float
    elevation_radians: float
    quaternion_wxyz: NDArray[np.float64] = Field(description="(4,)")

    # All tracked points in eye-centered frame
    eyeball_center_mm: NDArray[np.float64] = Field(description="(3,)")
    pupil_center_mm: NDArray[np.float64] = Field(description="(3,)")
    pupil_points_mm: NDArray[np.float64] = Field(description="(8, 3)")
    tear_duct_mm: NDArray[np.float64] = Field(description="(3,)")
    outer_eye_mm: NDArray[np.float64] = Field(description="(3,)")

    @field_validator("gaze_direction", "pupil_center_mm", "tear_duct_mm", "outer_eye_mm", 'eyeball_center_mm',
                     mode="before")
    @classmethod
    def convert_vec3(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        return arr

    @field_validator("quaternion_wxyz", mode="before")
    @classmethod
    def convert_quat(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (4,):
            raise ValueError(f"Expected shape (4,), got {arr.shape}")
        return arr

    @field_validator("pupil_points_mm", mode="before")
    @classmethod
    def convert_pupil_points(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (NUM_PUPIL_POINTS, 3):
            raise ValueError(f"Expected shape ({NUM_PUPIL_POINTS}, 3), got {arr.shape}")
        return arr

    @model_validator(mode="after")
    def validate_geometry(self) -> "EyeballState":
        # Check gaze is unit vector
        gaze_mag = float(np.linalg.norm(self.gaze_direction))
        if abs(gaze_mag - 1.0) > 1e-6:
            raise ValueError(f"Gaze must be unit vector, magnitude={gaze_mag}")

        # Check quaternion is unit
        quat_mag = float(np.linalg.norm(self.quaternion_wxyz))
        if abs(quat_mag - 1.0) > 1e-6:
            raise ValueError(f"Quaternion must be unit, magnitude={quat_mag}")

        return self


def compute_pupil_center_from_points(
    pupil_points: tuple[PixelCoordinate, ...],
    method: Literal["mean", "median"] = "median",
) -> PixelCoordinate:
    """Compute pupil center from p1-p8 points."""
    xs = [p.x for p in pupil_points]
    ys = [p.y for p in pupil_points]

    if method == "mean":
        return PixelCoordinate(x=float(np.mean(xs)), y=float(np.mean(ys)))
    elif method == "median":
        return PixelCoordinate(x=float(np.median(xs)), y=float(np.median(ys)))
    else:
        raise ValueError(f"Unknown method: {method}")


def load_raw_frame_data_from_csv(
    csv_path: Path,
    processing_level: str = "cleaned",
    pupil_aggregation: Literal["mean", "median"] = "median",
) -> list[RawEyeFrameData]:
    """
    Load eye tracking data from CSV file.

    Returns list of RawFrameData containing all tracked points.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Eye data CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["processing_level"] == processing_level]
    if len(df) == 0:
        raise ValueError(f"No data with processing_level='{processing_level}'")

    frames_data: list[RawEyeFrameData] = []
    frame_numbers = sorted(df["frame"].unique())

    for frame_number in frame_numbers:
        frame_df = df[df["frame"] == frame_number]
        timestamp = float(frame_df["timestamp"].iloc[0])

        # Get all pupil points (p1-p8) in order
        pupil_points: list[PixelCoordinate] = []
        missing_points: list[str] = []

        for keypoint_name in PUPIL_KEYPOINT_NAMES:
            kp_df = frame_df[frame_df["keypoint"] == keypoint_name]
            if len(kp_df) == 0:
                missing_points.append(keypoint_name)
                continue
            pupil_points.append(PixelCoordinate(
                x=float(kp_df["x"].iloc[0]),
                y=float(kp_df["y"].iloc[0]),
            ))


        # Get tear_duct and outer_eye
        tear_duct_df = frame_df[frame_df["keypoint"] == "tear_duct"]
        outer_eye_df = frame_df[frame_df["keypoint"] == "outer_eye"]

        if len(tear_duct_df) == 0 or len(outer_eye_df) == 0:
            continue

        # Compute pupil center
        pupil_points_tuple = tuple(pupil_points)
        pupil_center = compute_pupil_center_from_points(
            pupil_points=pupil_points_tuple,
            method=pupil_aggregation,
        )

        frames_data.append(RawEyeFrameData(
            frame_number=int(frame_number),
            timestamp_seconds=timestamp,
            pupil_points=pupil_points_tuple,
            pupil_center=pupil_center,
            tear_duct=PixelCoordinate(
                x=float(tear_duct_df["x"].iloc[0]),
                y=float(tear_duct_df["y"].iloc[0]),
            ),
            outer_eye=PixelCoordinate(
                x=float(outer_eye_df["x"].iloc[0]),
                y=float(outer_eye_df["y"].iloc[0]),
            ),
        ))

    if len(frames_data) == 0:
        raise ValueError(f"No valid frames found in {csv_path}")

    # Validate monotonic timestamps
    timestamps = [f.timestamp_seconds for f in frames_data]
    diffs = np.diff(timestamps)
    if not np.all(diffs > 0):
        bad_indices = np.where(diffs <= 0)[0]
        raise ValueError(f"Timestamps not monotonically increasing at indices: {bad_indices[:5].tolist()}")

    return frames_data


def pixel_to_ray_direction(
    pixel: PixelCoordinate,
    camera: PupilCoreCameraParameters,
) -> NDArray[np.float64]:
    """Convert pixel to normalized ray direction from camera origin."""
    ray_x = (pixel.x - camera.principal_point_x) / camera.focal_length_pixels
    ray_y = (pixel.y - camera.principal_point_y) / camera.focal_length_pixels
    ray_z = 1.0
    ray = np.array([ray_x, ray_y, ray_z], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def pixel_to_3d_at_depth(
    pixel: PixelCoordinate,
    depth_mm: float,
    camera: PupilCoreCameraParameters,
) -> NDArray[np.float64]:
    """
    Project pixel to 3D point at specified depth.

    Used for landmarks that aren't necessarily on the eye sphere surface.
    """
    x = (pixel.x - camera.principal_point_x) * depth_mm / camera.focal_length_pixels
    y = (pixel.y - camera.principal_point_y) * depth_mm / camera.focal_length_pixels
    z = depth_mm
    return np.array([x, y, z], dtype=np.float64)


def ray_sphere_intersection(
    ray_direction: NDArray[np.float64],
    sphere_center: NDArray[np.float64],
    sphere_radius: float,
) -> NDArray[np.float64]:
    """
    Find intersection of ray from camera origin with sphere.

    Returns the closer intersection point.
    """
    # Quadratic: t² - 2t(d·c) + |c|² - r² = 0
    d_dot_c = np.dot(ray_direction, sphere_center)
    c_squared = np.dot(sphere_center, sphere_center)

    a = 1.0
    b = -2.0 * d_dot_c
    c = c_squared - sphere_radius ** 2

    discriminant = b * b - 4.0 * a * c

    if discriminant < 0:
        raise ValueError(f"Ray does not intersect sphere (discriminant={discriminant:.4f})")

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    if t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    else:
        raise ValueError(f"Sphere is behind camera (t1={t1:.3f}, t2={t2:.3f})")

    return t * ray_direction


def compute_frame_geometry_camera_frame(
    frame_data: RawEyeFrameData,
    camera: PupilCoreCameraParameters,
    eye: EyeGeometryParameters,
) -> CameraCenteredEyeballState:
    """
    Compute 3D geometry for all tracked points in camera coordinates.
    """
    # Step 1: Estimate eye distance
    landmark_distance_pixels = frame_data.landmark_distance_pixels
    if landmark_distance_pixels < 1.0:
        raise ValueError(f"Landmark distance too small: {landmark_distance_pixels:.2f}px")

    eyeplane_distance_mm = (
        camera.focal_length_pixels *
        eye.eye_diameter_mm /
        landmark_distance_pixels
    )

    # Step 2: Eye center in camera frame
    eye_center_pixel = frame_data.eye_center_pixel
    eyeball_center_mm = pixel_to_3d_at_depth(eye_center_pixel, eyeplane_distance_mm+eye.eye_radius_mm, camera)

    # Step 3: Pupil center via ray-sphere intersection (it's on the eye surface)
    pupil_ray = pixel_to_ray_direction(frame_data.pupil_center, camera)
    pupil_center_mm = ray_sphere_intersection(pupil_ray, eyeball_center_mm, eye.eye_radius_mm)

    # Step 4: All pupil points (p1-p8) via ray-sphere intersection
    pupil_points_mm = np.zeros((NUM_PUPIL_POINTS, 3), dtype=np.float64)
    for frame_number, pupil_pixel in enumerate(frame_data.pupil_points):
        ray = pixel_to_ray_direction(pupil_pixel, camera)
        pupil_points_mm[frame_number] = ray_sphere_intersection(ray, eyeball_center_mm, eye.eye_radius_mm)

    # Step 5: Tear duct and outer eye - project to plane at eyeplane depth
    # These are fixed landmarks on the face, not on the eyeball sphere
    tear_duct_mm = pixel_to_3d_at_depth(frame_data.tear_duct, eyeplane_distance_mm, camera)
    outer_eye_mm = pixel_to_3d_at_depth(frame_data.outer_eye, eyeplane_distance_mm, camera)

    # Step 6: Gaze direction
    gaze_vector = pupil_center_mm - eyeball_center_mm
    gaze_direction = gaze_vector / np.linalg.norm(gaze_vector)

    return CameraCenteredEyeballState(
        frame_number=frame_data.frame_number,
        reference_frame="camera",
        timestamp_seconds=frame_data.timestamp_seconds,
        eye_distance_mm=eyeplane_distance_mm,
        eyeball_center_mm=eyeball_center_mm,
        pupil_center_mm=pupil_center_mm,
        gaze_direction=gaze_direction,
        pupil_points_mm=pupil_points_mm,
        tear_duct_mm=tear_duct_mm,
        outer_eye_mm=outer_eye_mm,
    )


# =============================================================================
# ROTATION MATH
# =============================================================================

def compute_rotation_matrix_aligning_vectors(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute rotation matrix R such that R @ source ≈ target."""
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)
    dot = np.dot(a, b)

    if dot > 0.999999:
        return np.eye(3, dtype=np.float64)

    if dot < -0.999999:
        if abs(a[0]) < 0.9:
            perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
        else:
            perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
        perp = perp / np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3, dtype=np.float64)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = dot

    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=np.float64)

    R = np.eye(3, dtype=np.float64) + K + (K @ K) * (1.0 - c) / (s * s)
    return R


def rotation_matrix_to_quaternion_wxyz(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

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

    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def transform_point_to_eye_frame(
    point_camera: NDArray[np.float64],
    eye_center_camera: NDArray[np.float64],
    rotation_camera_to_eye: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Transform a point from camera frame to eye-centered frame.

    1. Subtract THIS frame's eye center (translate to eye-centered)
    2. Apply rotation to align with rest frame
    """
    point_relative = point_camera - eye_center_camera
    return rotation_camera_to_eye @ point_relative


def transform_points_to_eye_frame(
    points_camera: NDArray[np.float64],
    eye_center_camera: NDArray[np.float64],
    rotation_camera_to_eye: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Transform multiple points from camera frame to eye-centered frame.

    Args:
        points_camera: (N, 3) array of points
        eye_center_camera: (3,) eye center for this frame
        rotation_camera_to_eye: (3, 3) rotation matrix

    Returns:
        (N, 3) transformed points
    """
    points_relative = points_camera - eye_center_camera
    return (rotation_camera_to_eye @ points_relative.T).T


# =============================================================================
# OUTPUT DATA MODEL
# =============================================================================

class TrackedPointsEyeFrame(BaseModel):
    """
    All tracked points for a recording, in eye-centered coordinates.

    Each array has shape (num_frames, 3) unless otherwise noted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    # Pupil
    pupil_center_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_p1_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_p2_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_p3_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_p4_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_p5_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_p6_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_p7_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_p8_mm: NDArray[np.float64] = Field(description="(N, 3)")

    # Landmarks
    tear_duct_mm: NDArray[np.float64] = Field(description="(N, 3)")
    outer_eye_mm: NDArray[np.float64] = Field(description="(N, 3)")

    @field_validator("*", mode="before")
    @classmethod
    def convert_array(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Values must be finite")
        return arr

    @model_validator(mode="after")
    def check_all_same_length(self) -> "TrackedPointsEyeFrame":
        n = len(self.pupil_center_mm)
        fields = [
            "pupil_center_mm", "pupil_p1_mm", "pupil_p2_mm", "pupil_p3_mm",
            "pupil_p4_mm", "pupil_p5_mm", "pupil_p6_mm", "pupil_p7_mm", "pupil_p8_mm",
            "tear_duct_mm", "outer_eye_mm",
        ]
        for field_name in fields:
            arr = getattr(self, field_name)
            if len(arr) != n:
                raise ValueError(f"{field_name} length {len(arr)} != {n}")
        return self

    @property
    def num_frames(self) -> int:
        return len(self.pupil_center_mm)

    def get_pupil_points_array(self) -> NDArray[np.float64]:
        """Get all pupil points as (N, 8, 3) array."""
        return np.stack([
            self.pupil_p1_mm, self.pupil_p2_mm, self.pupil_p3_mm, self.pupil_p4_mm,
            self.pupil_p5_mm, self.pupil_p6_mm, self.pupil_p7_mm, self.pupil_p8_mm,
        ], axis=1)


class TrackedPointsCameraFrame(BaseModel):
    """All tracked points in camera frame (for reference/debugging)."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    eye_center_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_center_mm: NDArray[np.float64] = Field(description="(N, 3)")
    pupil_points_mm: NDArray[np.float64] = Field(description="(N, 8, 3)")
    tear_duct_mm: NDArray[np.float64] = Field(description="(N, 3)")
    outer_eye_mm: NDArray[np.float64] = Field(description="(N, 3)")

    @field_validator("eye_center_mm", "pupil_center_mm", "tear_duct_mm", "outer_eye_mm",
                     mode="before")
    @classmethod
    def convert_nx3(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {arr.shape}")
        return arr

    @field_validator("pupil_points_mm", mode="before")
    @classmethod
    def convert_nx8x3(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1] != 8 or arr.shape[2] != 3:
            raise ValueError(f"Expected shape (N, 8, 3), got {arr.shape}")
        return arr


class FerretEyeKinematics(BaseModel):
    """
    Complete eye kinematics in eye-centered, rest-aligned coordinates.

    COORDINATE SYSTEM:
        Origin: Eye center (FIXED at [0, 0, 0] for ALL frames)
        +X: Rest gaze direction (forward)
        +Y: Roughly dorsal/superior
        +Z: Roughly mediolateral, nose-wards for right eye, ear-wards for left eye (to define right-handed coordinate system)

    At rest position:
        - Pupil at [+eye_radius, 0, 0]
        - Quaternion = [1, 0, 0, 0] (identity)
        - Azimuth = 0, Elevation = 0

    Contains ALL tracked points transformed to eye-centered frame.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    # Metadata
    name: str = Field(min_length=1)
    source_csv_path: str

    # Parameters
    camera_parameters: PupilCoreCameraParameters
    eye_geometry: EyeGeometryParameters

    # Time and frame info
    timestamps_seconds: NDArray[np.float64] = Field(description="(N,)")
    frame_numbers: NDArray[np.int64] = Field(description="(N,)")

    # Calibration info
    mean_eye_center_camera_mm: NDArray[np.float64] = Field(description="(3,)")
    rest_gaze_direction_camera: NDArray[np.float64] = Field(description="(3,)")
    camera_to_eye_rotation: NDArray[np.float64] = Field(description="(3,3)")

    # Core kinematics
    gaze_directions: NDArray[np.float64] = Field(description="(N, 3)")
    quaternions_wxyz: NDArray[np.float64] = Field(description="(N, 4)")
    azimuth_radians: NDArray[np.float64] = Field(description="(N,)")
    elevation_radians: NDArray[np.float64] = Field(description="(N,)")

    # All tracked points in eye-centered frame
    tracked_points_eye_frame: TrackedPointsEyeFrame

    # All tracked points in camera frame (for debugging/analysis)
    tracked_points_camera_frame: TrackedPointsCameraFrame

    @field_validator("timestamps_seconds", "azimuth_radians", "elevation_radians", mode="before")
    @classmethod
    def validate_1d_array(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Array contains non-finite values")
        return arr

    @field_validator("frame_numbers", mode="before")
    @classmethod
    def validate_frame_numbers(cls, v: NDArray[np.int64] | list) -> NDArray[np.int64]:
        arr = np.asarray(v, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {arr.shape}")
        return arr

    @field_validator("mean_eye_center_camera_mm", "rest_gaze_direction_camera", mode="before")
    @classmethod
    def validate_vec3(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        return arr

    @field_validator("camera_to_eye_rotation", mode="before")
    @classmethod
    def validate_rotation(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"Expected shape (3,3), got {arr.shape}")
        # Check orthogonality
        should_be_I = arr.T @ arr
        if np.max(np.abs(should_be_I - np.eye(3))) > 1e-6:
            raise ValueError("Rotation matrix not orthogonal")
        if abs(np.linalg.det(arr) - 1.0) > 1e-6:
            raise ValueError("Rotation matrix determinant != 1")
        return arr

    @field_validator("gaze_directions", mode="before")
    @classmethod
    def validate_gaze(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {arr.shape}")
        return arr

    @field_validator("quaternions_wxyz", mode="before")
    @classmethod
    def validate_quaternions(cls, v: NDArray[np.float64] | list) -> NDArray[np.float64]:
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"Expected shape (N, 4), got {arr.shape}")
        mags = np.linalg.norm(arr, axis=1)
        if np.max(np.abs(mags - 1.0)) > 1e-6:
            raise ValueError("Quaternions not unit length")
        return arr

    @model_validator(mode="after")
    def validate_shapes_match(self) -> "FerretEyeKinematics":
        n = len(self.timestamps_seconds)

        checks = [
            ("frame_numbers", len(self.frame_numbers)),
            ("gaze_directions", len(self.gaze_directions)),
            ("quaternions_wxyz", len(self.quaternions_wxyz)),
            ("azimuth_radians", len(self.azimuth_radians)),
            ("elevation_radians", len(self.elevation_radians)),
            ("tracked_points_eye_frame", self.tracked_points_eye_frame.num_frames),
            ("tracked_points_camera_frame.eye_center", len(self.tracked_points_camera_frame.eye_center_mm)),
        ]

        for name, length in checks:
            if length != n:
                raise ValueError(f"{name} length {length} != timestamps length {n}")

        # Check timestamps monotonic
        if n > 1 and not np.all(np.diff(self.timestamps_seconds) > 0):
            raise ValueError("Timestamps not monotonically increasing")

        return self

    @property
    def num_frames(self) -> int:
        return len(self.timestamps_seconds)

    @property
    def duration_seconds(self) -> float:
        return float(self.timestamps_seconds[-1] - self.timestamps_seconds[0])

    @property
    def azimuth_degrees(self) -> NDArray[np.float64]:
        return np.degrees(self.azimuth_radians)

    @property
    def elevation_degrees(self) -> NDArray[np.float64]:
        return np.degrees(self.elevation_radians)

    @property
    def residual_eye_center_motion_mm(self) -> NDArray[np.float64]:
        """Per-frame deviation of eye center from mean (measurement error)."""
        return self.tracked_points_camera_frame.eye_center_mm - self.mean_eye_center_camera_mm

    @property
    def camera_motion_residual_magnitude_mm(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.residual_eye_center_motion_mm, axis=1)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with all data."""
        # TODO  - convert this form so its:
        # [frame, timestamp_s, trajectory (e.g. orientation(az/el, rad), gaze(x/y/z), component (x/y/z, etc), value(#), units (mm, deg, etc)]

        data = {}
        


        # data = {
        #     "frame": self.frame_numbers,
        #     "timestamp_s": self.timestamps_seconds,
        #     "azimuth_rad": self.azimuth_radians,
        #     "elevation_rad": self.elevation_radians,
        #     "azimuth_deg": self.azimuth_degrees,
        #     "elevation_deg": self.elevation_degrees,
        #     "gaze_x": self.gaze_directions[:, 0],
        #     "gaze_y": self.gaze_directions[:, 1],
        #     "gaze_z": self.gaze_directions[:, 2],
        #     "quat_w": self.quaternions_wxyz[:, 0],
        #     "quat_x": self.quaternions_wxyz[:, 1],
        #     "quat_y": self.quaternions_wxyz[:, 2],
        #     "quat_z": self.quaternions_wxyz[:, 3],
        #     "camera_motion_residual_mm": self.camera_motion_residual_magnitude_mm,
        # }

        # Add eye-frame tracked points
        tp = self.tracked_points_eye_frame
        data["pupil_center_x"] = tp.pupil_center_mm[:, 0]
        data["pupil_center_y"] = tp.pupil_center_mm[:, 1]
        data["pupil_center_z"] = tp.pupil_center_mm[:, 2]

        for i, name in enumerate(["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]):
            arr = getattr(tp, f"pupil_{name}_mm")
            data[f"pupil_{name}_x"] = arr[:, 0]
            data[f"pupil_{name}_y"] = arr[:, 1]
            data[f"pupil_{name}_z"] = arr[:, 2]

        data["tear_duct_x"] = tp.tear_duct_mm[:, 0] 
        data["tear_duct_y"] = tp.tear_duct_mm[:, 1]
        data["tear_duct_z"] = tp.tear_duct_mm[:, 2]

        data["outer_eye_x"] = tp.outer_eye_mm[:, 0]
        data["outer_eye_y"] = tp.outer_eye_mm[:, 1]
        data["outer_eye_z"] = tp.outer_eye_mm[:, 2]

        return pd.DataFrame(data)

    def get_rigid_body_kinematics_arrays(self) -> dict[str, NDArray[np.float64]]:
        """Get arrays for RigidBodyKinematics.from_pose_arrays()."""
        return {
            "timestamps": self.timestamps_seconds.copy(),
            "position_xyz": np.zeros((self.num_frames, 3), dtype=np.float64),
            "quaternions_wxyz": self.quaternions_wxyz.copy(),
        }


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_ferret_eye_recording(
    csv_path: Path,
    name: str,
    camera: PupilCoreCameraParameters | None = None,
    eye: EyeGeometryParameters | None = None,
    processing_level: str = "cleaned",
    pupil_aggregation: Literal["mean", "median"] = "median",
    rest_direction_method: Literal["mean", "median"] = "median",
) -> FerretEyeKinematics:
    """
    Process ferret eye tracking CSV into complete kinematics.

    Transforms ALL tracked points to eye-centered, rest-aligned frame.
    """
    if camera is None:
        camera = PupilCoreCameraParameters.at_400x400()
    if eye is None:
        eye = EyeGeometryParameters(eye_diameter_mm=FERRET_EYE_DIAMETER_MM)

    # ==========================================================================
    # STEP 1: Load raw data
    # ==========================================================================
    print(f"Loading data from {csv_path}...")
    raw_frames = load_raw_frame_data_from_csv(
        csv_path=csv_path,
        processing_level=processing_level,
        pupil_aggregation=pupil_aggregation,
    )
    print(f"  Loaded {len(raw_frames)} valid frames")

    # ==========================================================================
    # STEP 2: Compute 3D geometry in camera frame
    # ==========================================================================
    print("Computing 3D geometry in camera frame...")

    camera_geoms: list[CameraCenteredEyeballState] = []
    valid_raw_frames: list[RawEyeFrameData] = []

    for raw_frame in raw_frames:
        try:
            geom = compute_frame_geometry_camera_frame(raw_frame, camera, eye)
            camera_geoms.append(geom)
            valid_raw_frames.append(raw_frame)
        except ValueError as e:
            print(f"  Warning: Skipping frame {raw_frame.frame_number}: {e}")

    num_frames = len(camera_geoms)
    print(f"  Valid geometry for {num_frames} frames")

    if num_frames == 0:
        raise ValueError("No frames with valid geometry")

    # ==========================================================================
    # STEP 3: Extract arrays from camera frame geometries
    # ==========================================================================
    timestamps = np.array([g.timestamp_seconds for g in camera_geoms])
    frame_numbers = np.array([g.frame_number for g in camera_geoms], dtype=np.int64)

    eye_centers_camera = np.array([g.eyeball_center_mm for g in camera_geoms])
    pupil_centers_camera = np.array([g.pupil_center_mm for g in camera_geoms])
    gaze_dirs_camera = np.array([g.gaze_direction for g in camera_geoms])
    pupil_points_camera = np.array([g.pupil_points_mm for g in camera_geoms])  # (N, 8, 3)
    tear_duct_camera = np.array([g.tear_duct_mm for g in camera_geoms])
    outer_eye_camera = np.array([g.outer_eye_mm for g in camera_geoms])

    # ==========================================================================
    # STEP 4: Find rest gaze direction
    # ==========================================================================
    print("Finding rest gaze direction...")

    if rest_direction_method == "mean":
        rest_gaze_unnorm = np.mean(gaze_dirs_camera, axis=0)
    else:
        rest_gaze_unnorm = np.median(gaze_dirs_camera, axis=0)

    rest_gaze_camera = rest_gaze_unnorm / np.linalg.norm(rest_gaze_unnorm)
    print(f"  Rest gaze (camera): {rest_gaze_camera}")

    # ==========================================================================
    # STEP 5: Compute rotation to align rest gaze with +X
    # ==========================================================================
    print("Computing alignment rotation...")

    target_rest = np.array([1.0, 0.0, 0.0])
    R = compute_rotation_matrix_aligning_vectors(rest_gaze_camera, target_rest)

    rotated_rest = R @ rest_gaze_camera
    print(f"  Rotated rest gaze: {rotated_rest} (should be [1,0,0])")

    # ==========================================================================
    # STEP 6: Transform ALL points to eye-centered frame
    # ==========================================================================
    print("Transforming all points to eye-centered frame...")

    # Initialize arrays
    pupil_center_eye = np.zeros((num_frames, 3), dtype=np.float64)
    gaze_eye = np.zeros((num_frames, 3), dtype=np.float64)
    pupil_points_eye = np.zeros((num_frames, 8, 3), dtype=np.float64)
    tear_duct_eye = np.zeros((num_frames, 3), dtype=np.float64)
    outer_eye_eye = np.zeros((num_frames, 3), dtype=np.float64)

    for i in range(num_frames):
        # CRITICAL: Use THIS frame's eye center for THIS frame's transform
        eye_center_this_frame = eye_centers_camera[i]

        # Transform each point
        pupil_center_eye[i] = transform_point_to_eye_frame(
            pupil_centers_camera[i], eye_center_this_frame, R
        )

        gaze_eye[i] = R @ gaze_dirs_camera[i]  # Direction, no translation

        pupil_points_eye[i] = transform_points_to_eye_frame(
            pupil_points_camera[i], eye_center_this_frame, R
        )

        tear_duct_eye[i] = transform_point_to_eye_frame(
            tear_duct_camera[i], eye_center_this_frame, R
        )

        outer_eye_eye[i] = transform_point_to_eye_frame(
            outer_eye_camera[i], eye_center_this_frame, R
        )

    # Sanity check: pupil center should be at eye radius from origin
    pupil_dists = np.linalg.norm(pupil_center_eye, axis=1)
    max_dist_error = np.max(np.abs(pupil_dists - eye.eye_radius_mm))
    print(f"  Max pupil distance error: {max_dist_error:.6f} mm")

    # ==========================================================================
    # STEP 7: Compute angles and quaternions
    # ==========================================================================
    print("Computing angles and quaternions...")

    gx, gy, gz = gaze_eye[:, 0], gaze_eye[:, 1], gaze_eye[:, 2]

    azimuth_radians = np.arctan2(gz, gx)
    horiz_dist = np.sqrt(gx**2 + gz**2)
    elevation_radians = np.arctan2(-gy, horiz_dist)

    print(f"  Azimuth: [{np.degrees(azimuth_radians.min()):.2f}°, {np.degrees(azimuth_radians.max()):.2f}°]")
    print(f"  Elevation: [{np.degrees(elevation_radians.min()):.2f}°, {np.degrees(elevation_radians.max()):.2f}°]")

    quaternions = np.zeros((num_frames, 4), dtype=np.float64)
    rest_gaze_eye = np.array([1.0, 0.0, 0.0])

    for i in range(num_frames):
        R_frame = compute_rotation_matrix_aligning_vectors(rest_gaze_eye, gaze_eye[i])
        quaternions[i] = rotation_matrix_to_quaternion_wxyz(R_frame)

    # ==========================================================================
    # STEP 8: Build output
    # ==========================================================================
    print("Building output...")

    tracked_eye = TrackedPointsEyeFrame(
        pupil_center_mm=pupil_center_eye,
        pupil_p1_mm=pupil_points_eye[:, 0, :],
        pupil_p2_mm=pupil_points_eye[:, 1, :],
        pupil_p3_mm=pupil_points_eye[:, 2, :],
        pupil_p4_mm=pupil_points_eye[:, 3, :],
        pupil_p5_mm=pupil_points_eye[:, 4, :],
        pupil_p6_mm=pupil_points_eye[:, 5, :],
        pupil_p7_mm=pupil_points_eye[:, 6, :],
        pupil_p8_mm=pupil_points_eye[:, 7, :],
        tear_duct_mm=tear_duct_eye,
        outer_eye_mm=outer_eye_eye,
    )

    tracked_camera = TrackedPointsCameraFrame(
        eye_center_mm=eye_centers_camera,
        pupil_center_mm=pupil_centers_camera,
        pupil_points_mm=pupil_points_camera,
        tear_duct_mm=tear_duct_camera,
        outer_eye_mm=outer_eye_camera,
    )

    result = FerretEyeKinematics(
        name=name,
        source_csv_path=str(csv_path),
        camera_parameters=camera,
        eye_geometry=eye,
        timestamps_seconds=timestamps,
        frame_numbers=frame_numbers,
        mean_eye_center_camera_mm=np.mean(eye_centers_camera, axis=0),
        rest_gaze_direction_camera=rest_gaze_camera,
        camera_to_eye_rotation=R,
        gaze_directions=gaze_eye,
        quaternions_wxyz=quaternions,
        azimuth_radians=azimuth_radians,
        elevation_radians=elevation_radians,
        tracked_points_eye_frame=tracked_eye,
        tracked_points_camera_frame=tracked_camera,
    )

    print(f"\nProcessing complete!")
    print(f"  Frames: {result.num_frames}")
    print(f"  Duration: {result.duration_seconds:.2f} s")
    print(f"  Max residual: {result.camera_motion_residual_magnitude_mm.max():.3f} mm")

    return result


# =============================================================================
# VALIDATION TEST
# =============================================================================

def run_validation_test() -> None:
    """
    Test with synthetic data.

    NOTE: Uses symmetric sampling (integer number of cycles) so that the mean
    gaze direction equals the true rest direction [0, 0, -1]. This allows us
    to validate the math is correct.

    In real data, the mean gaze may differ from "true rest" if the animal's
    gaze distribution is asymmetric - this is expected and correct behavior.
    """
    print("=" * 70)
    print("VALIDATION TEST")
    print("=" * 70)

    camera = PupilCoreCameraParameters.at_400x400()
    eye = EyeGeometryParameters(eye_diameter_mm=7.0)

    # Use integer number of cycles for symmetric sampling
    # 0.3 Hz azimuth * 10s = 3 full cycles
    # 0.2 Hz elevation * 10s = 2 full cycles
    num_frames = 100
    t = np.linspace(0, 10, num_frames, endpoint=False)  # endpoint=False for exact periodicity

    # Ground truth
    true_az_deg = 15 * np.sin(2 * np.pi * 0.3 * t)
    true_el_deg = 10 * np.cos(2 * np.pi * 0.2 * t)
    true_az_rad = np.radians(true_az_deg)
    true_el_rad = np.radians(true_el_deg)

    # Jitter
    eye_center_base = np.array([0.0, 0.0, 10.0])
    jitter = np.column_stack([
        0.5 * np.sin(2 * np.pi * 3 * t),
        0.3 * np.cos(2 * np.pi * 2 * t),
        0.2 * np.sin(2 * np.pi * 4 * t),
    ])

    print("Generating synthetic frames...")

    # Build frames
    frames: list[RawEyeFrameData] = []

    for i in range(num_frames):
        eye_center = eye_center_base + jitter[i]

        # Gaze direction (rest = -Z)
        az, el = true_az_rad[i], true_el_rad[i]
        gaze = np.array([
            np.sin(az) * np.cos(el),
            -np.sin(el),
            -np.cos(az) * np.cos(el),
        ])
        gaze = gaze / np.linalg.norm(gaze)

        # Pupil center
        pupil = eye_center + eye.eye_radius_mm * gaze

        # Generate p1-p8 around pupil (small circle on sphere)
        pupil_points: list[PixelCoordinate] = []
        for j in range(8):
            angle = 2 * np.pi * j / 8
            # Small offset perpendicular to gaze
            if abs(gaze[0]) < 0.9:
                perp1 = np.cross(gaze, np.array([1, 0, 0]))
            else:
                perp1 = np.cross(gaze, np.array([0, 1, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(gaze, perp1)

            offset = 0.3 * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            p_3d = pupil + offset
            # Project back to sphere
            p_dir = (p_3d - eye_center) / np.linalg.norm(p_3d - eye_center)
            p_on_sphere = eye_center + eye.eye_radius_mm * p_dir

            # To pixels
            px = camera.principal_point_x + camera.focal_length_pixels * p_on_sphere[0] / p_on_sphere[2]
            py = camera.principal_point_y + camera.focal_length_pixels * p_on_sphere[1] / p_on_sphere[2]
            pupil_points.append(PixelCoordinate(x=px, y=py))

        # Pupil center pixel
        pupil_px = PixelCoordinate(
            x=camera.principal_point_x + camera.focal_length_pixels * pupil[0] / pupil[2],
            y=camera.principal_point_y + camera.focal_length_pixels * pupil[1] / pupil[2],
        )

        # Landmarks
        half_w = eye.eye_diameter_mm / 2
        td_3d = eye_center + np.array([-half_w, 0, 0])
        oe_3d = eye_center + np.array([+half_w, 0, 0])

        td_px = PixelCoordinate(
            x=camera.principal_point_x + camera.focal_length_pixels * td_3d[0] / td_3d[2],
            y=camera.principal_point_y + camera.focal_length_pixels * td_3d[1] / td_3d[2],
        )
        oe_px = PixelCoordinate(
            x=camera.principal_point_x + camera.focal_length_pixels * oe_3d[0] / oe_3d[2],
            y=camera.principal_point_y + camera.focal_length_pixels * oe_3d[1] / oe_3d[2],
        )

        frames.append(RawEyeFrameData(
            frame_number=i,
            timestamp_seconds=t[i],
            pupil_points=tuple(pupil_points),
            pupil_center=pupil_px,
            tear_duct=td_px,
            outer_eye=oe_px,
        ))

    # Process
    print("Processing...")

    cam_geoms = [compute_frame_geometry_camera_frame(f, camera, eye) for f in frames]

    gaze_camera = np.array([g.gaze_direction for g in cam_geoms])
    eye_centers = np.array([g.eyeball_center_mm for g in cam_geoms])

    rest_gaze = np.mean(gaze_camera, axis=0)
    rest_gaze = rest_gaze / np.linalg.norm(rest_gaze)

    R = compute_rotation_matrix_aligning_vectors(rest_gaze, np.array([1, 0, 0]))

    # Verify mean gaze is close to true rest (should be with symmetric sampling)
    true_rest = np.array([0.0, 0.0, -1.0])
    mean_gaze_angle = np.degrees(np.arccos(np.clip(np.dot(rest_gaze, true_rest), -1, 1)))
    print(f"  Mean gaze vs true rest angle: {mean_gaze_angle:.4f}° (should be ~0 with symmetric sampling)")

    if mean_gaze_angle > 0.5:
        print(f"  WARNING: Mean gaze differs from true rest by {mean_gaze_angle:.2f}°")
        print(f"           This will cause a systematic offset in recovered angles.")
        print(f"           For validation, use symmetric sampling (integer number of cycles).")

    gaze_eye = (R @ gaze_camera.T).T

    recovered_az = np.arctan2(gaze_eye[:, 2], gaze_eye[:, 0])
    horiz = np.sqrt(gaze_eye[:, 0]**2 + gaze_eye[:, 2]**2)
    recovered_el = np.arctan2(-gaze_eye[:, 1], horiz)

    az_err = np.degrees(recovered_az - true_az_rad)
    el_err = np.degrees(recovered_el - true_el_rad)

    print(f"\n--- RESULTS ---")
    print(f"Azimuth error: mean={np.mean(np.abs(az_err)):.4f}°, max={np.max(np.abs(az_err)):.4f}°")
    print(f"Elevation error: mean={np.mean(np.abs(el_err)):.4f}°, max={np.max(np.abs(el_err)):.4f}°")

    if np.max(np.abs(az_err)) < 0.5 and np.max(np.abs(el_err)) < 0.5:
        print("\n✓ VALIDATION PASSED")
    else:
        print("\n✗ VALIDATION FAILED")


if __name__ == "__main__":
    run_validation_test()