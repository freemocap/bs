from typing import Literal

import numpy as np

from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry, MarkerPosition, \
    CoordinateFrameDefinition, AxisDefinition, AxisType

PUPIL_KEYPOINT_NAMES: tuple[str, ...] = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")
NUM_PUPIL_POINTS: int = 8
FERRET_EYE_RADIUS_MM: float = 3.5
DEFAULT_FERRET_EYE_PUPIL_RADIUS_MM: float = 0.5
DEFAULT_FERRET_EYE_PUPIL_ECCENTRICITY: float = 0.8

def create_eyeball_reference_geometry(
    eye_radius_mm: float = FERRET_EYE_RADIUS_MM,
    default_pupil_radius_mm: float = DEFAULT_FERRET_EYE_PUPIL_RADIUS_MM,
    default_pupil_eccentricity: float = DEFAULT_FERRET_EYE_PUPIL_ECCENTRICITY) -> ReferenceGeometry:
    """
    Create reference geometry for the EYEBALL ONLY.

    This does NOT include tear_duct or outer_eye - those belong to the socket frame.

    Coordinate system (at rest):
        Origin: Eyeball center [0, 0, 0]
        +Z: Rest gaze direction (pupil_center is at [0,0,eye_radius])
        +X: Subject's left (perpendicular to gaze, in horizontal plane)
        +Y: Superior (up)

    Note: This is a right-handed coordinate system. The anatomical
    interpretation of +Y depends on which eye (medial for right, lateral for left).

    Markers:
        - eyeball_center: always [0, 0, 0]
        - pupil_center: [0,0,eye_radius] at rest
        - p1-p8: pupil boundary points in ellipse around pupil_center

    Args:
        eye_radius_mm: Radius of eyeball
        default_pupil_radius_mm: Semi-major axis of pupil ellipse (horizontal extent)
        default_pupil_eccentricity: Ratio of minor/major axis (1.0 = circle)

    Returns:
        ReferenceGeometry for the eyeball rigid body
    """
    R = eye_radius_mm
    a = default_pupil_radius_mm  # Semi-major (horizontal, along Y in tangent plane)
    b = default_pupil_radius_mm * default_pupil_eccentricity  # Semi-minor (vertical, along Z)

    keypoints: dict[str, MarkerPosition] = {}

    # Eyeball center (origin)
    keypoints["eyeball_center"] = MarkerPosition(x=0.0, y=0.0, z=0.0)

    # Pupil center (at +X on sphere surface)
    keypoints["pupil_center"] = MarkerPosition(x=0.0, y=0.0, z=R)

    # Pupil boundary points p1-p8 in ellipse around pupil center
    for i in range(NUM_PUPIL_POINTS):
        phi = 2 * np.pi * i / NUM_PUPIL_POINTS
        y_tangent = a * np.cos(phi)
        z_tangent = b * np.sin(phi)
        tangent_point = np.array([R, y_tangent, z_tangent])
        direction = tangent_point / np.linalg.norm(tangent_point)
        sphere_point = R * direction
        keypoints[f"p{i + 1}"] = MarkerPosition(
            x=float(sphere_point[0]),
            y=float(sphere_point[1]),
            z=float(sphere_point[2]),
        )

    coordinate_frame = CoordinateFrameDefinition(
        origin_keypoints=["eyeball_center"],
        z_axis=AxisDefinition(keypoints=["pupil_center"], type=AxisType.EXACT), # pointing "forward", down gaze/pupil center line
        y_axis=AxisDefinition(keypoints=["p2"],type=AxisType.APPROXIMATE),  # pointing "up" (superior/dorsal)
    )

    # Rigid edges (only one between eyeball_center and pupil_center)
    rigid_edges: list[tuple[str, str]] = [
        ("eyeball_center", "pupil_center"),
    ]
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
        keypoints=keypoints,
        display_edges=display_edges,
        rigid_edges=rigid_edges,
    )


