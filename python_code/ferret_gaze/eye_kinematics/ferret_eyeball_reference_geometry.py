"""
Ferret Eyeball Reference Geometry
=================================

Coordinate System (at rest, right-handed):
    Origin: Eyeball center [0, 0, 0]
    +Z: Rest gaze direction (pupil_center at [0, 0, eye_radius]) - "north pole"
    +Y: Superior (up)
    +X: Computed via Y × Z = subject's LEFT

This means +X points to the subject's LEFT for both eyes.

For anatomical directions that differ between eyes (medial/lateral),
use the anatomical accessors in FerretEyeKinematics.
"""
from typing import Literal

import numpy as np

from python_code.kinematics_core.reference_geometry_model import (
    ReferenceGeometry,
    MarkerPosition,
    CoordinateFrameDefinition,
    AxisDefinition,
    AxisType,
)

PUPIL_KEYPOINT_NAMES: tuple[str, ...] = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")
NUM_PUPIL_POINTS: int = 8
FERRET_EYE_RADIUS_MM: float = 3.5
DEFAULT_FERRET_EYE_PUPIL_RADIUS_MM: float = 0.5
DEFAULT_FERRET_EYE_PUPIL_ECCENTRICITY: float = 0.8


def create_eyeball_reference_geometry(
    eye_radius_mm: float = FERRET_EYE_RADIUS_MM,
    default_pupil_radius_mm: float = DEFAULT_FERRET_EYE_PUPIL_RADIUS_MM,
    default_pupil_eccentricity: float = DEFAULT_FERRET_EYE_PUPIL_ECCENTRICITY,
) -> ReferenceGeometry:
    """
    Create reference geometry for the EYEBALL ONLY.

    This does NOT include tear_duct or outer_eye - those belong to the socket frame.

    Coordinate system (at rest, right-handed):
        Origin: Eyeball center [0, 0, 0]
        +Z: Rest gaze direction (pupil_center at [0, 0, eye_radius]) - "north pole"
        +Y: Superior (up)  
        +X: Subject's left (computed via Y × Z)

    Anatomical interpretation of +X:
        - RIGHT eye: +X = medial (toward nose)
        - LEFT eye: +X = lateral (away from nose)

    Markers:
        - eyeball_center: always [0, 0, 0]
        - pupil_center: [0, 0, eye_radius] at rest
        - p1-p8: pupil boundary points in ellipse around pupil_center
            - p1: +X direction at rest (subject's left)
            - p3: +Y direction at rest (up)
            - p5: -X direction at rest (subject's right)
            - p7: -Y direction at rest (down)

    Args:
        eye_radius_mm: Radius of eyeball
        default_pupil_radius_mm: Semi-major axis of pupil ellipse
        default_pupil_eccentricity: Ratio of minor/major axis (1.0 = circle)

    Returns:
        ReferenceGeometry for the eyeball rigid body
    """
    R = eye_radius_mm
    a = default_pupil_radius_mm  # Semi-major (along X in tangent plane)
    b = default_pupil_radius_mm * default_pupil_eccentricity  # Semi-minor (along Y)

    keypoints: dict[str, MarkerPosition] = {}

    # Eyeball center (origin)
    keypoints["eyeball_center"] = MarkerPosition(x=0.0, y=0.0, z=0.0)

    # Pupil center at +Z (north pole of sphere) - this is the rest gaze direction
    keypoints["pupil_center"] = MarkerPosition(x=0.0, y=0.0, z=R)

    # Pupil boundary points p1-p8 in ellipse around pupil center
    # The ellipse is in the XY tangent plane at z=R, then projected onto sphere
    # phi=0 → +X (right), phi=pi/2 → +Y (up), phi=pi → -X (left), phi=3pi/2 → -Y (down)
    for i in range(NUM_PUPIL_POINTS):
        phi = 2 * np.pi * i / NUM_PUPIL_POINTS
        x_tangent = a * np.cos(phi)
        y_tangent = b * np.sin(phi)
        # Point on tangent plane at z=R
        tangent_point = np.array([x_tangent, y_tangent, R])
        # Project onto sphere surface
        direction = tangent_point / np.linalg.norm(tangent_point)
        sphere_point = R * direction
        keypoints[f"p{i + 1}"] = MarkerPosition(
            x=float(sphere_point[0]),
            y=float(sphere_point[1]),
            z=float(sphere_point[2]),
        )

    # Coordinate frame definition:
    # - z_axis (EXACT): points along gaze direction (toward pupil_center)
    # - y_axis (APPROXIMATE): points "up" (toward p3 which is at +Y in tangent plane)
    # - x_axis: computed via Y × Z for right-handed system → points right
    #
    # p3 is at phi = 2*pi*2/8 = pi/2, so it's at x=0, y=+b (top of ellipse)
    coordinate_frame = CoordinateFrameDefinition(
        origin_keypoints=["eyeball_center"],
        z_axis=AxisDefinition(keypoints=["pupil_center"], type=AxisType.EXACT),
        y_axis=AxisDefinition(keypoints=["p3"], type=AxisType.APPROXIMATE),
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