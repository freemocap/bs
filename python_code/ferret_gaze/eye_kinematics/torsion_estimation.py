"""
Torsion Estimation from Pupil Ellipse Orientation
=================================================

This module estimates eye torsion (roll around the gaze axis) by analyzing
the orientation of the p1-p8 pupil boundary ellipse.

The key insight:
- Gaze direction (pitch, yaw) can be determined from pupil center position
- Torsion CANNOT be determined from pupil center alone
- But the p1-p8 ellipse orientation around the gaze axis encodes torsion

Algorithm:
1. For each frame, project p1-p8 onto the plane perpendicular to gaze
2. Fit an ellipse (via PCA) to find the major axis orientation
3. The angle of the major axis relative to a reference direction = torsion

REST POSITION (quaternion = [1, 0, 0, 0]):
- Pupil center at [+R, 0, 0] on the +X axis
- Pupil ellipse major axis along Y (equator of the eye sphere)
- Pupil ellipse minor axis along Z (prime meridian)
- Zero torsion = major axis aligned with the equator

Assumptions:
- p1-p8 are ordered consistently (p1 at ~0°, p2 at ~45°, etc.)
- At zero torsion, the major axis is roughly horizontal (along ±Y)
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def estimate_ellipse_orientation_pca(
    points_2d: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate ellipse orientation from 2D points using PCA.

    Args:
        points_2d: (N, 2) array of points

    Returns:
        angle: Angle of major axis from +x direction (radians)
        major_axis: (2,) unit vector along major axis
        minor_axis: (2,) unit vector along minor axis
    """
    # Center the points
    centroid = np.mean(points_2d, axis=0)
    centered = points_2d - centroid

    # Covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Major axis has larger eigenvalue
    if eigenvalues[1] >= eigenvalues[0]:
        major_axis = eigenvectors[:, 1]
        minor_axis = eigenvectors[:, 0]
    else:
        major_axis = eigenvectors[:, 0]
        minor_axis = eigenvectors[:, 1]

    # Angle of major axis from +x
    angle = np.arctan2(major_axis[1], major_axis[0])

    return angle, major_axis, minor_axis


def build_gaze_perpendicular_basis(
    gaze: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build orthonormal basis vectors in the plane perpendicular to gaze.

    Uses world +Z (superior) as reference for "up" direction.
    At rest (gaze = [1,0,0]), this gives:
    - v_up = [0, 0, 1] (world +Z)
    - v_right = [0, -1, 0] (world -Y, since gaze × up with right-hand rule)

    Args:
        gaze: (3,) unit vector of gaze direction

    Returns:
        v_up: (3,) "up" vector in gaze-perpendicular plane
        v_right: (3,) "right" vector in gaze-perpendicular plane
    """
    # Reference "up" is world +Z (superior)
    world_up = np.array([0.0, 0.0, 1.0])

    # Project onto plane perpendicular to gaze
    up_in_plane = world_up - np.dot(world_up, gaze) * gaze
    norm = np.linalg.norm(up_in_plane)

    if norm < 1e-6:
        # Gaze is nearly vertical, use world +Y as backup
        world_up = np.array([0.0, 1.0, 0.0])
        up_in_plane = world_up - np.dot(world_up, gaze) * gaze
        norm = np.linalg.norm(up_in_plane)

    v_up = up_in_plane / norm

    # Right = gaze × up (right-handed)
    v_right = np.cross(gaze, v_up)
    v_right = v_right / np.linalg.norm(v_right)

    return v_up, v_right


def project_points_to_gaze_plane(
    points_3d: NDArray[np.float64],
    gaze: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Project 3D points onto 2D plane perpendicular to gaze.

    Args:
        points_3d: (N, 3) array of 3D points
        gaze: (3,) unit vector of gaze direction

    Returns:
        points_2d: (N, 2) array where x=right, y=up in gaze plane
    """
    v_up, v_right = build_gaze_perpendicular_basis(gaze)

    # Center points
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid

    # Project onto 2D basis
    n_points = len(points_3d)
    points_2d = np.zeros((n_points, 2), dtype=np.float64)

    for i in range(n_points):
        points_2d[i, 0] = np.dot(centered[i], v_right)  # x = right
        points_2d[i, 1] = np.dot(centered[i], v_up)     # y = up

    return points_2d


def estimate_single_frame_torsion(
    pupil_points: NDArray[np.float64],
    gaze: NDArray[np.float64],
) -> float:
    """
    Estimate torsion for a single frame.

    At rest (gaze = [1,0,0], zero torsion):
    - The pupil ellipse has major axis along Y (equator)
    - When projected to gaze-perpendicular plane:
      - v_right = -Y direction
      - v_up = +Z direction
    - Major axis projects onto the "right" direction (x in 2D)
    - So at zero torsion, angle ≈ 0

    Args:
        pupil_points: (8, 3) array of pupil boundary points
        gaze: (3,) unit vector of gaze direction

    Returns:
        Raw torsion angle in radians (before anatomical sign correction)
    """
    # Project pupil points to gaze-perpendicular plane
    points_2d = project_points_to_gaze_plane(pupil_points, gaze)

    # Fit ellipse to get major axis orientation
    angle, _, _ = estimate_ellipse_orientation_pca(points_2d)

    # Normalize angle to [-π/2, π/2] (ellipse has 180° ambiguity)
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi

    return angle


def estimate_torsion_timeseries(
    pupil_points: NDArray[np.float64],
    gaze_directions: NDArray[np.float64],
    eye_side: Literal["left", "right"],
    zero_reference: Literal["median", "first", "none"] = "median",
) -> NDArray[np.float64]:
    """
    Estimate torsion angle for each frame from pupil ellipse orientation.

    Args:
        pupil_points: (N, 8, 3) array of pupil boundary points
        gaze_directions: (N, 3) array of gaze unit vectors
        eye_side: "left" or "right" for anatomical sign correction
        zero_reference: How to establish zero torsion
            - "median": Use median angle as zero (robust)
            - "first": Use first frame as zero
            - "none": No offset correction

    Returns:
        (N,) array of torsion angles in radians
        Positive = extorsion (top of eye tilts away from nose)
        Negative = intorsion (top of eye tilts toward nose)
    """
    n_frames = len(pupil_points)
    raw_angles = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        raw_angles[i] = estimate_single_frame_torsion(
            pupil_points[i],
            gaze_directions[i],
        )

    # Apply zero reference correction
    if zero_reference == "median":
        offset = np.median(raw_angles)
    elif zero_reference == "first":
        offset = raw_angles[0]
    else:
        offset = 0.0

    torsion = raw_angles - offset

    # Apply anatomical sign correction
    # Convention: positive torsion = extorsion (top of eye tilts laterally)
    #
    # For RIGHT eye: In the gaze-perpendicular plane, "up" is superior (+Z).
    #   At zero torsion, major axis is horizontal (along ±Y in eye frame).
    #   Extorsion = top of eye tilts toward -Y = lateral.
    #   This corresponds to major axis rotating CCW (positive angle).
    #   → Sign = +1
    #
    # For LEFT eye: Same "up" reference.
    #   Extorsion = top of eye tilts toward +Y = lateral.
    #   This corresponds to major axis rotating CW (negative angle).
    #   → Sign = -1

    anatomical_sign = 1.0 if eye_side == "right" else -1.0
    torsion = anatomical_sign * torsion

    return torsion


def compute_torsion_enhanced_quaternions(
    gaze_directions: NDArray[np.float64],
    torsion_angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute quaternions that encode both gaze direction AND torsion.

    The standard pipeline only computes quaternions from gaze direction,
    which loses torsion information. This function adds torsion back.

    Args:
        gaze_directions: (N, 3) array of gaze unit vectors in eye frame
        torsion_angles: (N,) array of torsion angles in radians

    Returns:
        (N, 4) array of quaternions [w, x, y, z] encoding full 3-DOF rotation
    """
    n_frames = len(gaze_directions)
    quaternions = np.zeros((n_frames, 4), dtype=np.float64)

    rest_gaze = np.array([1.0, 0.0, 0.0])

    for i in range(n_frames):
        gaze = gaze_directions[i]
        torsion = torsion_angles[i]

        # Step 1: Compute rotation that aligns rest_gaze to current gaze
        # (This is what the original pipeline does)
        q_gaze = _rotation_aligning_vectors_to_quaternion(rest_gaze, gaze)

        # Step 2: Compute torsion rotation around gaze axis
        # Quaternion for rotation by angle θ around axis a:
        # q = [cos(θ/2), sin(θ/2) * a]
        half_angle = torsion / 2.0
        q_torsion = np.array([
            np.cos(half_angle),
            np.sin(half_angle) * gaze[0],
            np.sin(half_angle) * gaze[1],
            np.sin(half_angle) * gaze[2],
        ])

        # Step 3: Compose: q_total = q_torsion * q_gaze
        # (Torsion applied AFTER gaze alignment)
        quaternions[i] = _quaternion_multiply(q_torsion, q_gaze)

    return quaternions


def _rotation_aligning_vectors_to_quaternion(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute quaternion that rotates source vector to target vector."""
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)

    dot = np.dot(a, b)

    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])

    if dot < -0.999999:
        # 180° rotation - find perpendicular axis
        if abs(a[0]) < 0.9:
            perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
        else:
            perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
        perp = perp / np.linalg.norm(perp)
        return np.array([0.0, perp[0], perp[1], perp[2]])

    # General case: rotation axis = a × b, angle from dot product
    axis = np.cross(a, b)

    # Quaternion: q = [cos(θ/2), sin(θ/2) * axis]
    # Using half-angle formulas:
    # cos(θ/2) = sqrt((1 + dot) / 2)
    # sin(θ/2) * axis = axis / (2 * cos(θ/2)) = axis / sqrt(2 * (1 + dot))

    w = np.sqrt((1.0 + dot) / 2.0)
    xyz_scale = 1.0 / np.sqrt(2.0 * (1.0 + dot))

    q = np.array([w, axis[0] * xyz_scale, axis[1] * xyz_scale, axis[2] * xyz_scale])
    return q / np.linalg.norm(q)


def _quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply two quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def add_torsion_to_kinematics_pipeline(
    gaze_directions_eye: NDArray[np.float64],
    pupil_points_eye: NDArray[np.float64],
    eye_side: Literal["left", "right"],
) -> NDArray[np.float64]:
    """
    Enhanced quaternion computation that includes torsion from pupil ellipse.

    Drop-in replacement for the basic gaze-only quaternion computation.

    Args:
        gaze_directions_eye: (N, 3) gaze unit vectors in eye frame
        pupil_points_eye: (N, 8, 3) pupil boundary points in eye frame
        eye_side: "left" or "right"

    Returns:
        (N, 4) quaternions [w, x, y, z] encoding gaze + torsion
    """
    # Estimate torsion from pupil ellipse
    torsion = estimate_torsion_timeseries(
        pupil_points=pupil_points_eye,
        gaze_directions=gaze_directions_eye,
        eye_side=eye_side,
        zero_reference="median",
    )

    # Compute quaternions with torsion
    quaternions = compute_torsion_enhanced_quaternions(
        gaze_directions=gaze_directions_eye,
        torsion_angles=torsion,
    )

    return quaternions


# =============================================================================
# TESTING
# =============================================================================

def test_torsion_estimation():
    """Test torsion estimation with synthetic data."""
    print("=" * 60)
    print("TORSION ESTIMATION TEST")
    print("=" * 60)

    n_frames = 100
    t = np.linspace(0, 2.0, n_frames)

    # True torsion: oscillates ±10°
    true_torsion = np.radians(10.0) * np.sin(2 * np.pi * 1.0 * t)

    # Gaze: slight oscillation
    gaze_directions = np.zeros((n_frames, 3), dtype=np.float64)
    azimuth = np.radians(5.0) * np.sin(2 * np.pi * 0.5 * t)
    gaze_directions[:, 0] = np.cos(azimuth)  # X
    gaze_directions[:, 1] = np.sin(azimuth)  # Y
    gaze_directions[:, 2] = 0.0  # Z

    # Generate pupil points with known torsion
    pupil_radius = 0.5
    eccentricity = 0.8
    pupil_points = np.zeros((n_frames, 8, 3), dtype=np.float64)

    for i in range(n_frames):
        gaze = gaze_directions[i]
        v_up, v_right = build_gaze_perpendicular_basis(gaze)

        # Apply torsion to basis (CCW rotation when looking along gaze)
        # Positive torsion = extorsion = top of eye tilts laterally
        cos_t = np.cos(true_torsion[i])
        sin_t = np.sin(true_torsion[i])
        # Standard CCW rotation: [cos -sin; sin cos]
        v_up_rot = cos_t * v_up - sin_t * v_right
        v_right_rot = sin_t * v_up + cos_t * v_right

        # Generate ellipse points
        # At rest: major axis along "right" (which projects from Y)
        # minor axis along "up" (which projects from Z)
        a = pupil_radius  # major (along right)
        b = pupil_radius * eccentricity  # minor (along up)

        for j in range(8):
            phi = 2 * np.pi * j / 8
            local_right = a * np.cos(phi)
            local_up = b * np.sin(phi)
            pupil_points[i, j] = local_right * v_right_rot + local_up * v_up_rot

    # Estimate torsion
    estimated = estimate_torsion_timeseries(
        pupil_points=pupil_points,
        gaze_directions=gaze_directions,
        eye_side="right",
        zero_reference="median",
    )

    # Compare
    correlation = np.corrcoef(true_torsion, estimated)[0, 1]
    rmse = np.sqrt(np.mean((true_torsion - estimated) ** 2))

    print(f"\nResults:")
    print(f"  True torsion range: ±{np.degrees(np.max(np.abs(true_torsion))):.1f}°")
    print(f"  Estimated range: ±{np.degrees(np.max(np.abs(estimated))):.1f}°")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  RMSE: {np.degrees(rmse):.2f}°")

    assert correlation > 0.99, f"Correlation too low: {correlation}"
    assert np.degrees(rmse) < 1.0, f"RMSE too high: {np.degrees(rmse)}°"

    print("\n✓ Test passed!")
    return true_torsion, estimated


if __name__ == "__main__":
    test_torsion_estimation()