"""
Ferret Eye Kinematics Pipeline (Refactored)
==========================================

This module processes raw eye tracking CSV data and produces FerretEyeKinematics
with proper separation of eyeball kinematics and socket landmarks.

The key change from the original pipeline is that we now:
1. Create a RigidBodyKinematics for the eyeball (with angular velocity computation)
2. Keep socket landmarks separate (they don't rotate with the eye)
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_model import FerretEyeKinematics
from python_code.ferret_gaze.eye_kinematics.eyeball_viewer import create_animated_eye_figure


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


def compute_gaze_quaternion(
    gaze_direction_eye_frame: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute quaternion that rotates [1, 0, 0] (rest gaze) to the given gaze direction.

    Args:
        gaze_direction_eye_frame: (3,) unit vector in eye-centered frame

    Returns:
        (4,) quaternion as [w, x, y, z]
    """
    rest_gaze = np.array([1.0, 0.0, 0.0])
    R = compute_rotation_matrix_aligning_vectors(rest_gaze, gaze_direction_eye_frame)
    return rotation_matrix_to_quaternion_wxyz(R)


def process_ferret_eye_data(
    name: str,
    source_path: str,
    eye_side: Literal["left", "right"],
    timestamps: NDArray[np.float64],
    gaze_directions_camera: NDArray[np.float64],
    pupil_centers_camera: NDArray[np.float64],
    pupil_points_camera: NDArray[np.float64],
    eye_centers_camera: NDArray[np.float64],
    tear_duct_camera: NDArray[np.float64],
    outer_eye_camera: NDArray[np.float64],
    eye_radius_mm: float = 3.5,
    pupil_radius_mm: float = 0.5,
    pupil_eccentricity: float = 0.8,
) -> FerretEyeKinematics:
    """
    Process camera-frame eye tracking data into FerretEyeKinematics.

    Output Coordinate System (RIGHT-HANDED for both eyes):
        +X = Anterior (gaze direction at rest)
        +Y = Subject's left (medial for right eye, lateral for left eye)
        +Z = Superior (up)

    This keeps BOTH eyes in right-handed coordinate systems, which is essential
    for quaternion math, cross products, and rotation matrices to work correctly.

    The anatomical accessors (adduction_angle, torsion_angle, etc.) flip signs
    based on eye_side to provide consistent anatomical meaning despite the
    different anatomical interpretation of +Y for each eye.

    Args:
        name: Identifier for this recording
        source_path: Path to source data
        eye_side: "left" or "right"
        timestamps: (N,) timestamps in seconds
        gaze_directions_camera: (N, 3) gaze unit vectors in camera frame
        pupil_centers_camera: (N, 3) pupil center positions in camera frame
        pupil_points_camera: (N, 8, 3) pupil boundary points in camera frame
        eye_centers_camera: (N, 3) eyeball center positions in camera frame
        tear_duct_camera: (N, 3) tear duct positions in camera frame
        outer_eye_camera: (N, 3) outer eye positions in camera frame
        eye_radius_mm: Eyeball radius
        pupil_radius_mm: Pupil ellipse semi-major axis
        pupil_eccentricity: Pupil ellipse eccentricity

    Returns:
        FerretEyeKinematics with computed angular velocities
    """
    n_frames = len(timestamps)

    # Step 1: Compute rest gaze direction (median gaze in camera frame)
    rest_gaze_camera = np.median(gaze_directions_camera, axis=0)
    rest_gaze_camera = rest_gaze_camera / np.linalg.norm(rest_gaze_camera)

    # Step 2: Compute the Y-axis direction in camera frame
    # We use a CONSISTENT world direction for +Y (not anatomical)
    # to maintain right-handed coordinates for both eyes.
    #
    # Convention: +Y points toward subject's left in camera frame
    # For typical camera setup (looking at subject's face):
    #   - Camera +X = subject's left
    #   - Camera +Y = down
    #   - Camera +Z = into scene (toward subject)
    #
    # So we want eye +Y to roughly align with camera +X (subject's left)

    # Get approximate "subject's left" direction, orthogonal to gaze
    # We'll use the tear_duct-to-outer_eye vector, but consistently oriented
    mean_tear_duct = np.mean(tear_duct_camera, axis=0)
    mean_outer_eye = np.mean(outer_eye_camera, axis=0)

    # For RIGHT eye: tear_duct is medial (subject's left), outer_eye is lateral
    # For LEFT eye: tear_duct is medial (subject's right), outer_eye is lateral
    # So tear_duct - outer_eye points toward subject's left for right eye,
    # and toward subject's right for left eye.

    if eye_side == "right":
        # tear_duct is to subject's left of outer_eye
        y_approx_camera = mean_tear_duct - mean_outer_eye
    else:
        # tear_duct is to subject's right of outer_eye
        # We want +Y to point to subject's left, so flip
        y_approx_camera = mean_outer_eye - mean_tear_duct

    # Step 3: Build orthonormal basis (Gram-Schmidt)
    # X = gaze direction (exact)
    x_camera = rest_gaze_camera

    # Y = orthogonalize y_approx against X
    y_camera = y_approx_camera - np.dot(y_approx_camera, x_camera) * x_camera
    y_camera = y_camera / np.linalg.norm(y_camera)

    # Z = X × Y (right-hand rule: forward × left = up)
    z_camera = np.cross(x_camera, y_camera)
    z_camera = z_camera / np.linalg.norm(z_camera)

    # Step 4: Build rotation matrix from camera to eye frame
    # R @ v_camera = v_eye, where eye frame has X,Y,Z as identity basis
    R_camera_to_eye = np.column_stack([x_camera, y_camera, z_camera]).T

    # Step 3: Transform gaze directions to eye-centered frame
    # gaze_eye[i] = R @ gaze_camera[i]
    gaze_directions_eye = (R_camera_to_eye @ gaze_directions_camera.T).T

    # Normalize gaze directions
    gaze_norms = np.linalg.norm(gaze_directions_eye, axis=1, keepdims=True)
    gaze_directions_eye = gaze_directions_eye / gaze_norms

    # Step 4: Compute quaternion for each frame
    # Quaternion rotates [1, 0, 0] (rest gaze) to current gaze
    quaternions_wxyz = np.zeros((n_frames, 4), dtype=np.float64)
    for i in range(n_frames):
        quaternions_wxyz[i] = compute_gaze_quaternion(gaze_directions_eye[i])

    # Step 5: Transform socket landmarks to eye-centered frame
    # These are transformed but NOT rotated by eye orientation
    # (they're fixed in the skull frame, which we've aligned to eye rest frame)

    # Get mean eye center in camera frame
    mean_eye_center_camera = np.mean(eye_centers_camera, axis=0)

    # Transform landmarks: subtract eye center, then rotate to eye frame
    def transform_points_to_eye_frame(
        points_camera: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Transform points from camera to eye-centered frame."""
        # Subtract mean eye center (puts points relative to eye)
        relative = points_camera - mean_eye_center_camera
        # Rotate to eye-centered frame
        return (R_camera_to_eye @ relative.T).T

    tear_duct_eye = transform_points_to_eye_frame(tear_duct_camera)
    outer_eye_eye = transform_points_to_eye_frame(outer_eye_camera)

    # Step 6: Create FerretEyeKinematics
    # This will create RigidBodyKinematics for the eyeball with:
    # - Position always at [0, 0, 0]
    # - Orientation from quaternions
    # - Angular velocity computed automatically
    # - Keypoint trajectories computed from reference geometry + orientation

    return FerretEyeKinematics.from_pose_data(
        name=name,
        source_path=source_path,
        eye_side=eye_side,
        timestamps=timestamps,
        quaternions_wxyz=quaternions_wxyz,
        tear_duct_mm=tear_duct_eye,
        outer_eye_mm=outer_eye_eye,
        rest_gaze_direction_camera=rest_gaze_camera,
        camera_to_eye_rotation=R_camera_to_eye,
        eye_radius_mm=eye_radius_mm,
        pupil_radius_mm=pupil_radius_mm,
        pupil_eccentricity=pupil_eccentricity,
    )


def run_eye_kinematics_example() -> FerretEyeKinematics:
    """
    Demonstrate the new architecture with synthetic data.

    Shows:
    1. How pupil positions are computed from orientation
    2. How socket landmarks remain fixed
    3. Angular velocity computation
    """
    print("=" * 70)
    print("EYE KINEMATICS ARCHITECTURE DEMO")
    print("=" * 70)

    # Generate synthetic eye movement data
    # Eye oscillates left-right (yaw) with some elevation change
    n_frames = 100
    t = np.linspace(0, 2.0, n_frames)  # 2 seconds

    # Azimuth: +/- 15 degrees at 1 Hz
    azimuth_rad = np.radians(15.0) * np.sin(2 * np.pi * 1.0 * t)

    # Elevation: +/- 5 degrees at 0.5 Hz
    elevation_rad = np.radians(5.0) * np.sin(2 * np.pi * 0.5 * t)

    # Convert to gaze directions in camera frame
    # Assume rest gaze is -Z (into screen)
    gaze_camera = np.zeros((n_frames, 3), dtype=np.float64)
    gaze_camera[:, 0] = np.sin(azimuth_rad) * np.cos(elevation_rad)  # X
    gaze_camera[:, 1] = -np.sin(elevation_rad)  # Y (down is positive in camera)
    gaze_camera[:, 2] = -np.cos(azimuth_rad) * np.cos(elevation_rad)  # Z (into screen)

    # Eye center in camera frame (fixed with some jitter)
    eye_center_base = np.array([0.0, 0.0, 10.0])  # 10mm from camera
    jitter = 0.1 * np.random.randn(n_frames, 3)  # Small tracking noise
    eye_centers_camera = eye_center_base + jitter

    # Pupil centers (on sphere surface)
    eye_radius_mm = 3.5
    pupil_centers_camera = eye_centers_camera + eye_radius_mm * gaze_camera

    # Pupil boundary points (not needed for this demo)
    pupil_points_camera = np.zeros((n_frames, 8, 3), dtype=np.float64)

    # Socket landmarks (fixed in skull frame, with noise)
    # Tear duct is medial (+X in camera), outer eye is lateral (-X)
    tear_duct_base = eye_center_base + np.array([eye_radius_mm, 0, -eye_radius_mm])
    outer_eye_base = eye_center_base + np.array([-eye_radius_mm, 0, -eye_radius_mm])

    tear_duct_camera = tear_duct_base + 0.05 * np.random.randn(n_frames, 3)
    outer_eye_camera = outer_eye_base + 0.05 * np.random.randn(n_frames, 3)

    # Process data
    print("\nProcessing synthetic eye data...")
    kinematics = process_ferret_eye_data(
        name="synthetic_eye",
        source_path="<synthetic>",
        eye_side="right",
        timestamps=t,
        gaze_directions_camera=gaze_camera,
        pupil_centers_camera=pupil_centers_camera,
        pupil_points_camera=pupil_points_camera,
        eye_centers_camera=eye_centers_camera,
        tear_duct_camera=tear_duct_camera,
        outer_eye_camera=outer_eye_camera,
        eye_radius_mm=eye_radius_mm,
    )

    print(f"\nKinematics created:")
    print(f"  Name: {kinematics.name}")
    print(f"  Frames: {kinematics.n_frames}")
    print(f"  Duration: {kinematics.duration_seconds:.2f} s")

    # Check eyeball kinematics
    print(f"\nEyeball kinematics:")
    print(f"  Position (should be [0,0,0]): {kinematics.eyeball.position_xyz[0]}")
    print(f"  Orientation at t=0: {kinematics.quaternions_wxyz[0]}")
    print(f"  Gaze at t=0: {kinematics.gaze_directions[0]}")

    # Check angular velocity
    print(f"\nAngular velocity (computed automatically):")
    omega_global = kinematics.angular_velocity_global
    print(f"  Shape: {omega_global.shape}")
    print(f"  Mean |ω|: {np.mean(np.linalg.norm(omega_global, axis=1)):.4f} rad/s")
    print(f"  Max |ω|: {np.max(np.linalg.norm(omega_global, axis=1)):.4f} rad/s")

    # Check socket landmarks
    print(f"\nSocket landmarks (don't rotate with eye):")
    td_mean = np.mean(kinematics.tear_duct_mm, axis=0)
    oe_mean = np.mean(kinematics.outer_eye_mm, axis=0)
    print(f"  Mean tear_duct: [{td_mean[0]:.2f}, {td_mean[1]:.2f}, {td_mean[2]:.2f}]")
    print(f"  Mean outer_eye: [{oe_mean[0]:.2f}, {oe_mean[1]:.2f}, {oe_mean[2]:.2f}]")

    # Check pupil trajectory (computed from orientation)
    print(f"\nPupil center trajectory (rotates with eye):")
    pupil_traj = kinematics.pupil_center_trajectory
    print(f"  At t=0 (rest): [{pupil_traj[0, 0]:.3f}, {pupil_traj[0, 1]:.3f}, {pupil_traj[0, 2]:.3f}]")
    print(f"  At t=0.25 (max right): [{pupil_traj[25, 0]:.3f}, {pupil_traj[25, 1]:.3f}, {pupil_traj[25, 2]:.3f}]")
    print(f"  At t=0.5 (back to center): [{pupil_traj[50, 0]:.3f}, {pupil_traj[50, 1]:.3f}, {pupil_traj[50, 2]:.3f}]")

    # Verify that pupil moves but landmarks don't (much)
    print(f"\nVerifying architectural separation:")
    pupil_range = np.ptp(pupil_traj, axis=0)  # Range of motion
    landmark_range = np.ptp(kinematics.tear_duct_mm, axis=0)  # Should be small (just noise)
    print(f"  Pupil center range of motion: [{pupil_range[0]:.3f}, {pupil_range[1]:.3f}, {pupil_range[2]:.3f}]")
    print(f"  Tear duct range (noise only): [{landmark_range[0]:.3f}, {landmark_range[1]:.3f}, {landmark_range[2]:.3f}]")

    # Show available derived quantities
    print(f"\nDerived quantities available:")
    print(f"  Azimuth range: {np.min(kinematics.azimuth_degrees):.1f}° to {np.max(kinematics.azimuth_degrees):.1f}°")
    print(f"  Elevation range: {np.min(kinematics.elevation_degrees):.1f}° to {np.max(kinematics.elevation_degrees):.1f}°")
    print(f"  Roll (should be ~0): mean={np.mean(kinematics.roll.values):.4f}°")
    print(f"  Angular speed: mean={np.mean(kinematics.angular_speed.values):.3f} rad/s")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    return kinematics


if __name__ == "__main__":
    # Run the example to generate kinematics
    kinematics = run_eye_kinematics_example()

    # Create and show the animated visualization
    print("\nCreating animated visualization...")
    fig = create_animated_eye_figure(kinematics, frame_step=2)
    fig.show()