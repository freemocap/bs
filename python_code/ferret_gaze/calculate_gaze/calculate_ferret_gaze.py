"""
Calculate Ferret Gaze
=====================

Loads resampled skull and eye kinematics, transforms eye data to skull-mounted
world coordinates to compute gaze-in-world trajectories.

Coordinate Systems:
    Skull frame: Origin at eye midpoint, +X toward nose, +Y toward left_eye, +Z superior
    Eye frame (at rest): +Z = gaze (outward), +Y = superior, +X = subject's left

Socket mounting (eye rest frame in skull coordinates):
    Right eye: +X = skull +X, +Y = skull +Z, +Z = -skull Y (outward from right eye)
    Left eye: +X = -skull +X, +Y = skull +Z, +Z = +skull Y (outward from left eye)

The script transforms eye-in-head rotations to gaze-in-world by:
1. Mounting the eye in a socket frame aligned with the skull
2. Applying the skull's world transform to get world coordinates
"""
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.kinematics_core.reference_geometry_model import (
    ReferenceGeometry,
    MarkerPosition,
    CoordinateFrameDefinition,
    AxisDefinition,
    AxisType,
)
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def quaternion_from_rotation_matrix(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion as [w, x, y, z]
    """
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

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def batch_quaternion_multiply(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Multiply quaternions q1 * q2 (Hamilton product), batched.

    Args:
        q1: (N, 4) quaternions [w, x, y, z]
        q2: (N, 4) or (4,) quaternions [w, x, y, z]

    Returns:
        (N, 4) result quaternions
    """
    if q2.ndim == 1:
        q2 = np.broadcast_to(q2, q1.shape)

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def batch_rotate_vector_by_quaternion(
    q: NDArray[np.float64],
    v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Rotate vector(s) by quaternion(s).

    Args:
        q: (N, 4) quaternions [w, x, y, z]
        v: (3,) or (N, 3) vectors

    Returns:
        (N, 3) rotated vectors
    """
    if v.ndim == 1:
        v = np.broadcast_to(v, (len(q), 3))

    w = q[:, 0]
    u = q[:, 1:4]

    uv = np.cross(u, v)
    uuv = np.cross(u, uv)

    return v + 2.0 * w[:, np.newaxis] * uv + 2.0 * uuv


def get_eye_to_skull_rotation_matrix(eye_side: Literal["left", "right"]) -> NDArray[np.float64]:
    """
    Get rotation matrix from eye local frame to skull local frame.

    Eye frame (at rest): +Z = gaze, +Y = up, +X = subject's left
    Skull frame: +X = nose, +Y = toward left_eye, +Z = superior

    At rest, the eye should point outward from the skull:
    - Right eye (at skull -Y): gaze = -skull Y
    - Left eye (at skull +Y): gaze = +skull Y

    Args:
        eye_side: "left" or "right"

    Returns:
        3x3 rotation matrix R such that v_skull = R @ v_eye
    """
    if eye_side == "right":
        # Right eye at skull position (0, -d, 0)
        # Eye +X (subject's left) = Skull +X (toward nose, which is left when looking from right eye outward)
        # Eye +Y (up) = Skull +Z
        # Eye +Z (gaze) = -Skull Y (outward from right eye)
        eye_x_in_skull = np.array([1.0, 0.0, 0.0])
        eye_y_in_skull = np.array([0.0, 0.0, 1.0])
        eye_z_in_skull = np.array([0.0, -1.0, 0.0])
    else:
        # Left eye at skull position (0, +d, 0)
        # Eye +X (subject's left) = -Skull X (away from nose, which is left when looking from left eye outward)
        # Eye +Y (up) = Skull +Z
        # Eye +Z (gaze) = +Skull Y (outward from left eye)
        eye_x_in_skull = np.array([-1.0, 0.0, 0.0])
        eye_y_in_skull = np.array([0.0, 0.0, 1.0])
        eye_z_in_skull = np.array([0.0, 1.0, 0.0])

    # Columns of rotation matrix are eye basis vectors in skull coordinates
    R = np.column_stack([eye_x_in_skull, eye_y_in_skull, eye_z_in_skull])
    return R


def compute_gaze_kinematics(
    eye_kinematics: RigidBodyKinematics,
    skull_kinematics: RigidBodyKinematics,
    eye_position_in_skull: NDArray[np.float64],
    eye_side: Literal["left", "right"],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute gaze kinematics in world coordinates.

    Transforms eye orientation from eye-local frame to world frame by:
    1. Applying eye-to-skull rotation (socket mounting)
    2. Applying skull world rotation
    3. Computing world position of eye center

    Args:
        eye_kinematics: Eye RigidBodyKinematics (orientation only, position at origin)
        skull_kinematics: Skull RigidBodyKinematics in world coordinates
        eye_position_in_skull: (3,) eye center position in skull local coords
        eye_side: "left" or "right"

    Returns:
        Tuple of (gaze_timestamps, gaze_position_xyz, gaze_quaternions_wxyz)
    """
    n_frames = eye_kinematics.n_frames

    if not np.allclose(eye_kinematics.timestamps, skull_kinematics.timestamps, rtol=1e-9):
        raise ValueError("Eye and skull timestamps must match")

    # Get eye-to-skull rotation matrix and its quaternion
    R_eye_to_skull = get_eye_to_skull_rotation_matrix(eye_side)
    q_eye_to_skull = quaternion_from_rotation_matrix(R_eye_to_skull)

    logger.info(f"  Eye-to-skull rotation quaternion: {q_eye_to_skull}")

    # Compute world position of eye center for each frame
    # world_pos = skull_pos + q_skull.rotate(eye_pos_in_skull)
    gaze_position_xyz = skull_kinematics.position_xyz + batch_rotate_vector_by_quaternion(
        skull_kinematics.quaternions_wxyz, eye_position_in_skull
    )

    # Compute world orientation of eye
    # q_gaze_world = q_skull * q_eye_to_skull * q_eye
    # First: q_eye_to_skull * q_eye (socket-mounted eye orientation in skull frame)
    q_eye_in_skull = batch_quaternion_multiply(
        np.broadcast_to(q_eye_to_skull, (n_frames, 4)),
        eye_kinematics.quaternions_wxyz,
    )

    # Then: q_skull * q_eye_in_skull (world orientation)
    gaze_quaternions_wxyz = batch_quaternion_multiply(
        skull_kinematics.quaternions_wxyz,
        q_eye_in_skull,
    )

    # Normalize quaternions
    norms = np.linalg.norm(gaze_quaternions_wxyz, axis=1, keepdims=True)
    gaze_quaternions_wxyz = gaze_quaternions_wxyz / np.maximum(norms, 1e-10)

    return eye_kinematics.timestamps.copy(), gaze_position_xyz, gaze_quaternions_wxyz


GAZE_TARGET_DISTANCE_MM: float = 100.0


def create_gaze_reference_geometry(eye_radius_mm: float = 3.5) -> ReferenceGeometry:
    """
    Create a reference geometry for gaze kinematics.

    This is similar to the eyeball reference geometry but includes a
    gaze_target keypoint at 100mm along +Z (the gaze direction).

    Args:
        eye_radius_mm: Radius of eyeball in mm

    Returns:
        ReferenceGeometry for gaze visualization
    """
    keypoints: dict[str, MarkerPosition] = {
        # Origin at eye center
        "eyeball_center": MarkerPosition(x=0.0, y=0.0, z=0.0),
        # Pupil center at eye surface
        "pupil_center": MarkerPosition(x=0.0, y=0.0, z=eye_radius_mm),
        # Gaze target at 100mm along gaze direction
        "gaze_target": MarkerPosition(x=0.0, y=0.0, z=GAZE_TARGET_DISTANCE_MM),
    }

    # Coordinate frame: +Z = gaze, +Y = up
    coordinate_frame = CoordinateFrameDefinition(
        origin_keypoints=["eyeball_center"],
        z_axis=AxisDefinition(keypoints=["gaze_target"], type=AxisType.EXACT),
        y_axis=AxisDefinition(keypoints=["pupil_center"], type=AxisType.APPROXIMATE),
    )

    return ReferenceGeometry(
        units="mm",
        coordinate_frame=coordinate_frame,
        keypoints=keypoints,
        display_edges=[("eyeball_center", "gaze_target")],
        rigid_edges=[("eyeball_center", "pupil_center")],
    )


def create_gaze_kinematics(
    name: str,
    timestamps: NDArray[np.float64],
    position_xyz: NDArray[np.float64],
    quaternions_wxyz: NDArray[np.float64],
    eyeball_reference_geometry: ReferenceGeometry,
) -> RigidBodyKinematics:
    """
    Create a RigidBodyKinematics object for gaze data.

    Creates a gaze-specific reference geometry that includes a gaze_target
    keypoint at 100mm along +Z. This allows the visualization to compute
    the gaze target position in world coordinates.

    Args:
        name: Name for the gaze kinematics (e.g., "left_gaze" or "right_gaze")
        timestamps: (N,) timestamps
        position_xyz: (N, 3) world positions of eye center
        quaternions_wxyz: (N, 4) world orientations
        eyeball_reference_geometry: Reference geometry from the eyeball (used for eye_radius)

    Returns:
        RigidBodyKinematics object representing gaze in world coordinates
    """
    # Extract eye radius from eyeball geometry (from pupil_center z coordinate)
    pupil_center_pos = eyeball_reference_geometry.get_keypoint_position("pupil_center")
    eye_radius_mm = float(pupil_center_pos[2])  # z coordinate is the eye radius

    # Create gaze-specific reference geometry with gaze_target at 100mm
    gaze_geometry = create_gaze_reference_geometry(eye_radius_mm=eye_radius_mm)

    return RigidBodyKinematics.from_pose_arrays(
        name=name,
        reference_geometry=gaze_geometry,
        timestamps=timestamps,
        position_xyz=position_xyz,
        quaternions_wxyz=quaternions_wxyz,
    )


def calculate_ferret_gaze(
    resampled_data_dir: Path,
    output_dir: Path,
) -> None:
    """
    Main function to calculate ferret gaze kinematics.

    Loads resampled skull and eye kinematics, transforms eye data to
    skull-mounted world coordinates, and saves results.

    Args:
        resampled_data_dir: Directory containing resampled data:
            - skull_kinematics/skull_kinematics.csv
            - skull_kinematics/skull_reference_geometry.json
            - left_eye_kinematics/ (FerretEyeKinematics directory)
            - right_eye_kinematics/ (FerretEyeKinematics directory)
        output_dir: Directory to save output files
    """
    logger.info("=" * 80)
    logger.info("CALCULATING FERRET GAZE KINEMATICS")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD SKULL KINEMATICS
    # =========================================================================
    logger.info("\nLoading skull kinematics...")

    skull_kinematics_csv = resampled_data_dir / "skull_kinematics" / "skull_kinematics.csv"
    skull_reference_geometry_json = resampled_data_dir / "skull_kinematics" / "skull_reference_geometry.json"

    skull_kinematics = RigidBodyKinematics.load_from_disk(
        kinematics_csv_path=skull_kinematics_csv,
        reference_geometry_json_path=skull_reference_geometry_json,
    )

    skull_reference_geometry = ReferenceGeometry.from_json_file(skull_reference_geometry_json)

    logger.info(f"  Frames: {skull_kinematics.n_frames}")
    logger.info(f"  Time range: {skull_kinematics.timestamps[0]:.4f}s to {skull_kinematics.timestamps[-1]:.4f}s")

    # Load eye positions in skull reference frame
    left_eye_position_in_skull = skull_reference_geometry.get_keypoint_position("left_eye")
    right_eye_position_in_skull = skull_reference_geometry.get_keypoint_position("right_eye")

    logger.info(f"  Left eye position in skull: {left_eye_position_in_skull}")
    logger.info(f"  Right eye position in skull: {right_eye_position_in_skull}")

    # =========================================================================
    # LOAD EYE KINEMATICS
    # =========================================================================
    logger.info("\nLoading eye kinematics...")

    left_eye_kinematics = FerretEyeKinematics.load_from_directory(
        eye_name="left_eye",
        input_directory=resampled_data_dir / "left_eye_kinematics",
    )
    logger.info(f"  Left eye frames: {left_eye_kinematics.n_frames}")

    right_eye_kinematics = FerretEyeKinematics.load_from_directory(
        eye_name="right_eye",
        input_directory=resampled_data_dir / "right_eye_kinematics",
    )
    logger.info(f"  Right eye frames: {right_eye_kinematics.n_frames}")

    # Verify frame counts match
    if skull_kinematics.n_frames != left_eye_kinematics.n_frames:
        raise ValueError(
            f"Frame count mismatch: skull ({skull_kinematics.n_frames}) != "
            f"left eye ({left_eye_kinematics.n_frames})"
        )
    if skull_kinematics.n_frames != right_eye_kinematics.n_frames:
        raise ValueError(
            f"Frame count mismatch: skull ({skull_kinematics.n_frames}) != "
            f"right eye ({right_eye_kinematics.n_frames})"
        )

    # =========================================================================
    # COMPUTE GAZE KINEMATICS
    # =========================================================================
    logger.info("\nComputing gaze kinematics...")

    logger.info("  Processing left eye...")

    # DEBUG: Check if eye quaternions have variation (not all identity)
    eye_quats = left_eye_kinematics.eyeball.quaternions_wxyz
    quat_std = np.std(eye_quats, axis=0)
    logger.info(f"    Left eye quaternion std: w={quat_std[0]:.6f}, x={quat_std[1]:.6f}, y={quat_std[2]:.6f}, z={quat_std[3]:.6f}")
    if np.allclose(quat_std, 0.0, atol=1e-6):
        logger.warning("    WARNING: Left eye quaternions appear constant (no eye movement)!")

    left_gaze_timestamps, left_gaze_position_xyz, left_gaze_quaternions_wxyz = compute_gaze_kinematics(
        eye_kinematics=left_eye_kinematics.eyeball,
        skull_kinematics=skull_kinematics,
        eye_position_in_skull=left_eye_position_in_skull,
        eye_side="left",
    )

    logger.info("  Processing right eye...")

    # DEBUG: Check if eye quaternions have variation
    eye_quats = right_eye_kinematics.eyeball.quaternions_wxyz
    quat_std = np.std(eye_quats, axis=0)
    logger.info(f"    Right eye quaternion std: w={quat_std[0]:.6f}, x={quat_std[1]:.6f}, y={quat_std[2]:.6f}, z={quat_std[3]:.6f}")
    if np.allclose(quat_std, 0.0, atol=1e-6):
        logger.warning("    WARNING: Right eye quaternions appear constant (no eye movement)!")

    right_gaze_timestamps, right_gaze_position_xyz, right_gaze_quaternions_wxyz = compute_gaze_kinematics(
        eye_kinematics=right_eye_kinematics.eyeball,
        skull_kinematics=skull_kinematics,
        eye_position_in_skull=right_eye_position_in_skull,
        eye_side="right",
    )

    # DEBUG: Verify gaze quaternions include eye rotation (not just skull rotation)
    # Compute "relative gaze" = gaze * conj(skull) - this should show eye movement
    skull_q = skull_kinematics.quaternions_wxyz
    skull_q_conj = skull_q * np.array([1, -1, -1, -1])

    def _quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.column_stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    left_relative = _quat_mult(left_gaze_quaternions_wxyz, skull_q_conj)
    left_rel_std = np.std(left_relative, axis=0)
    logger.info(f"  Left gaze relative to skull - quaternion std: {left_rel_std}")

    right_relative = _quat_mult(right_gaze_quaternions_wxyz, skull_q_conj)
    right_rel_std = np.std(right_relative, axis=0)
    logger.info(f"  Right gaze relative to skull - quaternion std: {right_rel_std}")

    if np.allclose(left_rel_std[1:], 0.0, atol=1e-5) and np.allclose(right_rel_std[1:], 0.0, atol=1e-5):
        logger.error("  ERROR: Gaze quaternions appear to have NO eye rotation relative to skull!")
        logger.error("  The gaze should vary relative to skull if eye tracking captured any movement.")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    logger.info("\nSaving gaze kinematics...")

    # Create gaze kinematics objects using eyeball reference geometry
    left_gaze_kinematics = create_gaze_kinematics(
        name="left_gaze",
        timestamps=left_gaze_timestamps,
        position_xyz=left_gaze_position_xyz,
        quaternions_wxyz=left_gaze_quaternions_wxyz,
        eyeball_reference_geometry=left_eye_kinematics.eyeball.reference_geometry,
    )

    right_gaze_kinematics = create_gaze_kinematics(
        name="right_gaze",
        timestamps=right_gaze_timestamps,
        position_xyz=right_gaze_position_xyz,
        quaternions_wxyz=right_gaze_quaternions_wxyz,
        eyeball_reference_geometry=right_eye_kinematics.eyeball.reference_geometry,
    )

    # Save using the standard RigidBodyKinematics serialization
    left_gaze_kinematics.save_to_disk(output_directory=output_dir)
    logger.info(f"  Saved: left_gaze_kinematics.csv and left_gaze_reference_geometry.json")

    right_gaze_kinematics.save_to_disk(output_directory=output_dir)
    logger.info(f"  Saved: right_gaze_kinematics.csv and right_gaze_reference_geometry.json")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GAZE CALCULATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Frames: {left_gaze_kinematics.n_frames}")
    logger.info(f"Time range: {left_gaze_timestamps[0]:.4f}s to {left_gaze_timestamps[-1]:.4f}s")
    logger.info(f"Framerate: {left_gaze_kinematics.framerate_hz:.2f} Hz")
    logger.info("\nOutput files (same format as skull/eye kinematics):")
    logger.info("  - left_gaze_kinematics.csv")
    logger.info("  - left_gaze_reference_geometry.json")
    logger.info("  - right_gaze_kinematics.csv")
    logger.info("  - right_gaze_reference_geometry.json")
    logger.info("\nGaze kinematics represent the eyeball in world coordinates:")
    logger.info("  - Position: world position of eye center (mm)")
    logger.info("  - Orientation: world orientation quaternion")
    logger.info("  - +Z axis of oriented eyeball = gaze direction in world")


if __name__ == "__main__":
    # Example usage - modify paths as needed
    _resampled_data_dir = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\analyzable_output"
    )
    _output_dir = _resampled_data_dir / "gaze_kinematics"

    calculate_ferret_gaze(
        resampled_data_dir=_resampled_data_dir,
        output_dir=_output_dir,
    )