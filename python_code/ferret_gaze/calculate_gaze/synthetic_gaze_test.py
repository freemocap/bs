"""
Synthetic Gaze Pipeline Validation
====================================

Generates synthetic eye and skull kinematics with analytically known outputs,
runs them through the actual calculate_ferret_gaze() pipeline, loads the output,
and prints comparison metrics.

Three scenarios:
  1. eye_only     — Eye with adduction+elevation, skull at identity
  2. skull_yaw_only — Skull with yaw, eye at identity
  3. combined     — Both active simultaneously

PASS condition: max error < 1° for horizontal and elevation in all scenarios.

Usage:
    python python_code/ferret_gaze/calculate_gaze/synthetic_gaze_test.py
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from python_code.ferret_gaze.calculate_gaze.calculate_ferret_gaze import (
    calculate_ferret_gaze,
    batch_rotate_vector_by_quaternion,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_models import FerretEyeKinematics
from python_code.kinematics_core.reference_geometry_model import (
    ReferenceGeometry,
    MarkerPosition,
    CoordinateFrameDefinition,
    AxisDefinition,
    AxisType,
)
from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics

# ============================================================================
# PATHS
# ============================================================================

SYNTHETIC_DATA_DIR = Path(__file__).parent / "synthetic_test_data"

# ============================================================================
# TIMESTAMPS
# ============================================================================

TIMESTAMPS = np.linspace(0, 5, 1000, endpoint=False)

# ============================================================================
# SYNTHETIC QUATERNION GENERATORS
# ============================================================================


def make_eye_quaternions(timestamps: np.ndarray) -> np.ndarray:
    """
    Eye adduction+elevation quaternions.

    Adduction: 10 Hz sine, ±20°, rotation around eye +Y axis.
    Elevation:  1 Hz sine, ±10°, rotation around eye +X axis.
    Intrinsic 'YX' order: first Y (adduction), then X (elevation).

    Returns:
        (N, 4) quaternions in [w, x, y, z] order.
    """
    adduction_deg = 20.0 * np.sin(2.0 * np.pi * 10.0 * timestamps)
    elevation_deg = 10.0 * np.sin(2.0 * np.pi * 1.0 * timestamps)
    r = Rotation.from_euler("YX", np.column_stack([adduction_deg, elevation_deg]), degrees=True)
    xyzw = r.as_quat()
    return xyzw[:, [3, 0, 1, 2]]  # xyzw → wxyz


def make_skull_quaternions(timestamps: np.ndarray) -> np.ndarray:
    """
    Skull yaw quaternions.

    Yaw: 0.3 Hz sine, ±15°, rotation around world +Z axis.

    Returns:
        (N, 4) quaternions in [w, x, y, z] order.
    """
    yaw_deg = 15.0 * np.sin(2.0 * np.pi * 0.3 * timestamps)
    r = Rotation.from_euler("Z", np.column_stack([yaw_deg]), degrees=True)
    xyzw = r.as_quat()
    return xyzw[:, [3, 0, 1, 2]]  # xyzw → wxyz


def make_identity_quaternions(n_frames: int) -> np.ndarray:
    """Return (N, 4) identity quaternions [1, 0, 0, 0]."""
    q = np.zeros((n_frames, 4), dtype=np.float64)
    q[:, 0] = 1.0
    return q


# ============================================================================
# REFERENCE GEOMETRY
# ============================================================================


def make_skull_reference_geometry() -> ReferenceGeometry:
    """
    Skull reference geometry with eyes and nose.

    Skull frame: +X toward nose, +Y toward left_eye, +Z superior (computed).
    Origin at midpoint between eyes.
    """
    return ReferenceGeometry(
        units="mm",
        coordinate_frame=CoordinateFrameDefinition(
            origin_keypoints=["left_eye", "right_eye"],
            x_axis=AxisDefinition(keypoints=["nose"], type=AxisType.EXACT),
            y_axis=AxisDefinition(keypoints=["left_eye"], type=AxisType.APPROXIMATE),
        ),
        keypoints={
            "nose": MarkerPosition(x=60.0, y=0.0, z=0.0),
            "left_eye": MarkerPosition(x=0.0, y=50.0, z=0.0),
            "right_eye": MarkerPosition(x=0.0, y=-50.0, z=0.0),
        },
        display_edges=[
            ("left_eye", "right_eye"),
            ("nose", "left_eye"),
            ("nose", "right_eye"),
        ],
    )


# ============================================================================
# SCENARIO SETUP
# ============================================================================


def make_eye_kinematics(
    eye_name: str,
    timestamps: np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> FerretEyeKinematics:
    """Create a FerretEyeKinematics with the given quaternions and zero pose data."""
    n = len(timestamps)
    return FerretEyeKinematics.from_pose_data(
        eye_name=eye_name,
        eye_data_csv_path="synthetic",
        timestamps=timestamps,
        quaternions_wxyz=quaternions_wxyz,
        pupil_center_mm=np.zeros((n, 3), dtype=np.float64),
        pupil_points_mm=np.zeros((n, 8, 3), dtype=np.float64),
        tear_duct_mm=np.zeros((n, 3), dtype=np.float64),
        outer_eye_mm=np.zeros((n, 3), dtype=np.float64),
        rest_gaze_direction_camera=np.array([0.0, 0.0, 1.0]),
        camera_to_eye_rotation=np.eye(3),
    )


def make_skull_kinematics(
    timestamps: np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> RigidBodyKinematics:
    """Create a skull RigidBodyKinematics with the given quaternions and zero position."""
    n = len(timestamps)
    return RigidBodyKinematics.from_pose_arrays(
        name="skull",
        reference_geometry=make_skull_reference_geometry(),
        timestamps=timestamps,
        position_xyz=np.zeros((n, 3), dtype=np.float64),
        quaternions_wxyz=quaternions_wxyz,
    )


def save_scenario(
    scenario_dir: Path,
    skull: RigidBodyKinematics,
    left_eye: FerretEyeKinematics,
    right_eye: FerretEyeKinematics,
) -> None:
    """Save skull and eye kinematics for a scenario to disk."""
    skull.save_to_disk(scenario_dir / "skull_kinematics")
    left_eye.save_to_disk(scenario_dir / "left_eye_kinematics")
    right_eye.save_to_disk(scenario_dir / "right_eye_kinematics")


# ============================================================================
# COMPARISON METRICS
# ============================================================================


def compute_gaze_angles(gaze_kinematics: RigidBodyKinematics) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract horizontal and elevation gaze angles from output kinematics.

    Uses gaze direction = batch_rotate_vector_by_quaternion(q, [0,0,1]).

    Horizontal: atan2(x, -y)   — for lateral eye whose rest gaze is -Y in world
    Elevation:  atan2(z, sqrt(x²+y²))
    """
    gaze_quats = gaze_kinematics.quaternions_wxyz
    gaze_dir = batch_rotate_vector_by_quaternion(gaze_quats, np.array([0.0, 0.0, 1.0]))
    gaze_x, gaze_y, gaze_z = gaze_dir[:, 0], gaze_dir[:, 1], gaze_dir[:, 2]
    horizontal = np.degrees(np.arctan2(gaze_x, -gaze_y))
    elevation = np.degrees(np.arctan2(gaze_z, np.sqrt(gaze_x**2 + gaze_y**2)))
    return horizontal, elevation


def print_scenario_results(
    scenario_name: str,
    expected_horizontal: np.ndarray,
    expected_elevation: np.ndarray,
    actual_horizontal: np.ndarray,
    actual_elevation: np.ndarray,
    threshold_deg: float = 1.0,
) -> bool:
    """Print comparison metrics and return True if all checks pass."""
    h_error = np.abs(actual_horizontal - expected_horizontal)
    e_error = np.abs(actual_elevation - expected_elevation)

    h_max_error = np.max(h_error)
    e_max_error = np.max(e_error)

    h_corr = float(np.corrcoef(expected_horizontal, actual_horizontal)[0, 1])
    e_corr = float(np.corrcoef(expected_elevation, actual_elevation)[0, 1])

    h_pass = h_max_error < threshold_deg
    e_pass = e_max_error < threshold_deg

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")
    print(f"  Horizontal gaze:")
    print(f"    Expected peak:  {np.max(np.abs(expected_horizontal)):.2f}°")
    print(f"    Actual peak:    {np.max(np.abs(actual_horizontal)):.2f}°")
    print(f"    Max error:      {h_max_error:.4f}°  {'PASS' if h_pass else 'FAIL'}")
    print(f"    Correlation:    {h_corr:.6f}")
    print(f"  Elevation gaze:")
    print(f"    Expected peak:  {np.max(np.abs(expected_elevation)):.2f}°")
    print(f"    Actual peak:    {np.max(np.abs(actual_elevation)):.2f}°")
    print(f"    Max error:      {e_max_error:.4f}°  {'PASS' if e_pass else 'FAIL'}")
    print(f"    Correlation:    {e_corr:.6f}")

    return h_pass and e_pass


# ============================================================================
# MAIN
# ============================================================================


def run_synthetic_tests() -> None:
    """Generate all scenarios, run pipeline, and report comparison metrics."""
    timestamps = TIMESTAMPS
    n = len(timestamps)

    eye_quats = make_eye_quaternions(timestamps)
    skull_quats = make_skull_quaternions(timestamps)
    identity_quats = make_identity_quaternions(n)

    # Expected signal components
    adduction_deg = 20.0 * np.sin(2.0 * np.pi * 10.0 * timestamps)
    elevation_deg = 10.0 * np.sin(2.0 * np.pi * 1.0 * timestamps)
    skull_yaw_deg = 15.0 * np.sin(2.0 * np.pi * 0.3 * timestamps)

    scenarios: list[tuple[str, RigidBodyKinematics, FerretEyeKinematics, FerretEyeKinematics]] = [
        (
            "eye_only",
            make_skull_kinematics(timestamps, identity_quats),
            make_eye_kinematics("left_eye", timestamps, eye_quats),
            make_eye_kinematics("right_eye", timestamps, eye_quats),
        ),
        (
            "skull_yaw_only",
            make_skull_kinematics(timestamps, skull_quats),
            make_eye_kinematics("left_eye", timestamps, identity_quats),
            make_eye_kinematics("right_eye", timestamps, identity_quats),
        ),
        (
            "combined",
            make_skull_kinematics(timestamps, skull_quats),
            make_eye_kinematics("left_eye", timestamps, eye_quats),
            make_eye_kinematics("right_eye", timestamps, eye_quats),
        ),
    ]

    all_passed = True

    for scenario_name, skull, left_eye, right_eye in scenarios:
        recording_dir = SYNTHETIC_DATA_DIR / scenario_name / "full_recording"
        analyzable_dir = recording_dir / "analyzable_output"
        output_dir = analyzable_dir / "gaze_kinematics"

        # Create empty sibling directories to match expected folder structure
        (recording_dir / "eye_data").mkdir(parents=True, exist_ok=True)
        (recording_dir / "mocap_data").mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating scenario: {scenario_name}")
        save_scenario(analyzable_dir, skull, left_eye, right_eye)

        calculate_ferret_gaze(
            resampled_data_dir=analyzable_dir,
            output_dir=output_dir,
        )

        # Load right gaze output
        right_gaze = RigidBodyKinematics.load_from_disk(
            kinematics_csv_path=output_dir / "right_gaze_kinematics.csv",
            reference_geometry_json_path=output_dir / "right_gaze_reference_geometry.json",
        )

        actual_h, actual_e = compute_gaze_angles(right_gaze)

        # Expected values (right eye, analytically derived)
        if scenario_name == "eye_only":
            expected_h = adduction_deg
            expected_e = elevation_deg
        elif scenario_name == "skull_yaw_only":
            expected_h = skull_yaw_deg
            expected_e = np.zeros_like(timestamps)
        else:  # combined
            expected_h = skull_yaw_deg + adduction_deg
            expected_e = elevation_deg

        passed = print_scenario_results(
            scenario_name=scenario_name,
            expected_horizontal=expected_h,
            expected_elevation=expected_e,
            actual_horizontal=actual_h,
            actual_elevation=actual_e,
        )
        all_passed = all_passed and passed

    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_synthetic_tests()
