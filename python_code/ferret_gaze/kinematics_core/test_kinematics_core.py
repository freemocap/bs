"""
Comprehensive tests for kinematics_core module.

Tests both the dataclass and validates the pydantic version structure.
Run with: python test_kinematics_core.py
"""
import json as json_module

import numpy as np
from numpy.typing import NDArray

from python_code.ferret_gaze.kinematics_core.timeseries_model import Timeseries
from python_code.ferret_gaze.kinematics_core.vector3_trajectory_model import Vector3Trajectory
from python_code.ferret_gaze.kinematics_core.quaternion_trajectory_model import QuaternionTrajectory
from python_code.ferret_gaze.kinematics_core.angular_velocity_trajectory_model import AngularVelocityTrajectory
from python_code.ferret_gaze.kinematics_core.rigid_body_state_model import RigidBodyState
from python_code.ferret_gaze.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from quaternion_model import Quaternion, resample_quaternions
from reference_geometry_model import ReferenceGeometry, EXAMPLE_JSON


# =============================================================================
# TEST FIXTURES
# =============================================================================


def create_test_geometry() -> ReferenceGeometry:
    """Create a simple test reference geometry."""
    return ReferenceGeometry(**{
        "units": "mm",
        "coordinate_frame": {
            "origin_markers": ["left_eye", "right_eye"],
            "x_axis": {"markers": ["nose"], "type": "exact"},
            "y_axis": {"markers": ["left_eye"], "type": "approximate"},
        },
        "markers": {
            "nose": {"x": 18.0, "y": 0.0, "z": 0.0},
            "left_eye": {"x": 0.0, "y": 12.0, "z": 0.0},
            "right_eye": {"x": 0.0, "y": -12.0, "z": 0.0},
            "back": {"x": -20.0, "y": 0.0, "z": 5.0},
        },
    })


def create_test_kinematics(n_frames: int = 10) -> RigidBodyKinematics:
    """Create test kinematics with circular motion."""
    geom = create_test_geometry()

    timestamps = np.linspace(0.0, 1.0, n_frames)

    # Circular motion in XY plane
    angles = np.linspace(0, np.pi / 2, n_frames)
    position_xyz = np.column_stack([
        100 * np.cos(angles),
        100 * np.sin(angles),
        np.zeros(n_frames),
    ])

    # Compute velocity via gradient
    velocity_xyz = np.gradient(position_xyz, timestamps, axis=0)

    # Orientation: rotating around Z to face direction of motion
    orientations = [
        Quaternion(w=np.cos(a / 2), x=0, y=0, z=np.sin(a / 2))
        for a in angles
    ]

    # Angular velocity
    angular_velocity_global = np.zeros((n_frames, 3))
    angular_velocity_global[:, 2] = np.gradient(angles, timestamps)
    angular_velocity_local = angular_velocity_global.copy()

    return RigidBodyKinematics(
        reference_geometry=geom,
        timestamps=timestamps,
        position_xyz=position_xyz,
        velocity_xyz=velocity_xyz,
        orientations=orientations,
        angular_velocity_global=angular_velocity_global,
        angular_velocity_local=angular_velocity_local,
    )


def create_stationary_kinematics(n_frames: int = 5) -> RigidBodyKinematics:
    """Create kinematics with no motion (for baseline tests)."""
    geom = create_test_geometry()
    timestamps = np.linspace(0.0, 1.0, n_frames)

    return RigidBodyKinematics(
        reference_geometry=geom,
        timestamps=timestamps,
        position_xyz=np.zeros((n_frames, 3)),
        velocity_xyz=np.zeros((n_frames, 3)),
        orientations=[Quaternion.identity() for _ in range(n_frames)],
        angular_velocity_global=np.zeros((n_frames, 3)),
        angular_velocity_local=np.zeros((n_frames, 3)),
    )


# =============================================================================
# QUATERNION HELPER TESTS
# =============================================================================


def test_quaternion_identity() -> None:
    """Test identity quaternion."""
    print("\n=== Test: Identity Quaternion ===")
    q_id = Quaternion.identity()
    assert q_id.w == 1.0 and q_id.x == 0.0 and q_id.y == 0.0 and q_id.z == 0.0
    v = np.array([1.0, 2.0, 3.0])
    v_rot = q_id.rotate_vector(v)
    assert np.allclose(v, v_rot), f"Identity should not change vector: {v} -> {v_rot}"
    print("  ✓ Identity quaternion works correctly")


def test_quaternion_90_degree_z_rotation() -> None:
    """Test 90° rotation around Z."""
    print("\n=== Test: 90° Z-rotation ===")
    angle = np.pi / 2
    q_z90 = Quaternion(w=np.cos(angle / 2), x=0, y=0, z=np.sin(angle / 2))
    v_x = np.array([1.0, 0.0, 0.0])
    v_y_expected = np.array([0.0, 1.0, 0.0])
    v_y_result = q_z90.rotate_vector(v_x)
    assert np.allclose(v_y_result, v_y_expected, atol=1e-10), f"Expected {v_y_expected}, got {v_y_result}"
    print(f"  ✓ Rotating [1,0,0] by 90° around Z gives {v_y_result}")


def test_quaternion_multiplication() -> None:
    """Test quaternion multiplication (composition)."""
    print("\n=== Test: Quaternion Multiplication ===")
    q_z45 = Quaternion(w=np.cos(np.pi / 8), x=0, y=0, z=np.sin(np.pi / 8))  # 45°
    q_composed = q_z45 * q_z45  # Should be 90°
    v_x = np.array([1.0, 0.0, 0.0])
    v_y_expected = np.array([0.0, 1.0, 0.0])
    v_result = q_composed.rotate_vector(v_x)
    assert np.allclose(v_result, v_y_expected, atol=1e-10), f"45° + 45° should equal 90°"
    print("  ✓ Two 45° rotations compose to 90°")


def test_quaternion_inverse() -> None:
    """Test quaternion inverse."""
    print("\n=== Test: Quaternion Inverse ===")
    angle = np.pi / 2
    q_z90 = Quaternion(w=np.cos(angle / 2), x=0, y=0, z=np.sin(angle / 2))
    q_inv = q_z90.inverse()
    q_product = q_z90 * q_inv
    assert np.allclose([q_product.w, q_product.x, q_product.y, q_product.z], [1, 0, 0, 0], atol=1e-10)
    print("  ✓ q * q^-1 = identity")


def test_quaternion_rotation_matrix_roundtrip() -> None:
    """Test rotation matrix roundtrip."""
    print("\n=== Test: Rotation Matrix Roundtrip ===")
    angle = np.pi / 2
    q_z90 = Quaternion(w=np.cos(angle / 2), x=0, y=0, z=np.sin(angle / 2))
    R = q_z90.to_rotation_matrix()
    q_from_R = Quaternion.from_rotation_matrix(R)
    # Note: q and -q represent the same rotation
    dot = abs(q_z90.dot(q_from_R))
    assert np.isclose(dot, 1.0, atol=1e-10), f"Dot product should be ±1, got {dot}"
    print("  ✓ Quaternion -> Matrix -> Quaternion roundtrip works")


def test_quaternion_euler_angles() -> None:
    """Test euler angle extraction."""
    print("\n=== Test: Euler Angles ===")
    angle = np.pi / 2
    q_z90 = Quaternion(w=np.cos(angle / 2), x=0, y=0, z=np.sin(angle / 2))
    roll, pitch, yaw = q_z90.to_euler_xyz()
    assert np.isclose(yaw, np.pi / 2, atol=1e-10), f"Yaw should be π/2, got {yaw}"
    assert np.isclose(roll, 0, atol=1e-10), f"Roll should be 0, got {roll}"
    assert np.isclose(pitch, 0, atol=1e-10), f"Pitch should be 0, got {pitch}"
    print(f"  ✓ Euler angles for 90° Z-rotation: roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")


def test_quaternion_axis_angle() -> None:
    """Test axis-angle extraction."""
    print("\n=== Test: Axis-Angle ===")
    angle = np.pi / 2
    q_z90 = Quaternion(w=np.cos(angle / 2), x=0, y=0, z=np.sin(angle / 2))
    axis, angle_out = q_z90.to_axis_angle()
    assert np.allclose(axis, [0, 0, 1], atol=1e-10), f"Axis should be [0,0,1], got {axis}"
    assert np.isclose(angle_out, np.pi / 2, atol=1e-10), f"Angle should be π/2, got {angle_out}"
    print(f"  ✓ Axis: {axis}, Angle: {np.degrees(angle_out):.1f}°")


def test_quaternion_slerp() -> None:
    """Test SLERP interpolation."""
    print("\n=== Test: SLERP ===")
    q0 = Quaternion.identity()
    angle = np.pi / 2
    q1 = Quaternion(w=np.cos(angle / 2), x=0, y=0, z=np.sin(angle / 2))
    q_half = Quaternion.slerp(q0, q1, 0.5)
    _, angle_half = q_half.to_axis_angle()
    assert np.isclose(angle_half, np.pi / 4, atol=1e-10), f"Half-SLERP should give 45°, got {np.degrees(angle_half)}°"
    print(f"  ✓ SLERP(identity, 90°Z, 0.5) = {np.degrees(angle_half):.1f}°")


def test_quaternion_slerp_edge_cases() -> None:
    """Test SLERP edge cases."""
    print("\n=== Test: SLERP Edge Cases ===")
    q0 = Quaternion.identity()
    angle = np.pi / 2
    q1 = Quaternion(w=np.cos(angle / 2), x=0, y=0, z=np.sin(angle / 2))
    # t=0 should give q0
    q_t0 = Quaternion.slerp(q0, q1, 0.0)
    assert np.isclose(abs(q_t0.dot(q0)), 1.0, atol=1e-10)
    # t=1 should give q1
    q_t1 = Quaternion.slerp(q0, q1, 1.0)
    assert np.isclose(abs(q_t1.dot(q1)), 1.0, atol=1e-10)
    print("  ✓ SLERP at t=0 and t=1 work correctly")


def test_quaternion_180_degree_rotation() -> None:
    """Test 180° rotation (edge case)."""
    print("\n=== Test: 180° Rotation ===")
    q_180z = Quaternion(w=0, x=0, y=0, z=1)  # 180° around Z
    v_x = np.array([1.0, 0.0, 0.0])
    v_neg_x = q_180z.rotate_vector(v_x)
    assert np.allclose(v_neg_x, [-1, 0, 0], atol=1e-10), f"180° Z should flip X: got {v_neg_x}"
    print(f"  ✓ 180° Z-rotation: [1,0,0] -> {v_neg_x}")


def test_quaternion_auto_normalization() -> None:
    """Test auto-normalization."""
    print("\n=== Test: Auto-normalization ===")
    q_unnorm = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0)
    norm_sq = q_unnorm.w**2 + q_unnorm.x**2 + q_unnorm.y**2 + q_unnorm.z**2
    assert np.isclose(norm_sq, 1.0, atol=1e-10), f"Should be normalized, got norm² = {norm_sq}"
    print("  ✓ Quaternions are auto-normalized on creation")


def test_quaternion_resampling() -> None:
    """Test quaternion resampling."""
    print("\n=== Test: Quaternion Resampling ===")
    angles = np.linspace(0, np.pi / 2, 5)
    quats = [Quaternion(w=np.cos(a / 2), x=0, y=0, z=np.sin(a / 2)) for a in angles]
    orig_ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    target_ts = np.array([0.125, 0.375, 0.625, 0.875])

    resampled = resample_quaternions(quats, orig_ts, target_ts)
    assert len(resampled) == 4
    # Check middle point (t=0.375 should be ~33.75° rotation)
    _, angle_mid = resampled[1].to_axis_angle()
    expected_angle = 0.375 * (np.pi / 2)
    assert np.isclose(angle_mid, expected_angle, atol=0.02), f"Expected {np.degrees(expected_angle):.1f}°, got {np.degrees(angle_mid):.1f}°"
    print(f"  ✓ Resampled quaternion at t=0.375: {np.degrees(angle_mid):.1f}° (expected {np.degrees(expected_angle):.1f}°)")


def test_quaternion_rotation_matrix_properties() -> None:
    """Test rotation matrix properties."""
    print("\n=== Test: Rotation Matrix Properties ===")
    # Random-ish quaternion
    q_test = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
    R_test = q_test.to_rotation_matrix()
    # Check orthonormality: R @ R.T = I
    RRT = R_test @ R_test.T
    assert np.allclose(RRT, np.eye(3), atol=1e-10), f"R @ R.T should be identity"
    # Check determinant = +1 (proper rotation)
    det = np.linalg.det(R_test)
    assert np.isclose(det, 1.0, atol=1e-10), f"det(R) should be 1, got {det}"
    print("  ✓ Rotation matrix is orthonormal with det=+1")


# =============================================================================
# REFERENCE GEOMETRY TESTS
# =============================================================================


def test_reference_geometry_load_example() -> None:
    """Test loading example reference geometry."""
    print("\n=== Test: Reference Geometry Load Example ===")
    example_data = json_module.loads(EXAMPLE_JSON)
    geom = ReferenceGeometry.model_validate(example_data)

    assert geom.units == "mm"
    assert geom.coordinate_frame.origin_markers == ["left_eye", "right_eye"]
    print(f"  Units: {geom.units}")
    print(f"  Origin markers: {geom.coordinate_frame.origin_markers}")
    print(f"  Markers ({len(geom.markers)}): {list(geom.markers.keys())}")
    print("  ✓ Example geometry loaded correctly")


def test_reference_geometry_axis_definitions() -> None:
    """Test axis definition parsing."""
    print("\n=== Test: Reference Geometry Axis Definitions ===")
    example_data = json_module.loads(EXAMPLE_JSON)
    geom = ReferenceGeometry.model_validate(example_data)

    exact, approx = geom.coordinate_frame.get_defined_axes()
    computed = geom.coordinate_frame.get_computed_axis()

    assert exact == "x_axis"
    assert approx == "y_axis"
    assert computed == "z_axis"

    print(f"  Exact axis: {exact}")
    print(f"  Approximate axis: {approx}")
    print(f"  Computed axis: {computed}")
    print("  ✓ Axis definitions parsed correctly")


def test_reference_geometry_basis_vectors() -> None:
    """Test basis vector computation."""
    print("\n=== Test: Reference Geometry Basis Vectors ===")
    example_data = json_module.loads(EXAMPLE_JSON)
    geom = ReferenceGeometry.model_validate(example_data)

    basis, origin = geom.compute_basis_vectors()

    # Verify orthonormality
    assert np.isclose(np.linalg.norm(basis[0]), 1.0, atol=1e-6), f"|X| = {np.linalg.norm(basis[0])}"
    assert np.isclose(np.linalg.norm(basis[1]), 1.0, atol=1e-6), f"|Y| = {np.linalg.norm(basis[1])}"
    assert np.isclose(np.linalg.norm(basis[2]), 1.0, atol=1e-6), f"|Z| = {np.linalg.norm(basis[2])}"
    assert np.isclose(np.dot(basis[0], basis[1]), 0.0, atol=1e-6), f"X·Y = {np.dot(basis[0], basis[1])}"
    assert np.isclose(np.dot(basis[0], basis[2]), 0.0, atol=1e-6), f"X·Z = {np.dot(basis[0], basis[2])}"
    assert np.isclose(np.dot(basis[1], basis[2]), 0.0, atol=1e-6), f"Y·Z = {np.dot(basis[1], basis[2])}"
    assert np.isclose(np.linalg.det(basis), 1.0, atol=1e-6), f"det(basis) = {np.linalg.det(basis)}"

    print(f"  Origin: {origin}")
    print(f"  X: {basis[0]}")
    print(f"  Y: {basis[1]}")
    print(f"  Z: {basis[2]}")
    print("  ✓ Basis vectors are orthonormal with det=+1")


def test_reference_geometry_reject_single_axis() -> None:
    """Test rejection of single axis definition."""
    print("\n=== Test: Reference Geometry Reject Single Axis ===")
    bad_data = json_module.loads(EXAMPLE_JSON)
    del bad_data["coordinate_frame"]["y_axis"]
    try:
        ReferenceGeometry.model_validate(bad_data)
        raise AssertionError("Should have rejected single axis definition")
    except ValueError as e:
        print(f"  ✓ Correctly rejected single axis: {e}")


def test_reference_geometry_reject_two_exact_axes() -> None:
    """Test rejection of two exact axes."""
    print("\n=== Test: Reference Geometry Reject Two Exact Axes ===")
    bad_data = json_module.loads(EXAMPLE_JSON)
    bad_data["coordinate_frame"]["y_axis"]["type"] = "exact"
    try:
        ReferenceGeometry.model_validate(bad_data)
        raise AssertionError("Should have rejected two exact axes")
    except ValueError as e:
        print(f"  ✓ Correctly rejected two exact axes: {e}")


def test_reference_geometry_reject_invalid_marker() -> None:
    """Test rejection of invalid marker reference."""
    print("\n=== Test: Reference Geometry Reject Invalid Marker ===")
    bad_data = json_module.loads(EXAMPLE_JSON)
    bad_data["coordinate_frame"]["x_axis"]["markers"] = ["nonexistent"]
    try:
        ReferenceGeometry.model_validate(bad_data)
        raise AssertionError("Should have rejected invalid marker reference")
    except ValueError as e:
        print(f"  ✓ Correctly rejected invalid marker: {e}")


# =============================================================================
# TIMESERIES TESTS
# =============================================================================


def test_timeseries_basic() -> None:
    """Test Timeseries class basic functionality."""
    print("\n=== Testing Timeseries Basic ===")

    timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    values = np.array([0.0, 1.0, 4.0, 9.0, 16.0])

    ts = Timeseries(name="test", timestamps=timestamps, values=values)

    assert ts.n_frames == 5, f"Expected 5 frames, got {ts.n_frames}"
    assert ts.duration == 0.4, f"Expected duration 0.4, got {ts.duration}"
    assert ts[2] == 4.0, f"Expected ts[2]=4.0, got {ts[2]}"
    assert len(ts) == 5, f"Expected len=5, got {len(ts)}"
    assert np.isclose(ts.mean_dt, 0.1), f"Expected mean_dt=0.1, got {ts.mean_dt}"

    print("  ✓ Basic properties correct")


def test_timeseries_differentiation() -> None:
    """Test Timeseries differentiation."""
    print("\n=== Testing Timeseries Differentiation ===")

    # Use a linear function: y = 2t + 1, derivative should be 2
    timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    values = 2 * timestamps + 1

    ts = Timeseries(name="linear", timestamps=timestamps, values=values)
    dts = ts.differentiate()

    assert dts.name == "d(linear)/dt"
    # Central differences for middle points should give exact derivative
    assert np.allclose(dts.values[1:-1], 2.0, atol=1e-10), f"Derivative should be 2, got {dts.values}"
    print(f"  Derivative values: {dts.values}")
    print("  ✓ Differentiation correct for linear function")


def test_timeseries_differentiation_quadratic() -> None:
    """Test differentiation of quadratic function."""
    print("\n=== Testing Timeseries Differentiation (Quadratic) ===")

    # y = t^2, derivative should be 2t
    timestamps = np.linspace(0.0, 1.0, 21)  # Dense sampling for accuracy
    values = timestamps**2

    ts = Timeseries(name="quadratic", timestamps=timestamps, values=values)
    dts = ts.differentiate()

    expected_derivative = 2 * timestamps
    # Check middle points (avoid boundary effects)
    assert np.allclose(dts.values[2:-2], expected_derivative[2:-2], atol=0.01)
    print("  ✓ Differentiation correct for quadratic function")


def test_timeseries_interpolation() -> None:
    """Test Timeseries interpolation."""
    print("\n=== Testing Timeseries Interpolation ===")

    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 10.0, 20.0, 30.0])  # Linear

    ts = Timeseries(name="linear", timestamps=timestamps, values=values)
    new_timestamps = np.array([0.5, 1.5, 2.5])
    interp = ts.interpolate(new_timestamps)

    expected = np.array([5.0, 15.0, 25.0])
    assert np.allclose(interp.values, expected), f"Expected {expected}, got {interp.values}"
    assert len(interp) == 3
    print(f"  Interpolated values: {interp.values}")
    print("  ✓ Interpolation correct")


def test_timeseries_validation() -> None:
    """Test Timeseries validation."""
    print("\n=== Testing Timeseries Validation ===")

    # Mismatched lengths should raise
    try:
        Timeseries(
            name="bad",
            timestamps=np.array([0.0, 1.0]),
            values=np.array([0.0, 1.0, 2.0]),
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly rejected mismatched lengths: {e}")


# =============================================================================
# VEC3 TRAJECTORY TESTS
# =============================================================================


def test_vec3_trajectory_basic() -> None:
    """Test Vec3Trajectory class basic functionality."""
    print("\n=== Testing Vec3Trajectory Basic ===")

    timestamps = np.array([0.0, 0.1, 0.2, 0.3])
    values = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
    ])

    traj = Vector3Trajectory(name="position", timestamps=timestamps, values=values)

    assert traj.n_frames == 4
    assert traj.x.name == "position.x"
    assert np.allclose(traj.x.values, [0, 1, 2, 3])
    assert np.allclose(traj.y.values, [0, 2, 4, 6])
    assert np.allclose(traj.z.values, [0, 3, 6, 9])

    # Test indexing
    assert np.allclose(traj[2], [2.0, 4.0, 6.0])

    print("  ✓ Vec3Trajectory basic properties correct")


def test_vec3_trajectory_magnitude() -> None:
    """Test Vec3Trajectory magnitude computation."""
    print("\n=== Testing Vec3Trajectory Magnitude ===")

    timestamps = np.array([0.0, 1.0])
    values = np.array([
        [3.0, 4.0, 0.0],  # magnitude = 5
        [0.0, 0.0, 5.0],  # magnitude = 5
    ])

    traj = Vector3Trajectory(name="test", timestamps=timestamps, values=values)
    mag = traj.magnitude

    assert np.allclose(mag.values, [5.0, 5.0])
    print(f"  Magnitudes: {mag.values}")
    print("  ✓ Magnitude computation correct")


def test_vec3_trajectory_differentiation() -> None:
    """Test Vec3Trajectory differentiation."""
    print("\n=== Testing Vec3Trajectory Differentiation ===")

    timestamps = np.linspace(0.0, 1.0, 11)
    # Linear motion: position = t * [1, 2, 3]
    values = np.outer(timestamps, np.array([1.0, 2.0, 3.0]))

    traj = Vector3Trajectory(name="position", timestamps=timestamps, values=values)
    dtraj = traj.differentiate()

    # Velocity should be constant [1, 2, 3]
    expected_vel = np.array([1.0, 2.0, 3.0])
    # Check middle points
    for i in range(2, len(timestamps) - 2):
        assert np.allclose(dtraj[i], expected_vel, atol=0.01), f"Frame {i}: expected {expected_vel}, got {dtraj[i]}"

    print("  ✓ Vec3Trajectory differentiation correct")


def test_vec3_trajectory_validation() -> None:
    """Test Vec3Trajectory validation."""
    print("\n=== Testing Vec3Trajectory Validation ===")

    try:
        Vector3Trajectory(
            name="bad",
            timestamps=np.array([0.0, 1.0]),
            values=np.array([[1, 2], [3, 4]]),  # Wrong shape (N, 2) not (N, 3)
        )
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly rejected wrong shape: {e}")


# =============================================================================
# QUATERNION TRAJECTORY TESTS
# =============================================================================


def test_quaternion_trajectory_basic() -> None:
    """Test QuaternionTrajectory class basic functionality."""
    print("\n=== Testing QuaternionTrajectory Basic ===")

    timestamps = np.array([0.0, 0.1, 0.2, 0.3])
    angles = np.array([0, np.pi / 6, np.pi / 3, np.pi / 2])

    quaternions = [
        Quaternion(w=np.cos(a / 2), x=0, y=0, z=np.sin(a / 2))
        for a in angles
    ]

    traj = QuaternionTrajectory(name="orientation", timestamps=timestamps, quaternions=quaternions)

    assert traj.n_frames == 4
    assert traj[0].w == 1.0  # Identity at t=0

    print("  ✓ QuaternionTrajectory basic properties correct")


def test_quaternion_trajectory_euler_angles() -> None:
    """Test QuaternionTrajectory euler angle extraction."""
    print("\n=== Testing QuaternionTrajectory Euler Angles ===")

    timestamps = np.array([0.0, 0.1, 0.2, 0.3])
    angles = np.array([0, np.pi / 6, np.pi / 3, np.pi / 2])

    # Z-rotation only
    quaternions = [
        Quaternion(w=np.cos(a / 2), x=0, y=0, z=np.sin(a / 2))
        for a in angles
    ]

    traj = QuaternionTrajectory(name="orientation", timestamps=timestamps, quaternions=quaternions)

    # Yaw should match angles
    yaw_ts = traj.yaw
    assert np.allclose(yaw_ts.values, angles, atol=1e-10), f"Yaw mismatch: expected {angles}, got {yaw_ts.values}"

    # Roll and pitch should be zero for Z-rotation
    assert np.allclose(traj.roll.values, 0, atol=1e-10), f"Roll should be 0, got {traj.roll.values}"
    assert np.allclose(traj.pitch.values, 0, atol=1e-10), f"Pitch should be 0, got {traj.pitch.values}"

    print(f"  Yaw angles (deg): {np.degrees(yaw_ts.values)}")
    print("  ✓ Euler angles correct for Z-rotation")


def test_quaternion_trajectory_rotation_matrices() -> None:
    """Test QuaternionTrajectory rotation matrix conversion."""
    print("\n=== Testing QuaternionTrajectory Rotation Matrices ===")

    timestamps = np.array([0.0, 1.0])
    quaternions = [
        Quaternion.identity(),
        Quaternion(w=np.cos(np.pi / 4), x=0, y=0, z=np.sin(np.pi / 4)),  # 90° Z
    ]

    traj = QuaternionTrajectory(name="orientation", timestamps=timestamps, quaternions=quaternions)
    R_all = traj.to_rotation_matrices()

    assert R_all.shape == (2, 3, 3)

    # First should be identity
    assert np.allclose(R_all[0], np.eye(3), atol=1e-10)

    # Second should be 90° Z rotation
    expected_R1 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    assert np.allclose(R_all[1], expected_R1, atol=1e-10), f"Expected:\n{expected_R1}\nGot:\n{R_all[1]}"

    print("  ✓ Rotation matrices correct")


def test_quaternion_trajectory_components() -> None:
    """Test QuaternionTrajectory component extraction."""
    print("\n=== Testing QuaternionTrajectory Components ===")

    timestamps = np.array([0.0, 1.0])
    quaternions = [
        Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        Quaternion(w=0.5, x=0.5, y=0.5, z=0.5),
    ]

    traj = QuaternionTrajectory(name="q", timestamps=timestamps, quaternions=quaternions)

    assert np.allclose(traj.w.values, [1.0, 0.5])
    assert np.allclose(traj.x.values, [0.0, 0.5])
    assert np.allclose(traj.y.values, [0.0, 0.5])
    assert np.allclose(traj.z.values, [0.0, 0.5])

    print("  ✓ Component extraction correct")


# =============================================================================
# ANGULAR VELOCITY TRAJECTORY TESTS
# =============================================================================


def test_angular_velocity_trajectory_basic() -> None:
    """Test AngularVelocityTrajectory class basic functionality."""
    print("\n=== Testing AngularVelocityTrajectory Basic ===")

    timestamps = np.array([0.0, 0.1, 0.2])
    global_xyz = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ])
    local_xyz = global_xyz * 0.9

    traj = AngularVelocityTrajectory(
        name="omega",
        timestamps=timestamps,
        global_xyz=global_xyz,
        local_xyz=local_xyz,
    )

    assert traj.n_frames == 3
    assert np.allclose(traj.global_roll.values, [0.1, 0.2, 0.3])
    assert np.allclose(traj.global_pitch.values, [0.2, 0.3, 0.4])
    assert np.allclose(traj.global_yaw.values, [0.3, 0.4, 0.5])

    print("  ✓ AngularVelocityTrajectory basic properties correct")


def test_angular_velocity_trajectory_magnitude() -> None:
    """Test AngularVelocityTrajectory magnitude computation."""
    print("\n=== Testing AngularVelocityTrajectory Magnitude ===")

    timestamps = np.array([0.0, 1.0])
    global_xyz = np.array([
        [3.0, 4.0, 0.0],  # magnitude = 5
        [0.0, 0.0, 2.0],  # magnitude = 2
    ])
    local_xyz = global_xyz.copy()

    traj = AngularVelocityTrajectory(
        name="omega",
        timestamps=timestamps,
        global_xyz=global_xyz,
        local_xyz=local_xyz,
    )

    mag = traj.global_magnitude
    assert np.allclose(mag.values, [5.0, 2.0])
    print(f"  Angular speeds: {mag.values}")
    print("  ✓ Magnitude computation correct")


# =============================================================================
# RIGID BODY POSE TESTS
# =============================================================================


def test_rigid_body_pose_basic() -> None:
    """Test RigidBodyPose class basic functionality."""
    print("\n=== Testing RigidBodyPose Basic ===")

    geom = create_test_geometry()

    pose = RigidBodyState(
        reference_geometry=geom,
        timestamp=1.5,
        position=np.array([100.0, 50.0, 0.0]),
        velocity=np.array([10.0, 20.0, 0.0]),
        orientation=Quaternion.identity(),
        angular_velocity_global=np.array([0.0, 0.0, 1.0]),
        angular_velocity_local=np.array([0.0, 0.0, 1.0]),
    )

    assert pose.timestamp == 1.5
    assert np.isclose(pose.speed, np.sqrt(10**2 + 20**2))
    assert pose.angular_speed == 1.0

    print(f"  Speed: {pose.speed:.2f} mm/s")
    print(f"  Angular speed: {pose.angular_speed:.2f} rad/s")
    print("  ✓ Basic properties correct")


def test_rigid_body_pose_keypoints_identity() -> None:
    """Test RigidBodyPose keypoints with identity orientation."""
    print("\n=== Testing RigidBodyPose Keypoints (Identity) ===")

    geom = create_test_geometry()

    pose = RigidBodyState(
        reference_geometry=geom,
        timestamp=0.0,
        position=np.array([100.0, 50.0, 0.0]),
        velocity=np.zeros(3),
        orientation=Quaternion.identity(),
        angular_velocity_global=np.zeros(3),
        angular_velocity_local=np.zeros(3),
    )

    keypoints = pose.keypoints
    assert "nose" in keypoints

    # With identity orientation, keypoints are just position + local offset
    expected_nose = np.array([100.0 + 18.0, 50.0, 0.0])
    assert np.allclose(keypoints["nose"], expected_nose), f"Expected {expected_nose}, got {keypoints['nose']}"

    # Test single keypoint accessor
    nose = pose.get_keypoint("nose")
    assert np.allclose(nose, expected_nose)

    print(f"  Nose position: {keypoints['nose']}")
    print("  ✓ Keypoints correct with identity orientation")


def test_rigid_body_pose_keypoints_rotated() -> None:
    """Test RigidBodyPose keypoints with rotation."""
    print("\n=== Testing RigidBodyPose Keypoints (Rotated) ===")

    geom = create_test_geometry()

    # 90° rotation around Z: X -> Y, Y -> -X
    q_90z = Quaternion(w=np.cos(np.pi / 4), x=0, y=0, z=np.sin(np.pi / 4))

    pose = RigidBodyState(
        reference_geometry=geom,
        timestamp=0.0,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        orientation=q_90z,
        angular_velocity_global=np.zeros(3),
        angular_velocity_local=np.zeros(3),
    )

    # Nose is at local [18, 0, 0], after 90° Z rotation should be at [0, 18, 0]
    nose = pose.get_keypoint("nose")
    expected_nose = np.array([0.0, 18.0, 0.0])
    assert np.allclose(nose, expected_nose, atol=1e-10), f"Expected {expected_nose}, got {nose}"

    # Left eye is at local [0, 12, 0], after 90° Z rotation should be at [-12, 0, 0]
    left_eye = pose.get_keypoint("left_eye")
    expected_left_eye = np.array([-12.0, 0.0, 0.0])
    assert np.allclose(left_eye, expected_left_eye, atol=1e-10), f"Expected {expected_left_eye}, got {left_eye}"

    print(f"  Nose (rotated): {nose}")
    print(f"  Left eye (rotated): {left_eye}")
    print("  ✓ Keypoints correct with rotation")


def test_rigid_body_pose_basis_vectors() -> None:
    """Test RigidBodyPose basis vectors."""
    print("\n=== Testing RigidBodyPose Basis Vectors ===")

    geom = create_test_geometry()

    # Identity orientation
    pose_id = RigidBodyState(
        reference_geometry=geom,
        timestamp=0.0,
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=Quaternion.identity(),
        angular_velocity_global=np.zeros(3),
        angular_velocity_local=np.zeros(3),
    )

    assert np.allclose(pose_id.basis_x, [1, 0, 0])
    assert np.allclose(pose_id.basis_y, [0, 1, 0])
    assert np.allclose(pose_id.basis_z, [0, 0, 1])

    # 90° Z rotation
    q_90z = Quaternion(w=np.cos(np.pi / 4), x=0, y=0, z=np.sin(np.pi / 4))
    pose_rot = RigidBodyState(
        reference_geometry=geom,
        timestamp=0.0,
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=q_90z,
        angular_velocity_global=np.zeros(3),
        angular_velocity_local=np.zeros(3),
    )

    assert np.allclose(pose_rot.basis_x, [0, 1, 0], atol=1e-10)
    assert np.allclose(pose_rot.basis_y, [-1, 0, 0], atol=1e-10)
    assert np.allclose(pose_rot.basis_z, [0, 0, 1], atol=1e-10)

    print("  ✓ Basis vectors correct")


def test_rigid_body_pose_euler_angles() -> None:
    """Test RigidBodyPose euler angle extraction."""
    print("\n=== Testing RigidBodyPose Euler Angles ===")

    geom = create_test_geometry()

    # 45° yaw
    q_45z = Quaternion(w=np.cos(np.pi / 8), x=0, y=0, z=np.sin(np.pi / 8))
    pose = RigidBodyState(
        reference_geometry=geom,
        timestamp=0.0,
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=q_45z,
        angular_velocity_global=np.zeros(3),
        angular_velocity_local=np.zeros(3),
    )

    roll, pitch, yaw = pose.euler_angles
    assert np.isclose(yaw, np.pi / 4, atol=1e-10), f"Yaw should be π/4, got {yaw}"
    assert np.isclose(roll, 0, atol=1e-10)
    assert np.isclose(pitch, 0, atol=1e-10)

    print(f"  Euler angles: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")
    print("  ✓ Euler angles correct")


def test_rigid_body_pose_homogeneous_transform() -> None:
    """Test RigidBodyPose homogeneous transform."""
    print("\n=== Testing RigidBodyPose Homogeneous Transform ===")

    geom = create_test_geometry()

    position = np.array([10.0, 20.0, 30.0])
    q_90z = Quaternion(w=np.cos(np.pi / 4), x=0, y=0, z=np.sin(np.pi / 4))

    pose = RigidBodyState(
        reference_geometry=geom,
        timestamp=0.0,
        position=position,
        velocity=np.zeros(3),
        orientation=q_90z,
        angular_velocity_global=np.zeros(3),
        angular_velocity_local=np.zeros(3),
    )

    T = pose.homogeneous_transform
    assert T.shape == (4, 4)

    # Check translation
    assert np.allclose(T[:3, 3], position)

    # Check rotation
    assert np.allclose(T[:3, :3], pose.basis_vectors)

    # Check last row
    assert np.allclose(T[3, :], [0, 0, 0, 1])

    # Test transformation of a point
    local_point = np.array([1.0, 0.0, 0.0, 1.0])  # Homogeneous
    world_point = T @ local_point
    expected = position + q_90z.rotate_vector(local_point[:3])
    assert np.allclose(world_point[:3], expected)

    print("  ✓ Homogeneous transform correct")


def test_rigid_body_pose_keypoint_not_found() -> None:
    """Test RigidBodyPose keypoint error handling."""
    print("\n=== Testing RigidBodyPose Keypoint Error ===")

    geom = create_test_geometry()
    pose = RigidBodyState(
        reference_geometry=geom,
        timestamp=0.0,
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=Quaternion.identity(),
        angular_velocity_global=np.zeros(3),
        angular_velocity_local=np.zeros(3),
    )

    try:
        pose.get_keypoint("nonexistent")
        raise AssertionError("Should have raised KeyError")
    except KeyError as e:
        print(f"  ✓ Correctly raised KeyError: {e}")


# =============================================================================
# RIGID BODY KINEMATICS TESTS
# =============================================================================


def test_rigid_body_kinematics_basic() -> None:
    """Test RigidBodyKinematics class basic functionality."""
    print("\n=== Testing RigidBodyKinematics Basic ===")

    kin = create_test_kinematics(n_frames=10)

    assert kin.n_frames == 10
    assert kin.duration == 1.0
    assert len(kin) == 10

    print(f"  n_frames: {kin.n_frames}")
    print(f"  duration: {kin.duration}s")
    print("  ✓ Basic properties correct")


def test_rigid_body_kinematics_horizontal_slice() -> None:
    """Test RigidBodyKinematics horizontal slicing (get Pose)."""
    print("\n=== Testing RigidBodyKinematics Horizontal Slice ===")

    kin = create_test_kinematics(n_frames=10)

    pose0 = kin[0]
    assert pose0.timestamp == 0.0
    assert np.allclose(pose0.position, [100, 0, 0])

    pose5 = kin[5]
    print(f"  Pose at frame 5: t={pose5.timestamp:.2f}, pos=({pose5.position[0]:.1f}, {pose5.position[1]:.1f}, {pose5.position[2]:.1f})")

    # Alternative method
    pose_alt = kin.get_state_at_frame(5)
    assert np.allclose(pose_alt.position, pose5.position)

    print("  ✓ Horizontal slicing correct")


def test_rigid_body_kinematics_iteration() -> None:
    """Test RigidBodyKinematics iteration."""
    print("\n=== Testing RigidBodyKinematics Iteration ===")

    kin = create_test_kinematics(n_frames=10)

    count = 0
    timestamps_seen = []
    for pose in kin:
        count += 1
        timestamps_seen.append(pose.timestamp)

    assert count == 10
    assert np.allclose(timestamps_seen, kin.timestamps)

    print(f"  Iterated over {count} poses")
    print("  ✓ Iteration correct")


def test_rigid_body_kinematics_vertical_slice() -> None:
    """Test RigidBodyKinematics vertical slicing (get Trajectory)."""
    print("\n=== Testing RigidBodyKinematics Vertical Slice ===")

    kin = create_test_kinematics(n_frames=10)

    pos_traj = kin.position_trajectory
    assert pos_traj.n_frames == 10
    assert pos_traj.x.name == "position.x"

    vel_traj = kin.velocity_trajectory
    assert vel_traj.n_frames == 10

    orient_traj = kin.orientation_trajectory
    assert orient_traj.n_frames == 10

    angvel_traj = kin.angular_velocity_trajectory
    assert angvel_traj.n_frames == 10

    print("  ✓ Vertical slicing correct")


def test_rigid_body_kinematics_convenience_accessors() -> None:
    """Test RigidBodyKinematics convenience accessors."""
    print("\n=== Testing RigidBodyKinematics Convenience Accessors ===")

    kin = create_test_kinematics(n_frames=10)

    # Position components
    assert kin.x.n_frames == 10
    assert kin.y.n_frames == 10
    assert kin.z.n_frames == 10

    # Velocity components
    assert kin.vx.n_frames == 10
    assert kin.vy.n_frames == 10
    assert kin.vz.n_frames == 10

    # Speed
    speed = kin.speed
    assert speed.n_frames == 10

    # Orientation angles
    assert kin.roll.n_frames == 10
    assert kin.pitch.n_frames == 10
    assert kin.yaw.n_frames == 10

    # Angular velocity
    assert kin.angular_speed.n_frames == 10

    print("  ✓ Convenience accessors work correctly")


def test_rigid_body_kinematics_keypoint_trajectory() -> None:
    """Test RigidBodyKinematics keypoint trajectory extraction."""
    print("\n=== Testing RigidBodyKinematics Keypoint Trajectory ===")

    kin = create_test_kinematics(n_frames=10)

    nose_traj = kin.get_keypoint_trajectory("nose")
    assert nose_traj.n_frames == 10
    assert nose_traj.name == "keypoint.nose"

    # Check keypoint names
    names = kin.keypoint_names
    assert "nose" in names
    assert "left_eye" in names
    assert "right_eye" in names
    assert "back" in names

    print(f"  Keypoint names: {names}")
    print(f"  Nose X range: {nose_traj.x.values.min():.1f} - {nose_traj.x.values.max():.1f} mm")
    print("  ✓ Keypoint trajectory extraction correct")


def test_rigid_body_kinematics_keypoint_not_found() -> None:
    """Test RigidBodyKinematics keypoint error handling."""
    print("\n=== Testing RigidBodyKinematics Keypoint Error ===")

    kin = create_test_kinematics(n_frames=5)

    try:
        kin.get_keypoint_trajectory("nonexistent")
        raise AssertionError("Should have raised KeyError")
    except KeyError as e:
        print(f"  ✓ Correctly raised KeyError: {e}")


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================


def test_kinematics_keypoint_consistency() -> None:
    """Test that keypoints are consistent between Pose and Trajectory views."""
    print("\n=== Testing Keypoint Consistency ===")

    kin = create_test_kinematics(n_frames=5)

    # Get nose trajectory via vertical slice
    nose_traj = kin.get_keypoint_trajectory("nose")

    # Compare with horizontal slice at each frame
    for i in range(kin.n_frames):
        pose = kin[i]
        nose_from_pose = pose.get_keypoint("nose")
        nose_from_traj = nose_traj[i]

        diff = np.linalg.norm(nose_from_pose - nose_from_traj)
        assert diff < 1e-10, f"Mismatch at frame {i}: {diff}"

    print("  ✓ Keypoint consistency verified across all frames")


def test_kinematics_velocity_consistency() -> None:
    """Test that velocity is consistent with position differentiation."""
    print("\n=== Testing Velocity Consistency ===")

    kin = create_test_kinematics(n_frames=20)

    # Differentiate position to get velocity
    x = kin.x
    dx_dt = x.differentiate()

    # Compare with actual velocity
    vx = kin.velocity_trajectory.x

    # Should be similar (not exact due to different differentiation methods)
    correlation = np.corrcoef(dx_dt.values[1:-1], vx.values[1:-1])[0, 1]
    print(f"  Correlation between d(x)/dt and vx: {correlation:.4f}")
    assert correlation > 0.99, f"Correlation too low: {correlation}"

    print("  ✓ Velocity consistency verified")


def test_kinematics_stationary() -> None:
    """Test kinematics with no motion."""
    print("\n=== Testing Stationary Kinematics ===")

    kin = create_stationary_kinematics(n_frames=5)

    # All positions should be at origin
    assert np.allclose(kin.position_xyz, 0)

    # All velocities should be zero
    assert np.allclose(kin.velocity_xyz, 0)

    # Speed should be zero
    assert np.allclose(kin.speed.values, 0)

    # Angular speed should be zero
    assert np.allclose(kin.angular_speed.values, 0)

    # Keypoints should be at their local positions (no rotation)
    nose_traj = kin.get_keypoint_trajectory("nose")
    assert np.allclose(nose_traj.x.values, 18.0)  # Nose local X
    assert np.allclose(nose_traj.y.values, 0.0)
    assert np.allclose(nose_traj.z.values, 0.0)

    print("  ✓ Stationary kinematics correct")


# =============================================================================
# FACTORY METHOD TESTS
# =============================================================================


def test_from_pose_arrays() -> None:
    """Test RigidBodyKinematics.from_pose_arrays factory method."""
    print("\n=== Testing from_pose_arrays Factory ===")

    geom = create_test_geometry()
    n_frames = 10

    timestamps = np.linspace(0.0, 1.0, n_frames)

    # Circular motion
    angles = np.linspace(0, np.pi / 2, n_frames)
    position_xyz = np.column_stack([
        100 * np.cos(angles),
        100 * np.sin(angles),
        np.zeros(n_frames),
    ])

    # Quaternions as array
    quaternions_wxyz = np.column_stack([
        np.cos(angles / 2),  # w
        np.zeros(n_frames),  # x
        np.zeros(n_frames),  # y
        np.sin(angles / 2),  # z
    ])

    kin = RigidBodyKinematics.from_pose_arrays(
        reference_geometry=geom,
        timestamps=timestamps,
        position_xyz=position_xyz,
        quaternions_wxyz=quaternions_wxyz,
    )

    assert kin.n_frames == n_frames
    assert kin.duration == 1.0

    # Check that velocities were computed
    assert not np.allclose(kin.velocity_xyz, 0)

    # Check that angular velocities were computed
    # For constant angular velocity Z-rotation, should be approximately constant
    omega_z = kin.angular_velocity_trajectory.global_yaw.values
    print(f"  Angular velocity Z range: {omega_z.min():.2f} - {omega_z.max():.2f} rad/s")

    print("  ✓ from_pose_arrays factory works correctly")


# =============================================================================
# MATH VALIDATION TESTS
# =============================================================================


def test_circular_motion_physics() -> None:
    """Validate circular motion physics."""
    print("\n=== Testing Circular Motion Physics ===")

    n_frames = 50
    kin = create_test_kinematics(n_frames=n_frames)

    # For circular motion, speed should be approximately constant
    # v = r * omega, where r = 100mm and omega = (π/2) rad/s
    speed = kin.speed.values

    # Exclude boundary effects
    speed_mid = speed[5:-5]
    speed_mean = np.mean(speed_mid)
    speed_std = np.std(speed_mid)

    print(f"  Speed: mean={speed_mean:.1f}, std={speed_std:.2f} mm/s")
    assert speed_std / speed_mean < 0.1, f"Speed should be nearly constant, but std/mean = {speed_std/speed_mean:.2f}"

    # Theoretical speed: v = r * omega = 100 * (π/2) ≈ 157 mm/s
    theoretical_speed = 100 * (np.pi / 2)
    print(f"  Theoretical speed: {theoretical_speed:.1f} mm/s")

    # Check radius is constant
    pos = kin.position_trajectory
    radius = pos.magnitude.values
    assert np.allclose(radius, 100, atol=1.0), f"Radius should be 100mm, got range [{radius.min():.1f}, {radius.max():.1f}]"

    print("  ✓ Circular motion physics validated")


def test_yaw_progression() -> None:
    """Test that yaw angle progresses correctly."""
    print("\n=== Testing Yaw Progression ===")

    kin = create_test_kinematics(n_frames=10)

    yaw = kin.yaw.values
    expected_yaw = np.linspace(0, np.pi / 2, 10)

    assert np.allclose(yaw, expected_yaw, atol=1e-10), "Yaw mismatch"

    print(f"  Yaw range: {np.degrees(yaw[0]):.1f}° - {np.degrees(yaw[-1]):.1f}°")
    print("  ✓ Yaw progression correct")


# =============================================================================
# REFERENCE GEOMETRY TESTS (axis combinations)
# =============================================================================


def test_compute_basis_vectors_all_combinations() -> None:
    """Test that compute_basis_vectors works correctly for all axis combinations."""
    print("\n=== Testing compute_basis_vectors All Combinations ===")

    base_markers = {
        "point_x": {"x": 10.0, "y": 0.0, "z": 0.0},
        "point_y": {"x": 0.0, "y": 10.0, "z": 0.0},
        "point_z": {"x": 0.0, "y": 0.0, "z": 10.0},
        "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    # All valid combinations: (exact_axis, approx_axis, computed_axis, exact_marker, approx_marker)
    combinations = [
        ("x_axis", "y_axis", "z_axis", "point_x", "point_y"),
        ("x_axis", "z_axis", "y_axis", "point_x", "point_z"),
        ("y_axis", "x_axis", "z_axis", "point_y", "point_x"),
        ("y_axis", "z_axis", "x_axis", "point_y", "point_z"),
        ("z_axis", "x_axis", "y_axis", "point_z", "point_x"),
        ("z_axis", "y_axis", "x_axis", "point_z", "point_y"),
    ]

    for exact_axis, approx_axis, computed_axis, exact_marker, approx_marker in combinations:
        coord_frame = {
            "origin_markers": ["origin"],
            exact_axis: {"markers": [exact_marker], "type": "exact"},
            approx_axis: {"markers": [approx_marker], "type": "approximate"},
        }

        geom = ReferenceGeometry(
            units="mm",
            coordinate_frame=coord_frame,
            markers=base_markers,
        )

        basis, origin = geom.compute_basis_vectors()

        # Check orthonormality
        assert np.allclose(np.linalg.norm(basis[0]), 1.0), f"X not unit: {basis[0]}"
        assert np.allclose(np.linalg.norm(basis[1]), 1.0), f"Y not unit: {basis[1]}"
        assert np.allclose(np.linalg.norm(basis[2]), 1.0), f"Z not unit: {basis[2]}"
        assert np.allclose(np.dot(basis[0], basis[1]), 0.0), "X·Y != 0"
        assert np.allclose(np.dot(basis[0], basis[2]), 0.0), "X·Z != 0"
        assert np.allclose(np.dot(basis[1], basis[2]), 0.0), "Y·Z != 0"

        # Check right-handed (det = +1)
        det = np.linalg.det(basis)
        assert np.allclose(det, 1.0), f"det = {det}, should be +1"

        # Check X × Y = Z
        cross_z = np.cross(basis[0], basis[1])
        assert np.allclose(cross_z, basis[2]), "X × Y != Z"

        print(f"  ✓ {exact_axis}(exact) + {approx_axis}(approx) -> {computed_axis}(computed)")

    print("  ✓ All axis combinations work correctly")


# =============================================================================
# RUN ALL TESTS
# =============================================================================


def run_all_tests() -> None:
    """Run all tests."""
    print("=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)

    # Quaternion helper tests
    test_quaternion_identity()
    test_quaternion_90_degree_z_rotation()
    test_quaternion_multiplication()
    test_quaternion_inverse()
    test_quaternion_rotation_matrix_roundtrip()
    test_quaternion_euler_angles()
    test_quaternion_axis_angle()
    test_quaternion_slerp()
    test_quaternion_slerp_edge_cases()
    test_quaternion_180_degree_rotation()
    test_quaternion_auto_normalization()
    test_quaternion_resampling()
    test_quaternion_rotation_matrix_properties()

    # Reference geometry tests
    test_reference_geometry_load_example()
    test_reference_geometry_axis_definitions()
    test_reference_geometry_basis_vectors()
    test_reference_geometry_reject_single_axis()
    test_reference_geometry_reject_two_exact_axes()
    test_reference_geometry_reject_invalid_marker()

    # Timeseries tests
    test_timeseries_basic()
    test_timeseries_differentiation()
    test_timeseries_differentiation_quadratic()
    test_timeseries_interpolation()
    test_timeseries_validation()

    # Vec3Trajectory tests
    test_vec3_trajectory_basic()
    test_vec3_trajectory_magnitude()
    test_vec3_trajectory_differentiation()
    test_vec3_trajectory_validation()

    # QuaternionTrajectory tests
    test_quaternion_trajectory_basic()
    test_quaternion_trajectory_euler_angles()
    test_quaternion_trajectory_rotation_matrices()
    test_quaternion_trajectory_components()

    # AngularVelocityTrajectory tests
    test_angular_velocity_trajectory_basic()
    test_angular_velocity_trajectory_magnitude()

    # RigidBodyPose tests
    test_rigid_body_pose_basic()
    test_rigid_body_pose_keypoints_identity()
    test_rigid_body_pose_keypoints_rotated()
    test_rigid_body_pose_basis_vectors()
    test_rigid_body_pose_euler_angles()
    test_rigid_body_pose_homogeneous_transform()
    test_rigid_body_pose_keypoint_not_found()

    # RigidBodyKinematics tests
    test_rigid_body_kinematics_basic()
    test_rigid_body_kinematics_horizontal_slice()
    test_rigid_body_kinematics_iteration()
    test_rigid_body_kinematics_vertical_slice()
    test_rigid_body_kinematics_convenience_accessors()
    test_rigid_body_kinematics_keypoint_trajectory()
    test_rigid_body_kinematics_keypoint_not_found()

    # Consistency tests
    test_kinematics_keypoint_consistency()
    test_kinematics_velocity_consistency()
    test_kinematics_stationary()

    # Factory method tests
    test_from_pose_arrays()

    # Math validation tests
    test_circular_motion_physics()
    test_yaw_progression()

    # Reference geometry axis combination tests
    test_compute_basis_vectors_all_combinations()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
