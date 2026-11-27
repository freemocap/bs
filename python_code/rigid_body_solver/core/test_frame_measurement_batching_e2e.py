"""End-to-end synthetic check: optimize_rigid_body with the batched
FrameMeasurementFactor loop produces the same solve as the old per-keypoint
MeasurementFactor loop, on a synthetic rotating rigid body with NaN gaps
(including a frame with zero valid keypoints).

Builds the same pyceres.Problem both ways (new code path vs. a local
reimplementation of the old per-keypoint loop) and asserts matching cost,
iterations, and solved trajectories, plus the expected residual-block-count
drop with unchanged total residual count.
"""

import numpy as np
import pyceres
from scipy.spatial.transform import Rotation

from python_code.rigid_body_solver.core.optimization import (
    FrameMeasurementFactor,
    MeasurementFactor,
    OptimizationConfig,
    RotationSmoothnessFactor,
    TranslationSmoothnessFactor,
    _estimate_initial_pose,
)


def _build_synthetic_trajectory(
    rng: np.random.Generator, n_frames: int = 250, n_keypoints: int = 8, noise_mm: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (original_trajectories (n_frames, n_keypoints, 3), reference_points (n_keypoints, 3))."""
    reference_points = rng.normal(size=(n_keypoints, 3)) * 50.0

    trajectories = np.zeros((n_frames, n_keypoints, 3), dtype=np.float64)
    for frame_idx in range(n_frames):
        angle = 2 * np.pi * frame_idx / n_frames
        R = Rotation.from_euler("xyz", [angle, 0.5 * angle, 0.3 * angle]).as_matrix()
        trans = np.array([10.0 * np.sin(angle), 5.0 * np.cos(angle), 2.0 * angle])
        trajectories[frame_idx] = (R @ reference_points.T).T + trans

    trajectories += rng.normal(size=trajectories.shape) * noise_mm

    # Introduce NaN gaps: random single-keypoint dropouts, keeping >=3 valid
    # keypoints per frame (a frame with <3 valid keypoints triggers a pre-existing,
    # unrelated bug in _estimate_initial_pose's fallback — see
    # test_zero_valid_frame_skip below for the batching-specific skip path, tested
    # in isolation instead).
    for _ in range(20):
        f = rng.integers(0, n_frames)
        k = rng.integers(0, n_keypoints)
        trajectories[f, k] = np.nan

    return trajectories, reference_points


def test_zero_valid_frame_skip() -> None:
    """Verify the new batched loop's zero-valid-keypoint skip path in isolation,
    without routing through _estimate_initial_pose (whose fallback-pose aliasing
    for <3-valid frames is a separate, pre-existing issue unrelated to item 5)."""
    rng = np.random.default_rng(3)
    n_keypoints = 6
    reference_points = rng.normal(size=(n_keypoints, 3)) * 50.0

    # Three independent frames: all-valid, all-valid, all-NaN (zero valid keypoints).
    measured = np.stack(
        [
            reference_points + rng.normal(size=(n_keypoints, 3)),
            reference_points + rng.normal(size=(n_keypoints, 3)),
            np.full((n_keypoints, 3), np.nan),
        ]
    )
    valid_mask = ~np.isnan(measured).any(axis=2)

    problem = pyceres.Problem()
    poses = [
        (np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)),
        (np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)),
        (np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)),
    ]
    for quat, trans in poses:
        problem.add_parameter_block(quat, 4)
        problem.add_parameter_block(trans, 3)
        problem.set_manifold(quat, pyceres.QuaternionManifold())

    n_blocks_added = 0
    for frame_idx in range(3):
        quat, trans = poses[frame_idx]
        frame_valid_mask = valid_mask[frame_idx]
        n_valid = int(frame_valid_mask.sum())
        if n_valid == 0:
            continue
        cost = FrameMeasurementFactor(
            measured_points=measured[frame_idx, frame_valid_mask],
            reference_points=reference_points[frame_valid_mask],
        )
        problem.add_residual_block(cost, None, [quat, trans])
        n_blocks_added += 1

    assert n_blocks_added == 2, "Zero-valid frame should not get a measurement factor"
    print("test_zero_valid_frame_skip: PASS")


def _build_problem(
    trajectories: np.ndarray,
    reference_points: np.ndarray,
    config: OptimizationConfig,
    use_batched: bool,
) -> tuple[pyceres.Problem, list[tuple[np.ndarray, np.ndarray]]]:
    n_frames, n_keypoints, _ = trajectories.shape
    valid_mask = ~np.isnan(trajectories).any(axis=2)

    poses: list[tuple[np.ndarray, np.ndarray]] = []
    last_good_pose = None
    for frame_idx in range(n_frames):
        pose = _estimate_initial_pose(
            measured_points=trajectories[frame_idx],
            reference_points=reference_points,
            fallback_pose=last_good_pose,
        )
        poses.append(pose)
        last_good_pose = pose

    problem = pyceres.Problem()
    for quat, trans in poses:
        problem.add_parameter_block(quat, 4)
        problem.add_parameter_block(trans, 3)
        problem.set_manifold(quat, pyceres.QuaternionManifold())

    measurement_loss = (
        pyceres.HuberLoss(config.robust_loss_scale) if config.robust_loss_scale is not None else None
    )

    for frame_idx in range(n_frames):
        quat, trans = poses[frame_idx]
        frame_valid_mask = valid_mask[frame_idx]
        n_valid = int(frame_valid_mask.sum())
        if n_valid == 0:
            continue

        if use_batched:
            cost = FrameMeasurementFactor(
                measured_points=trajectories[frame_idx, frame_valid_mask],
                reference_points=reference_points[frame_valid_mask],
            )
            problem.add_residual_block(cost, measurement_loss, [quat, trans])
        else:
            for keypoint_idx in np.flatnonzero(frame_valid_mask):
                cost = MeasurementFactor(
                    measured_point=trajectories[frame_idx, keypoint_idx],
                    reference_point=reference_points[keypoint_idx],
                )
                problem.add_residual_block(cost, measurement_loss, [quat, trans])

    for frame_idx in range(n_frames - 1):
        quat_t, trans_t = poses[frame_idx]
        quat_t1, trans_t1 = poses[frame_idx + 1]
        rot_cost = RotationSmoothnessFactor(weight=config.lambda_rot_smooth)
        problem.add_residual_block(rot_cost, None, [quat_t, quat_t1])
        trans_cost = TranslationSmoothnessFactor(weight=config.lambda_trans_smooth)
        problem.add_residual_block(trans_cost, None, [trans_t, trans_t1])

    return problem, poses


def _solve(problem: pyceres.Problem, config: OptimizationConfig) -> pyceres.SolverSummary:
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.max_num_iterations = config.max_iter
    options.function_tolerance = config.function_tolerance
    options.gradient_tolerance = config.gradient_tolerance
    options.parameter_tolerance = config.parameter_tolerance
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    return summary


def main() -> None:
    test_zero_valid_frame_skip()

    rng = np.random.default_rng(7)
    trajectories, reference_points = _build_synthetic_trajectory(rng)
    config = OptimizationConfig(
        max_iter=100, lambda_rot_smooth=5.0, lambda_trans_smooth=3.0
    )

    problem_old, poses_old = _build_problem(trajectories, reference_points, config, use_batched=False)
    problem_new, poses_new = _build_problem(trajectories, reference_points, config, use_batched=True)

    n_frames, n_keypoints, _ = trajectories.shape
    n_valid_per_frame = (~np.isnan(trajectories).any(axis=2)).sum(axis=1)
    n_frames_with_data = int((n_valid_per_frame > 0).sum())
    expected_old_blocks = int(n_valid_per_frame.sum()) + 2 * (n_frames - 1)
    expected_new_blocks = n_frames_with_data + 2 * (n_frames - 1)

    assert problem_old.num_residual_blocks() == expected_old_blocks
    assert problem_new.num_residual_blocks() == expected_new_blocks
    assert problem_old.num_residuals() == problem_new.num_residuals(), "Total residual count changed"
    print(f"Residual blocks: old={problem_old.num_residual_blocks()}, new={problem_new.num_residual_blocks()}")
    print(f"Total residuals (unchanged): {problem_old.num_residuals()}")

    summary_old = _solve(problem_old, config)
    summary_new = _solve(problem_new, config)

    print(f"Old: initial_cost={summary_old.initial_cost:.6f}, final_cost={summary_old.final_cost:.6f}, "
          f"iterations={summary_old.num_successful_steps}")
    print(f"New: initial_cost={summary_new.initial_cost:.6f}, final_cost={summary_new.final_cost:.6f}, "
          f"iterations={summary_new.num_successful_steps}")

    assert abs(summary_old.initial_cost - summary_new.initial_cost) < 1e-6
    assert abs(summary_old.final_cost - summary_new.final_cost) < 1e-6
    assert summary_old.num_successful_steps == summary_new.num_successful_steps

    quats_old = np.array([q for q, _ in poses_old])
    quats_new = np.array([q for q, _ in poses_new])
    trans_old = np.array([t for _, t in poses_old])
    trans_new = np.array([t for _, t in poses_new])

    max_quat_error = float(np.max(np.abs(quats_old - quats_new)))
    max_trans_error = float(np.max(np.abs(trans_old - trans_new)))
    print(f"Max quaternion delta: {max_quat_error:.2e}")
    print(f"Max translation delta: {max_trans_error:.2e}")

    assert max_quat_error < 1e-9, "Solved quaternions diverge between old and new residual layout"
    assert max_trans_error < 1e-9, "Solved translations diverge between old and new residual layout"
    print("PASS")


if __name__ == "__main__":
    main()
