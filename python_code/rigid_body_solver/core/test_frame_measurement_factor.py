"""Smoke test: verify FrameMeasurementFactor (one batched residual block per frame)
is bit-for-bit equivalent to n_valid separate MeasurementFactor residual blocks over
the same points/weight, sharing the same [quat, trans] parameters.

Both sides use the same analytic residual/Jacobian code (_quaternion_point_jacobian),
so this checks stacking/ordering correctness introduced by batching, not the
underlying math (already validated by test_measurement_jacobian.py). Tolerance is
tight (near machine precision), not the loose finite-difference tolerance used
elsewhere.
"""

import numpy as np

from python_code.rigid_body_solver.core.optimization import (
    FrameMeasurementFactor,
    MeasurementFactor,
)


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    return quat / np.linalg.norm(quat)


def _evaluate_frame_factor(
    quat: np.ndarray, trans: np.ndarray, measured_points: np.ndarray, reference_points: np.ndarray, weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_valid = measured_points.shape[0]
    factor = FrameMeasurementFactor(
        measured_points=measured_points, reference_points=reference_points, weight=weight
    )
    residuals = np.zeros(3 * n_valid, dtype=np.float64)
    quat_jac = np.zeros(3 * n_valid * 4, dtype=np.float64)
    trans_jac = np.zeros(3 * n_valid * 3, dtype=np.float64)
    factor.Evaluate([quat, trans], residuals, [quat_jac, trans_jac])
    return residuals, quat_jac.reshape(3 * n_valid, 4), trans_jac.reshape(3 * n_valid, 3)


def _evaluate_stacked_single_factors(
    quat: np.ndarray, trans: np.ndarray, measured_points: np.ndarray, reference_points: np.ndarray, weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_valid = measured_points.shape[0]
    residuals = np.zeros(3 * n_valid, dtype=np.float64)
    quat_jac = np.zeros((3 * n_valid, 4), dtype=np.float64)
    trans_jac = np.zeros((3 * n_valid, 3), dtype=np.float64)
    for i in range(n_valid):
        factor = MeasurementFactor(
            measured_point=measured_points[i], reference_point=reference_points[i], weight=weight
        )
        r = np.zeros(3, dtype=np.float64)
        qj = np.zeros(3 * 4, dtype=np.float64)
        tj = np.zeros(3 * 3, dtype=np.float64)
        factor.Evaluate([quat, trans], r, [qj, tj])
        residuals[3 * i : 3 * i + 3] = r
        quat_jac[3 * i : 3 * i + 3, :] = qj.reshape(3, 4)
        trans_jac[3 * i : 3 * i + 3, :] = tj.reshape(3, 3)
    return residuals, quat_jac, trans_jac


def _check_case(rng: np.random.Generator, n_valid: int, weight: float = 100.0) -> float:
    quat = _random_quat(rng)
    trans = rng.normal(size=3) * 20.0
    measured_points = rng.normal(size=(n_valid, 3)) * 50.0
    reference_points = rng.normal(size=(n_valid, 3)) * 50.0

    frame_residuals, frame_quat_jac, frame_trans_jac = _evaluate_frame_factor(
        quat, trans, measured_points, reference_points, weight
    )
    stacked_residuals, stacked_quat_jac, stacked_trans_jac = _evaluate_stacked_single_factors(
        quat, trans, measured_points, reference_points, weight
    )

    max_error = max(
        float(np.max(np.abs(frame_residuals - stacked_residuals))),
        float(np.max(np.abs(frame_quat_jac - stacked_quat_jac))),
        float(np.max(np.abs(frame_trans_jac - stacked_trans_jac))),
    )
    return max_error


def main() -> None:
    rng = np.random.default_rng(42)
    max_error = 0.0

    # Fixed edge cases: single point, two points, and a larger "all keypoints valid" case.
    for n_valid in (1, 2, 8):
        max_error = max(max_error, _check_case(rng, n_valid))

    # Randomized trials over varying n_valid.
    n_trials = 500
    for _ in range(n_trials):
        n_valid = int(rng.integers(1, 12))
        max_error = max(max_error, _check_case(rng, n_valid))

    print(f"Trials: {n_trials + 3}")
    print(f"Max abs error (FrameMeasurementFactor vs. stacked MeasurementFactors): {max_error:.2e}")
    assert max_error < 1e-10, "FrameMeasurementFactor diverges from stacked per-keypoint reference"
    print("PASS")


if __name__ == "__main__":
    main()
