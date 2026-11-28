"""Smoke test: verify the analytic quaternion Jacobian in MeasurementFactor matches the
old finite-difference implementation it replaced, across random quaternions/points.

Closed form derived via sympy (differentiating R(q/|q|) @ p w.r.t. raw [w,x,y,z], then
evaluated at |q|=1 since QuaternionManifold keeps the parameter block on the unit
sphere) and validated there against central differences to ~1e-7 before transcription.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from python_code.rigid_body_solver.core.optimization import _quaternion_point_jacobian


def _finite_difference_jacobian(
    quat: np.ndarray, point: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Reference implementation: the forward-difference Jacobian MeasurementFactor.Evaluate
    used before the analytic replacement (same normalize-then-rotate semantics)."""
    base_R = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    base_predicted = base_R @ point

    jacobian = np.zeros((3, 4))
    for j in range(4):
        quat_plus = quat.copy()
        quat_plus[j] += eps
        quat_plus = quat_plus / np.linalg.norm(quat_plus)
        R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
        predicted_plus = R_plus @ point
        jacobian[:, j] = (predicted_plus - base_predicted) / eps
    return jacobian


def main() -> None:
    rng = np.random.default_rng(42)
    n_trials = 2000
    max_error = 0.0

    for _ in range(n_trials):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        point = rng.normal(size=3) * 50.0

        analytic = _quaternion_point_jacobian(quat, point)
        finite_difference = _finite_difference_jacobian(quat, point)

        max_error = max(max_error, float(np.max(np.abs(analytic - finite_difference))))

    print(f"Trials: {n_trials}")
    print(f"Max abs error (analytic vs. finite-difference): {max_error:.2e}")
    # The forward-difference reference itself has ~4e-4 truncation error at eps=1e-6
    # (confirmed separately against central differences, where the analytic Jacobian
    # matches to ~1e-7) on points scaled to ~50mm, so the tolerance here is set to catch
    # a wrong formula (which would be off by orders of magnitude), not to be as tight as
    # the analytic Jacobian actually is.
    assert max_error < 1e-3, "Analytic Jacobian diverges from finite-difference reference"
    print("PASS")


if __name__ == "__main__":
    main()
