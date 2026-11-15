"""
Orientation Jump Detection
===========================

Detects frames where the rigid body solver's output orientation looks like
it converged to a different local minimum than its neighbors ("jumps"),
as opposed to frames where the body is genuinely rotating fast.

The key metric is the quaternion "triangle discrepancy":

    discrepancy[t] = angle(q[t-1], q[t]) + angle(q[t], q[t+1]) - angle(q[t-1], q[t+1])

By the triangle inequality for geodesic distance on SO(3), this is always
>= 0. If frame t lies roughly on the geodesic path between its neighbors
(i.e. the body moved smoothly through it, however fast), the discrepancy is
small. If frame t is an outlier -- the solver jumped away from and then back
toward the neighboring frames' basin -- angle(t-1,t) and angle(t,t+1) are both
large while angle(t-1,t+1) stays small, so the discrepancy spikes. This makes
it a detector of *solver glitches*, not just *fast motion*.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics


def quaternion_angle_deg(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Geodesic angle in degrees between corresponding rows of two (N, 4) wxyz quaternion arrays."""
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def compute_triangle_discrepancy_deg(quaternions_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Per-frame triangle-inequality residual (degrees), NaN at the first/last frame.

    Flags a frame whose orientation deviates from the geodesic path between its
    immediate neighbors, independent of how fast the body is actually rotating.
    """
    n = quaternions_wxyz.shape[0]
    discrepancy = np.full(n, np.nan, dtype=np.float64)
    if n < 3:
        return discrepancy

    angle_prev = quaternion_angle_deg(quaternions_wxyz[:-2], quaternions_wxyz[1:-1])
    angle_next = quaternion_angle_deg(quaternions_wxyz[1:-1], quaternions_wxyz[2:])
    angle_skip = quaternion_angle_deg(quaternions_wxyz[:-2], quaternions_wxyz[2:])

    discrepancy[1:-1] = angle_prev + angle_next - angle_skip
    return discrepancy


class JumpReport(BaseModel):
    """Result of running jump detection on one rigid body's solved kinematics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rigid_body_name: str
    discrepancy_deg: NDArray[np.float64]  # (N,), NaN at endpoints
    threshold_deg: float
    flagged_frames: list[int]
    flagged_timestamps: list[float]
    flagged_discrepancies_deg: list[float]

    @property
    def n_flagged(self) -> int:
        return len(self.flagged_frames)


def detect_orientation_jumps(
    kinematics: RigidBodyKinematics,
    mad_multiplier: float = 8.0,
    min_discrepancy_deg: float = 2.0,
) -> JumpReport:
    """
    Flag frames where solved orientation jumps away from and back toward its
    neighbors, using a robust (median/MAD) threshold on the triangle discrepancy.

    Args:
        kinematics: solved RigidBodyKinematics to inspect.
        mad_multiplier: flag frames above median + mad_multiplier * robust_std.
            Robust std = 1.4826 * MAD, matching the thresholding convention
            already used in ferret_eye_kinematics_functions.py.
        min_discrepancy_deg: floor on the threshold so near-static recordings
            (tiny MAD) don't flag ordinary noise.

    Returns:
        JumpReport with the full discrepancy trace and the flagged frames.
    """
    discrepancy = compute_triangle_discrepancy_deg(kinematics.quaternions_wxyz)
    valid = discrepancy[~np.isnan(discrepancy)]

    median = float(np.median(valid)) if len(valid) else 0.0
    robust_std = float(1.4826 * np.median(np.abs(valid - median))) if len(valid) else 0.0
    threshold = max(median + mad_multiplier * robust_std, min_discrepancy_deg)

    flagged_mask = np.nan_to_num(discrepancy, nan=-np.inf) > threshold
    flagged_frames = np.nonzero(flagged_mask)[0].tolist()

    return JumpReport(
        rigid_body_name=kinematics.name,
        discrepancy_deg=discrepancy,
        threshold_deg=threshold,
        flagged_frames=flagged_frames,
        flagged_timestamps=[float(kinematics.timestamps[i]) for i in flagged_frames],
        flagged_discrepancies_deg=[float(discrepancy[i]) for i in flagged_frames],
    )


def plot_jump_report(report: JumpReport, kinematics: RigidBodyKinematics, output_path: str | Path) -> None:
    """Save a diagnostic plot of discrepancy and angular speed with flagged frames marked."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(kinematics.timestamps, report.discrepancy_deg, color="tab:blue", linewidth=0.8)
    axes[0].axhline(report.threshold_deg, color="tab:red", linestyle="--", label=f"threshold ({report.threshold_deg:.2f} deg)")
    if report.flagged_frames:
        axes[0].scatter(report.flagged_timestamps, report.flagged_discrepancies_deg, color="tab:red", zorder=5, label="flagged")
    axes[0].set_ylabel("triangle discrepancy (deg)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{report.rigid_body_name}: orientation jump detection ({report.n_flagged} flagged / {kinematics.n_frames} frames)")

    angular_speed = kinematics.angular_speed
    axes[1].plot(angular_speed.timestamps, np.degrees(angular_speed.values), color="tab:gray", linewidth=0.8)
    if report.flagged_frames:
        axes[1].scatter(
            report.flagged_timestamps,
            np.degrees(angular_speed.values[report.flagged_frames]),
            color="tab:red",
            zorder=5,
        )
    axes[1].set_ylabel("angular speed (deg/s)")
    axes[1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect orientation jumps in solved rigid body kinematics.")
    parser.add_argument("kinematics_csv", type=Path, help="Path to {name}_kinematics.csv")
    parser.add_argument("reference_geometry_json", type=Path, help="Path to {name}_reference_geometry.json")
    parser.add_argument("--mad-multiplier", type=float, default=8.0)
    parser.add_argument("--plot-output", type=Path, default=None)
    args = parser.parse_args()

    kin = RigidBodyKinematics.load_from_disk(args.kinematics_csv, args.reference_geometry_json)
    report = detect_orientation_jumps(kin, mad_multiplier=args.mad_multiplier)

    print(f"{report.rigid_body_name}: {report.n_flagged} flagged frames out of {kin.n_frames}")
    print(f"threshold: {report.threshold_deg:.3f} deg")
    for frame, ts, disc in zip(report.flagged_frames, report.flagged_timestamps, report.flagged_discrepancies_deg):
        print(f"  frame {frame:6d}  t={ts:.3f}s  discrepancy={disc:.2f} deg")

    if args.plot_output is not None:
        plot_jump_report(report, kin, args.plot_output)
        print(f"plot saved to {args.plot_output}")
