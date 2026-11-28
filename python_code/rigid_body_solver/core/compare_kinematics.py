"""Compare two solved RigidBodyKinematics outputs (e.g. before/after a solver change)
frame-by-frame, beyond just jump-detection counts: body-frame position and orientation
deltas, plus the per-keypoint reconstructed-marker deltas those imply.

Jump detection only tells you whether a frame crossed a hard outlier threshold. This
tells you how much every frame moved, which is what you actually want to know when
validating a change that's supposed to be a no-op (or a small, bounded improvement).
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics


def quaternion_angle_deg(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def _percentiles(values: NDArray[np.float64]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def compare_kinematics(before: RigidBodyKinematics, after: RigidBodyKinematics) -> dict:
    if before.n_frames != after.n_frames:
        raise ValueError(
            f"Frame count mismatch ({before.n_frames} vs {after.n_frames}) — "
            "these aren't the same recording/input data, comparison would be meaningless."
        )

    position_delta_mm = np.linalg.norm(after.position_xyz - before.position_xyz, axis=1)
    orientation_delta_deg = quaternion_angle_deg(before.quaternions_wxyz, after.quaternions_wxyz)

    before_markers = before.keypoint_trajectories.trajectories_fr_id_xyz
    after_markers = after.keypoint_trajectories.trajectories_fr_id_xyz
    marker_delta_mm = np.linalg.norm(after_markers - before_markers, axis=2)  # (n_frames, n_keypoints)

    worst_frames = np.argsort(position_delta_mm + orientation_delta_deg)[::-1][:10]

    summary = {
        "n_frames": before.n_frames,
        "position_delta_mm": _percentiles(position_delta_mm),
        "orientation_delta_deg": _percentiles(orientation_delta_deg),
        "per_marker_delta_mm": {
            name: _percentiles(marker_delta_mm[:, i])
            for i, name in enumerate(before.keypoint_trajectories.keypoint_names)
        },
        "worst_frames": [
            {
                "frame": int(f),
                "timestamp": float(before.timestamps[f]),
                "position_delta_mm": float(position_delta_mm[f]),
                "orientation_delta_deg": float(orientation_delta_deg[f]),
            }
            for f in worst_frames
        ],
    }
    return summary


def print_summary(summary: dict) -> None:
    print(f"Frames compared: {summary['n_frames']}")
    print("\nBody-frame position delta (mm):")
    for k, v in summary["position_delta_mm"].items():
        print(f"  {k:8s}: {v:.4f}")
    print("\nBody-frame orientation delta (deg):")
    for k, v in summary["orientation_delta_deg"].items():
        print(f"  {k:8s}: {v:.4f}")
    print("\nPer-marker reconstructed position delta (mm, mean/max):")
    for name, stats in summary["per_marker_delta_mm"].items():
        print(f"  {name:16s}: mean={stats['mean']:.4f}  max={stats['max']:.4f}")
    print("\nWorst 10 frames (position + orientation delta):")
    for w in summary["worst_frames"]:
        print(
            f"  frame {w['frame']:6d}  t={w['timestamp']:.3f}s  "
            f"pos_delta={w['position_delta_mm']:.3f}mm  orient_delta={w['orientation_delta_deg']:.3f}deg"
        )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compare two solved skull_kinematics outputs frame-by-frame."
    )
    parser.add_argument("before_kinematics_csv", type=Path)
    parser.add_argument("before_reference_geometry_json", type=Path)
    parser.add_argument("after_kinematics_csv", type=Path)
    parser.add_argument("after_reference_geometry_json", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    before_kin = RigidBodyKinematics.load_from_disk(
        kinematics_csv_path=args.before_kinematics_csv,
        reference_geometry_json_path=args.before_reference_geometry_json,
    )
    after_kin = RigidBodyKinematics.load_from_disk(
        kinematics_csv_path=args.after_kinematics_csv,
        reference_geometry_json_path=args.after_reference_geometry_json,
    )

    result = compare_kinematics(before_kin, after_kin)
    print_summary(result)

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2))
        print(f"\nSaved to: {args.output_json}")
