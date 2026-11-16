"""Run jump detection across every solved skull_kinematics.csv under a dataset root,
and write a JSON report so results from different solver versions can be diffed.

Recreates the one-off batch pattern used for the baseline measurement in
solver_improvement_plan.md (never committed), as a reusable script so later
re-measurements after each plan item are actually repeatable and comparable.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.rigid_body_solver.core.jump_detection import detect_orientation_jumps


def find_skull_kinematics_sessions(dataset_root: Path) -> list[Path]:
    """Return every session directory under dataset_root that has a skull_kinematics.csv."""
    return sorted(
        csv_path.parent.parent
        for csv_path in dataset_root.rglob("skull_kinematics/skull_kinematics.csv")
    )


def run_batch_jump_detection(
    dataset_root: Path,
    mad_multiplier: float = 8.0,
    min_discrepancy_deg: float = 2.0,
) -> dict:
    sessions = find_skull_kinematics_sessions(dataset_root)

    per_session: list[dict] = []
    total_flagged = 0
    total_frames = 0
    n_clean = 0
    n_errored = 0

    for session_dir in sessions:
        kinematics_csv = session_dir / "skull_kinematics" / "skull_kinematics.csv"
        reference_geometry_json = session_dir / "skull_kinematics" / "skull_reference_geometry.json"

        try:
            kinematics = RigidBodyKinematics.load_from_disk(
                kinematics_csv_path=kinematics_csv,
                reference_geometry_json_path=reference_geometry_json,
            )
            report = detect_orientation_jumps(
                kinematics, mad_multiplier=mad_multiplier, min_discrepancy_deg=min_discrepancy_deg
            )
        except Exception as e:
            print(f"ERROR processing {session_dir.name}: {e}")
            n_errored += 1
            continue

        n_frames = kinematics.n_frames
        n_flagged = report.n_flagged
        rate_per_1000 = 1000 * n_flagged / n_frames if n_frames else 0.0

        per_session.append(
            {
                "session": session_dir.name,
                "n_frames": n_frames,
                "n_flagged": n_flagged,
                "flagged_rate_per_1000": rate_per_1000,
                "threshold_deg": report.threshold_deg,
                "flagged_frames": report.flagged_frames,
            }
        )

        total_flagged += n_flagged
        total_frames += n_frames
        if n_flagged == 0:
            n_clean += 1

        print(f"{session_dir.name}: {n_flagged} flagged / {n_frames} frames ({rate_per_1000:.2f} per 1000)")

    per_session.sort(key=lambda s: s["flagged_rate_per_1000"], reverse=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "mad_multiplier": mad_multiplier,
        "min_discrepancy_deg": min_discrepancy_deg,
        "n_sessions_found": len(sessions),
        "n_sessions_processed": len(per_session),
        "n_sessions_errored": n_errored,
        "n_clean_sessions": n_clean,
        "total_flagged_frames": total_flagged,
        "total_frames": total_frames,
        "overall_flagged_pct": 100 * total_flagged / total_frames if total_frames else 0.0,
        "per_session": per_session,
    }

    print("\n" + "=" * 80)
    print(f"Sessions found:     {len(sessions)}")
    print(f"Sessions processed: {len(per_session)} ({n_errored} errored)")
    print(f"Clean sessions:     {n_clean}")
    print(f"Total flagged:      {total_flagged} / {total_frames} ({summary['overall_flagged_pct']:.2f}%)")
    print("Worst 5 sessions by rate per 1000 frames:")
    for s in per_session[:5]:
        print(f"  {s['session']}: {s['flagged_rate_per_1000']:.2f}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch-run jump detection across all skull_kinematics.csv files under a dataset root."
    )
    parser.add_argument("dataset_root", type=Path, help="Root directory to search (e.g. analyzable_outputs/)")
    parser.add_argument("--mad-multiplier", type=float, default=8.0)
    parser.add_argument("--min-discrepancy-deg", type=float, default=2.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to save the JSON report (default: jump_detection_report_<timestamp>.json in the cwd)",
    )
    args = parser.parse_args()

    result = run_batch_jump_detection(
        dataset_root=args.dataset_root,
        mad_multiplier=args.mad_multiplier,
        min_discrepancy_deg=args.min_discrepancy_deg,
    )

    output_path = args.output_json
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path(f"jump_detection_report_{timestamp}.json")

    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nReport saved to: {output_path}")
