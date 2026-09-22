"""
Verify that all resampled/gaze-pipeline outputs for a recording are mutually
synchronized before feeding them into the Rerun viewer.

Checks, for every component that plot_resampled_data.py logs to the shared
"time" timeline:
  - frame count matches common_timestamps.npy
  - start/end timestamp (zeroed) matches common_timestamps.npy
  - file mtime is not older than common_timestamps.npy (a stale output from an
    earlier resampling run is the classic cause of "different parts of the
    recording come in at different times" in the Rerun viewer)

Usage:
    python -m python_code.viz.rerun_viewer.verify_recording_sync <recording_folder>
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

TOLERANCE_S = 1e-6


@dataclass
class ComponentReport:
    name: str
    path: Path | None
    n_frames: int | None = None
    t_start: float | None = None
    t_end: float | None = None
    mtime: float | None = None
    error: str | None = None


def _report_from_tidy_csv(name: str, path: Path | None) -> ComponentReport:
    if path is None or not path.exists():
        return ComponentReport(name=name, path=path, error="missing")
    try:
        df = pl.read_csv(path)
        ts_col = "timestamp_s" if "timestamp_s" in df.columns else "timestamp"
        timestamps = df.select(pl.col(ts_col)).to_series().unique().sort().to_numpy()
        return ComponentReport(
            name=name,
            path=path,
            n_frames=len(timestamps),
            t_start=float(timestamps[0]),
            t_end=float(timestamps[-1]),
            mtime=path.stat().st_mtime,
        )
    except Exception as e:
        return ComponentReport(name=name, path=path, error=str(e))


def _report_from_video(name: str, path: Path | None) -> ComponentReport:
    if path is None or not path.exists():
        return ComponentReport(name=name, path=path, error="missing")
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return ComponentReport(name=name, path=path, error="could not open video")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return ComponentReport(name=name, path=path, n_frames=n_frames, mtime=path.stat().st_mtime)


def verify_recording_sync(recording_folder: RecordingFolder) -> bool:
    """Print a synchronization report. Returns True if everything is consistent."""
    if recording_folder.common_timestamps is None:
        print("No common_timestamps.npy found - recording has not been resampled yet.")
        return False

    common_timestamps = np.load(recording_folder.common_timestamps)
    reference = ComponentReport(
        name="common_timestamps.npy (reference)",
        path=recording_folder.common_timestamps,
        n_frames=len(common_timestamps),
        t_start=float(common_timestamps[0]),
        t_end=float(common_timestamps[-1]),
        mtime=recording_folder.common_timestamps.stat().st_mtime,
    )

    reports = [reference]
    reports.append(_report_from_tidy_csv("skull_kinematics", recording_folder.skull_kinematics_csv))
    reports.append(_report_from_tidy_csv("left_eye_kinematics", recording_folder.left_eye_kinematics_csv))
    reports.append(_report_from_tidy_csv("right_eye_kinematics", recording_folder.right_eye_kinematics_csv))
    reports.append(_report_from_tidy_csv("left_gaze_kinematics", recording_folder.left_gaze_kinematics_csv))
    reports.append(_report_from_tidy_csv("right_gaze_kinematics", recording_folder.right_gaze_kinematics_csv))
    reports.append(_report_from_tidy_csv("skull_and_spine_trajectories", recording_folder.skull_and_spine_resampled_trajectories))
    reports.append(_report_from_tidy_csv("toy_trajectories", recording_folder.toy_resampled_trajectories))
    reports.append(_report_from_video("left_eye_display_video", recording_folder.left_eye_display_video))
    reports.append(_report_from_video("right_eye_display_video", recording_folder.right_eye_display_video))
    reports.append(_report_from_video("topdown_mocap_display_video", recording_folder.topdown_mocap_display_video))

    print("=" * 100)
    print(f"SYNC REPORT: {recording_folder.recording_name}")
    print("=" * 100)
    header = f"{'component':<32}{'n_frames':>10}{'t_start':>12}{'t_end':>12}{'mtime':>22}"
    print(header)
    print("-" * len(header))

    all_ok = True
    for r in reports:
        if r.error:
            print(f"{r.name:<32}{'--':>10}{'--':>12}{'--':>12}   ERROR: {r.error}")
            if r is not reference:
                all_ok = False
            continue
        mtime_str = "" if r.mtime is None else f"{r.mtime:>22.0f}"
        t_start_str = "" if r.t_start is None else f"{r.t_start:>12.3f}"
        t_end_str = "" if r.t_end is None else f"{r.t_end:>12.3f}"
        print(f"{r.name:<32}{r.n_frames:>10}{t_start_str}{t_end_str}{mtime_str}")

    print("-" * len(header))

    for r in reports[1:]:
        if r.error:
            continue
        flags = []
        if r.n_frames != reference.n_frames:
            flags.append(f"n_frames {r.n_frames} != reference {reference.n_frames}")
        if r.t_start is not None and abs(r.t_start - reference.t_start) > TOLERANCE_S:
            flags.append(f"t_start {r.t_start:.3f} != reference {reference.t_start:.3f}")
        if r.t_end is not None and abs(r.t_end - reference.t_end) > TOLERANCE_S:
            flags.append(f"t_end {r.t_end:.3f} != reference {reference.t_end:.3f}")
        if r.mtime is not None and r.mtime < reference.mtime:
            flags.append(
                f"STALE: file is older than common_timestamps.npy "
                f"(regenerate this output - it likely predates the current resampling run)"
            )
        if flags:
            all_ok = False
            print(f"[MISMATCH] {r.name}:")
            for f in flags:
                print(f"    - {f}")

    print("=" * 100)
    print("ALL COMPONENTS SYNCHRONIZED" if all_ok else "SYNCHRONIZATION ISSUES FOUND (see above)")
    print("=" * 100)

    return all_ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m python_code.viz.rerun_viewer.verify_recording_sync <recording_folder>")
        sys.exit(1)

    recording_folder = RecordingFolder.from_folder_path(sys.argv[1])
    ok = verify_recording_sync(recording_folder)
    sys.exit(0 if ok else 1)
