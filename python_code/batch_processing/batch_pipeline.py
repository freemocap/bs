"""
Run full_pipeline on multiple recordings in sequence.

Each recording can specify its own calibration toml path, or None to let
the pipeline locate it automatically.
"""
from pathlib import Path
import time

from python_code.batch_processing.full_pipeline import full_pipeline


def batch_full_pipeline(
    recordings: list[tuple[Path, Path | None]],
    include_eye: bool = True,
    overwrite_synchronization: bool = False,
    overwrite_calibration: bool = False,
    overwrite_dlc: bool = False,
    overwrite_triangulation: bool = False,
    overwrite_eye_postprocessing: bool = False,
    overwrite_skull_postprocessing: bool = False,
    overwrite_gaze: bool = False,
):
    """
    Run full_pipeline on a list of recordings sequentially.

    Args:
        recordings: List of (recording_folder_path, calibration_toml_path) pairs.
            Pass None as the calibration path to let the pipeline find it automatically.
        include_eye: Whether to include eye processing.
        overwrite_*: Overwrite flags forwarded to full_pipeline for every recording.
    """
    batch_timings: dict[Path, float] = {}
    failures: dict[Path, Exception] = {}
    for recording_folder_path, calibration_toml_path in recordings:
        # Required setup for new sessions (not yet synchronized)
        if "clips" not in str(recording_folder_path) and "full_recording" not in str(recording_folder_path):
            recording_folder_path = recording_folder_path / "full_recording"
        recording_folder_path.mkdir(exist_ok=True, parents=False)
        (recording_folder_path / "mocap_data").mkdir(exist_ok=True, parents=False)
        (recording_folder_path / "eye_data").mkdir(exist_ok=True, parents=False)

        print(f"Processing {recording_folder_path}")
        print(f"\n{'=' * 60}")
        print(f"Processing: {recording_folder_path}")
        t0 = time.perf_counter()
        try:
            full_pipeline(
                recording_folder_path=recording_folder_path,
                calibration_toml_path=calibration_toml_path,
                include_eye=include_eye,
                overwrite_synchronization=overwrite_synchronization,
                overwrite_calibration=overwrite_calibration,
                overwrite_dlc=overwrite_dlc,
                overwrite_triangulation=overwrite_triangulation,
                overwrite_eye_postprocessing=overwrite_eye_postprocessing,
                overwrite_skull_postprocessing=overwrite_skull_postprocessing,
                overwrite_gaze=overwrite_gaze,
            )
        except Exception as e:
            failures[recording_folder_path] = e
            print(f"ERROR processing {recording_folder_path}: {e}")
        batch_timings[recording_folder_path] = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print("=== Batch Summary ===")
    for path, elapsed in batch_timings.items():
        status = "FAILED" if path in failures else "OK"
        print(f"  [{status}] {path.name}: {elapsed:.1f}s")
    print("  ---")
    print(f"  Total: {sum(batch_timings.values()):.1f}s")
    if failures:
        print(f"\n=== Failures ({len(failures)}) ===")
        for path, error in failures.items():
            print(f"  {path}:")
            print(f"    {type(error).__name__}: {error}")


if __name__ == "__main__":
    recordings: list[tuple[Path, Path | None]] = [
        # (recording_folder_path, calibration_toml_path or None)
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-06-28_ferret_753_EyeCameras_P30_EO2"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-06-28_ferret_757_EyeCameras_P30_EO2"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-06-29_ferret_753_EyeCameras_P31_EO3"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-06-29_ferret_757_EyeCameras_P31_EO3__1"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_753_EyeCameras_P33_EO5"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_757_EyeCameras_P33_EO5__2"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_757_EyeCameras_P33_EO5"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-07-03_ferret_753_EyeCameras_P35_EO7"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-07-05_ferret_753_EyeCameras_P37_EO9"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-07-05_ferret_757_EyeCameras_P37_EO9"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-07-07_ferret_753_EyeCameras_P39_E11"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-07-07_ferret_757_EyeCameras_P39_E11"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-07-09_ferret_753_EyeCameras_P41_E13"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-07-09_ferret_757_EyeCameras_P41_E13"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-10-11_ferret_402_E02"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-10-11_ferret_420_E02"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-10-12_ferret_402_E03"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-10-12_ferret_420_E03"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-10-13_ferret_402_E04"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-10-13_ferret_420_E04"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2025-10-14_ferret_402_E05"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-10-14_ferret_420_E05"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2025-10-21_ferret_420_E012"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2026-02-28_ferret_405_EO0"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2026-02-28_ferret_407_EO0"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2026-03-09_ferret_407_EO9"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2026-03-14_ferret_407_P47_E14"), None),
        # (Path("/home/scholl-lab/ferret_recordings/session_2026-03-16_ferret_403_P49_E7"), None),
    ]


    batch_full_pipeline(
        recordings=recordings,
        overwrite_synchronization=False,
        overwrite_calibration=False,
        overwrite_dlc=False,
        overwrite_triangulation=False,
        overwrite_eye_postprocessing=True,
        overwrite_skull_postprocessing=False,
        overwrite_gaze=False,
    )
