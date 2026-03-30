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
        batch_timings[recording_folder_path] = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print("=== Batch Summary ===")
    for path, elapsed in batch_timings.items():
        print(f"  {path.name}: {elapsed:.1f}s")
    print("  ---")
    print(f"  Total: {sum(batch_timings.values()):.1f}s")


if __name__ == "__main__":
    recordings: list[tuple[Path, Path | None]] = [
        # (recording_folder_path, calibration_toml_path or None)
        (Path("/home/scholl-lab/ferret_recordings/session_2026-03-16_ferret_411_P49_E8"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2026-03-10_ferret_411_P43_E2"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2026-03-02_ferret_405_EO2"), None),
        (Path("/home/scholl-lab/ferret_recordings/session_2026-03-09_ferret_407_EO9"), None),
    ]


    batch_full_pipeline(
        recordings=recordings,
        overwrite_synchronization=False,
        overwrite_calibration=False,
        overwrite_dlc=False,
        overwrite_triangulation=False,
        overwrite_eye_postprocessing=False,
        overwrite_skull_postprocessing=False,
        overwrite_gaze=False,
    )
