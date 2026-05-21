from pathlib import Path
import json
import subprocess
import os
import sys
import time

from python_code.batch_processing.check_progress.check_progress import _read_dlc_iteration
from python_code.batch_processing.full_pipeline import _dlc_metadata_is_outdated, run_calibration_subprocess, run_skellyclicker_subprocess, run_triangulation_subprocess
from python_code.batch_processing.postprocess_recording import process_recording
from python_code.cameras.postprocess import postprocess
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.utilities.processing_metadata import write_step_metadata
from python_code.validation.create_validation_analyzable_output import create_validation_analyzable_output


VALIDATION_DLC_ITERATION = 4


def validation_pipeline(
    recording_folder_path: Path,
    calibration_toml_path: Path | None = None,
    run_synchronization: bool = False,
    run_calibration: bool = False,
    run_dlc: bool = False,
    run_triangulation: bool = False,
    run_skull_postprocessing: bool = False,
):
    recording_folder = RecordingFolder.from_folder_path(folder=recording_folder_path)
    timings: dict[str, float | None] = {}

    # Propagate overwrite flags through dependent steps
    if run_synchronization:
        run_calibration = True

    # Synchronization
    if run_synchronization:
        print(f"Synchronizing videos at {recording_folder.base_recordings_folder}")
        t0 = time.perf_counter()
        postprocess(session_folder_path=recording_folder.base_recordings_folder, include_eyes=False)
        timings["Synchronization"] = time.perf_counter() - t0
        print(f"Synchronization complete ({timings['Synchronization']:.1f}s)")
    else:
        timings["Synchronization"] = None

    if timings["Synchronization"] is not None:
        write_step_metadata(
            recording_folder.processing_metadata_path,
            step="synchronization",
            parameters={"include_eyes": False},
        )

    # Calibration
    if run_calibration:
        print("Calibrating session...")
        t0 = time.perf_counter()
        calibration_videos_path = recording_folder_path.parent.parent / "calibration" / "synchronized_corrected_videos"
        run_calibration_subprocess(calibration_videos_path=calibration_videos_path)
        timings["Calibration"] = time.perf_counter() - t0
        print(f"Calibration complete ({timings['Calibration']:.1f}s)")
    else:
        timings["Calibration"] = None

    if timings["Calibration"] is not None:
        write_step_metadata(
            recording_folder.processing_metadata_path,
            step="calibration",
            parameters={
                "venv_path": "/home/scholl-lab/anaconda3/envs/fmc/bin/python",
                "script_path": "/home/scholl-lab/Documents/git_repos/freemocap/experimental/batch_process/headless_calibration.py",
            },
        )

    # DLC — check each model independently
    dlc_output = recording_folder.mocap_data / "dlc_output" / "error_measurement_model"
    run_dlc_head = run_dlc or _dlc_metadata_is_outdated(dlc_output, VALIDATION_DLC_ITERATION)

    if not run_dlc:
        if run_dlc_head:
            print("Body DLC outputs are from an outdated model iteration, forcing body DLC reprocessing")

    if run_dlc_head:
        print("Running pose estimation...")
        processing_script = "/home/scholl-lab/skellyclicker/skellyclicker/scripts/process_folder.py"
        t0 = time.perf_counter()
        run_skellyclicker_subprocess(
            recording_folder_path=recording_folder_path,
            script_path=processing_script,
        )
        timings["Pose estimation"] = time.perf_counter() - t0
        print(f"Pose estimation complete ({timings['Pose estimation']:.1f}s)")
    else:
        timings["Pose estimation"] = None
        print("Pose estimation: skipped")

    if timings["Pose estimation"] is not None:
        write_step_metadata(
            recording_folder.processing_metadata_path,
            step="pose_estimation",
            parameters={
                "script_path": processing_script,
            },
            extra={
                "dlc_iterations": {
                    "body": _read_dlc_iteration(dlc_output),
                }
            },
        )

    # Propagate DLC results to downstream steps

    if run_dlc_head and run_calibration:
        run_triangulation = True
    if run_triangulation:
        run_skull_postprocessing = True

    # Triangulation
    if run_triangulation:
        if calibration_toml_path is None:
            calibration_toml_path = recording_folder_path.parent.parent / "calibration" / "session_2025-07-11_camera_calibration.toml"
        if calibration_toml_path is None or not calibration_toml_path.exists():
            raise ValueError("No calibration toml file found, could not run triangulation")
        print("Running triangulation...")
        t0 = time.perf_counter()
        run_triangulation_subprocess(recording_folder_path=recording_folder_path, calibration_toml_path=calibration_toml_path, validation=True)
        timings["Triangulation"] = time.perf_counter() - t0
        print(f"Triangulation complete ({timings['Triangulation']:.1f}s)")
    else:
        timings["Triangulation"] = None

    if timings["Triangulation"] is not None:
        write_step_metadata(
            recording_folder.processing_metadata_path,
            step="triangulation",
            parameters={
                "skip_toy": True,
                "venv_path": "/home/scholl-lab/Documents/git_repos/dlc_to_3d/.venv/bin/python",
                "script_path": "/home/scholl-lab/Documents/git_repos/dlc_to_3d/dlc_reconstruction/dlc_to_3d.py",
            },
        )

    if run_skull_postprocessing:
        print("Running skull processing...")
        t0 = time.perf_counter()
        process_recording(
            recording_folder=recording_folder,
            skip_eye=True,
            skip_skull=not run_skull_postprocessing,
            skip_gaze=True,
            validate=False,
            visualize=False
        )
        timings["Gaze processing"] = time.perf_counter() - t0
        print(f"Gaze processing complete ({timings['Gaze processing']:.1f}s)")
    else:
        timings["Gaze processing"] = None

    create_validation_analyzable_output(recording_folder)

    print(f"\nSession processed: {recording_folder_path}")
    print("\n=== Pipeline Timing Summary ===")
    total = 0.0
    for step, elapsed in timings.items():
        if elapsed is None:
            print(f"  {step}: skipped")
        else:
            print(f"  {step}: {elapsed:.1f}s")
            total += elapsed
    print(f"  ---")
    print(f"  Total: {total:.1f}s")




if __name__=="__main__":
    recording_folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2026-05-04_error_measurements/exp1_0.5hz_yaw_05-04-26"
    )

    if "clips" not in str(recording_folder_path) and "full_recording" not in str(recording_folder_path):
        recording_folder_path = recording_folder_path / "full_recording"


    recording_folder_path.mkdir(exist_ok=True, parents=False)
    (recording_folder_path / "mocap_data").mkdir(exist_ok=True, parents=False)
    (recording_folder_path / "eye_data").mkdir(exist_ok=True, parents=False)
    print(f"Processing {recording_folder_path}")

    validation_pipeline(
        recording_folder_path=recording_folder_path,
        run_synchronization=True,
        run_calibration=False,
        run_dlc=True,
        run_triangulation=True,
        run_skull_postprocessing=True,
    )
