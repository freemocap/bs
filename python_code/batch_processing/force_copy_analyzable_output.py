"""
Force-copy analyzable_output folders to Dropbox.

Walks a directory of ferret recordings (structured as
`<recording_name>/full_recording/analyzable_output`) and copies each
analyzable_output folder to Dropbox, regardless of whether the normal
gaze pipeline's Dropbox copy step already ran or succeeded.

Useful when the automatic copy at the end of run_gaze_pipeline.py didn't
work (e.g. Dropbox wasn't running, network hiccup) and you want to
manually force the move without re-running the whole pipeline.

Usage:
    python python_code/batch_processing/force_copy_analyzable_output.py
"""
import logging
from pathlib import Path

from python_code.ferret_gaze.run_gaze_pipeline import copy_analyzable_output
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def force_copy_analyzable_output(
    ferret_recordings_path: Path,
    destination: Path = Path("/home/scholl-lab/Dropbox/projects/VisBehavDev/data/analyzable_outputs"),
) -> None:
    for subdir in sorted(ferret_recordings_path.iterdir()):
        if not subdir.is_dir():
            continue

        full_recording_path = subdir / "full_recording"
        if not full_recording_path.exists():
            continue

        try:
            recording_folder = RecordingFolder.from_folder_path(full_recording_path)
        except ValueError as e:
            logger.warning(f"Skipping {full_recording_path}: {e}")
            continue

        if recording_folder.analyzable_output is None:
            logger.info(f"No analyzable_output for {recording_folder.recording_name} — skipping")
            continue

        copy_analyzable_output(recording_folder, destination=destination)


if __name__ == "__main__":
    ferret_recordings_path = Path("/home/scholl-lab/ferret_recordings")
    force_copy_analyzable_output(ferret_recordings_path)
