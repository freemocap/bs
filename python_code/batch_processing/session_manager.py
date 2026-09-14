"""
YAML-backed registry of every known ferret recording session.

Sessions live in `sessions.yaml` with their descriptors (animal id, date,
day label) already parsed out, so `SessionManager` never has to re-parse a
folder name to answer a query. `verify()` re-derives those descriptors from
the folder name and flags any mismatch against what's stored, so the YAML
stays honest.

`not_processed_through` resolves entries against real recordings on disk
(via `RecordingFolder`), so it only produces meaningful results when run on
a machine where `base_recordings_root` actually exists (e.g. the Scholl Lab
machine) — everything else here works on the YAML alone.
"""
import re
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from python_code.utilities.folder_utilities.recording_folder import PipelineStep, RecordingFolder


DEFAULT_BASE_RECORDINGS_ROOT = Path("/home/scholl-lab/ferret_recordings")
DEFAULT_SESSIONS_YAML_PATH = Path(__file__).parent / "sessions.yaml"

# Matches e.g.:
#   session_2025-10-15_ferret_402_E06
#   session_2025-07-11_ferret_757_EyeCamera_P43_E15__1
#   session_2025-06-28_ferret_753_EyeCameras_P30_EO2
#   session_2026-03-14_ferret_407_P47_E14
SESSION_NAME_PATTERN = re.compile(
    r"^session_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_"
    r"ferret_(?P<animal_id>\d+)_"
    r"(?:Eye[Cc]ameras?_)?"
    r"(?:P\d+_)?"
    r"(?P<day_label>(?:EO|E)\d+)"
    r"(?:__\d+)?$"
)

# Maps a PipelineStep to the RecordingFolder predicate that checks whether
# a recording has reached (at least) that step.
_STEP_CHECKS = {
    PipelineStep.SYNCHRONIZED: RecordingFolder.is_synchronized,
    PipelineStep.DLCED: RecordingFolder.is_dlc_processed,
    PipelineStep.TRIANGULATED: RecordingFolder.is_triangulated,
    PipelineStep.EYE_POST_PROCESSED: RecordingFolder.is_eye_postprocessed,
    PipelineStep.SKULL_POST_PROCESSED: RecordingFolder.is_skull_postprocessed,
    PipelineStep.GAZE_POST_PROCESSED: RecordingFolder.is_gaze_postprocessed,
}


def parse_session_name(name: str) -> dict:
    """
    Parse a recording folder name into its animal_id/date/day_label descriptors.

    Raises ValueError if the name doesn't match the expected session naming pattern.
    """
    match = SESSION_NAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Could not parse session name: {name}")
    return {
        "animal_id": match.group("animal_id"),
        "date": date.fromisoformat(match.group("date")),
        "day_label": match.group("day_label"),
    }


class SessionEntry(BaseModel):
    name: str
    animal_id: str
    date: date
    day_label: str
    calibration_toml_path: Path | None = None
    notes: str | None = None


class SessionManager:
    def __init__(
        self,
        yaml_path: Path = DEFAULT_SESSIONS_YAML_PATH,
        base_recordings_root: Path = DEFAULT_BASE_RECORDINGS_ROOT,
    ):
        self.yaml_path = Path(yaml_path)
        self.base_recordings_root = base_recordings_root
        self.entries: list[SessionEntry] = self._load()

    def _load(self) -> list[SessionEntry]:
        with open(self.yaml_path) as f:
            raw_entries = yaml.safe_load(f) or []
        return [SessionEntry(**raw_entry) for raw_entry in raw_entries]

    def all(self) -> list[SessionEntry]:
        return list(self.entries)

    def by_animal(self, animal_id: str) -> list[SessionEntry]:
        return [entry for entry in self.entries if entry.animal_id == animal_id]

    def by_date(self, target: date) -> list[SessionEntry]:
        return [entry for entry in self.entries if entry.date == target]

    def by_date_range(self, start: date, end: date) -> list[SessionEntry]:
        return [entry for entry in self.entries if start <= entry.date <= end]

    def by_day_label(self, label: str, exact: bool = True) -> list[SessionEntry]:
        if exact:
            return [entry for entry in self.entries if entry.day_label == label]
        return [entry for entry in self.entries if entry.day_label.startswith(label)]

    def recording_folder_path(self, entry: SessionEntry) -> Path:
        return self.base_recordings_root / entry.name / "full_recording"

    def to_recordings(self, entries: list[SessionEntry]) -> list[tuple[Path, Path | None]]:
        return [
            (self.recording_folder_path(entry), entry.calibration_toml_path)
            for entry in entries
        ]

    def not_processed_through(
        self,
        step: PipelineStep,
        entries: list[SessionEntry] | None = None,
    ) -> list[tuple[Path, Path | None]]:
        """
        Return (recording_folder_path, calibration_toml_path) pairs — ready to
        pass straight to batch_full_pipeline/full_pipeline — for every entry
        whose recording on disk has not (yet) completed the given pipeline
        step. Entries whose recording folder doesn't exist on disk are skipped
        and reported, not raised.
        """
        is_step_complete = _STEP_CHECKS.get(step)
        if is_step_complete is None:
            raise ValueError(f"No processing check available for step: {step}")

        pending = []
        for entry in entries if entries is not None else self.entries:
            folder_path = self.recording_folder_path(entry)
            if not folder_path.exists():
                print(f"Skipping {entry.name}: {folder_path} does not exist")
                continue
            recording_folder = RecordingFolder.from_folder_path(folder_path)
            if not is_step_complete(recording_folder):
                pending.append((folder_path, entry.calibration_toml_path))
        return pending

    def verify(self) -> list[str]:
        """Cross-check stored descriptors against a fresh parse of each entry's name."""
        issues = []
        for entry in self.entries:
            try:
                parsed = parse_session_name(entry.name)
            except ValueError as e:
                issues.append(str(e))
                continue
            for field, parsed_value in parsed.items():
                stored_value = getattr(entry, field)
                if stored_value != parsed_value:
                    issues.append(
                        f"{entry.name}: stored {field}={stored_value!r} "
                        f"but name parses to {field}={parsed_value!r}"
                    )
        return issues


if __name__ == "__main__":
    session_manager = SessionManager()

    issues = session_manager.verify()
    if issues:
        print(f"=== {len(issues)} verification issue(s) ===")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"All {len(session_manager.all())} sessions verified OK")
