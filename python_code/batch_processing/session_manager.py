"""
YAML-backed registry of every known ferret recording session.

Sessions live in `sessions.yaml` with their descriptors (animal id, date,
postnatal day, day label) already parsed out, so `SessionManager` never has
to re-parse a folder name to answer a query. `verify()` re-derives those
descriptors from the folder name and flags any mismatch against what's
stored, so the YAML stays honest.

Querying goes through `SessionManager.query()`, which returns a chainable
`SessionQuery` — filters combine with AND by chaining calls:

    session_manager.query().animal_prefix("7").age_range(30, 45).names()
    session_manager.query().exclude_animal("753", "757").recordings()
    session_manager.query().animal("407").pending(PipelineStep.GAZE_POST_PROCESSED).recordings()

`pending`/`done` resolve entries against real recordings on disk (via
`RecordingFolder`), so they only produce meaningful results when run on a
machine where `base_recordings_root` actually exists (e.g. the Scholl Lab
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
    r"(?:P(?P<postnatal_day>\d+)_)?"
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


_DAY_LABEL_PATTERN = re.compile(r"^(?:EO|E)(?P<number>\d+)$")


def _day_label_number(day_label: str) -> int:
    """The numeric part of a day_label, ignoring the E/EO prefix — E and EO denote the same day count."""
    match = _DAY_LABEL_PATTERN.match(day_label)
    if match is None:
        raise ValueError(f"Could not parse day_label: {day_label}")
    return int(match.group("number"))


def parse_session_name(name: str) -> dict:
    """
    Parse a recording folder name into its animal_id/date/postnatal_day/day_label descriptors.

    Raises ValueError if the name doesn't match the expected session naming pattern.
    """
    match = SESSION_NAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Could not parse session name: {name}")
    postnatal_day = match.group("postnatal_day")
    return {
        "animal_id": match.group("animal_id"),
        "date": date.fromisoformat(match.group("date")),
        "postnatal_day": int(postnatal_day) if postnatal_day is not None else None,
        "day_label": match.group("day_label"),
    }


class SessionEntry(BaseModel):
    name: str
    animal_id: str
    date: date
    postnatal_day: int | None = None
    day_label: str
    calibration_toml_path: Path | None = None
    notes: str | None = None


class SessionQuery:
    """
    A chainable, filtered view over a list of `SessionEntry`. Every filter
    method returns a new `SessionQuery`, so filters combine with AND by
    chaining calls:

        manager.query().animal_prefix("7").age_range(30, 45).exclude_animal("753")

    Terminal methods (`names`, `paths`, `recordings`) pull the filtered
    entries back out in the shape most callers want.
    """

    def __init__(self, manager: "SessionManager", entries: list[SessionEntry]):
        self._manager = manager
        self.entries = entries

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"<SessionQuery: {len(self.entries)} session(s)>"

    def _filtered(self, entries: list[SessionEntry]) -> "SessionQuery":
        return SessionQuery(self._manager, entries)

    # --- animal filters ---

    def animal(self, *animal_ids: str) -> "SessionQuery":
        """Keep only entries whose animal_id is one of `animal_ids`."""
        wanted = set(animal_ids)
        return self._filtered([e for e in self.entries if e.animal_id in wanted])

    def animal_prefix(self, *prefixes: str) -> "SessionQuery":
        """Keep entries whose animal_id starts with any of `prefixes`, e.g. prefix="7" matches ferrets 700-799."""
        return self._filtered([e for e in self.entries if e.animal_id.startswith(prefixes)])

    def exclude_animal(self, *animal_ids: str) -> "SessionQuery":
        """Drop entries whose animal_id is one of `animal_ids`."""
        excluded = set(animal_ids)
        return self._filtered([e for e in self.entries if e.animal_id not in excluded])

    def exclude(self, *names: str) -> "SessionQuery":
        """Drop entries whose session name is one of `names`."""
        excluded = set(names)
        return self._filtered([e for e in self.entries if e.name not in excluded])

    # --- date filters ---

    def date(self, target: date) -> "SessionQuery":
        return self._filtered([e for e in self.entries if e.date == target])

    def date_range(self, start: date, end: date) -> "SessionQuery":
        return self._filtered([e for e in self.entries if start <= e.date <= end])

    # --- age (postnatal day) filters ---

    def age(self, postnatal_day: int) -> "SessionQuery":
        """Keep entries with this exact postnatal day. Entries with unknown age never match."""
        return self._filtered([e for e in self.entries if e.postnatal_day == postnatal_day])

    def age_range(self, start: int, end: int) -> "SessionQuery":
        """Keep entries whose postnatal day falls in [start, end] inclusive. Entries with unknown age never match."""
        return self._filtered([
            e for e in self.entries
            if e.postnatal_day is not None and start <= e.postnatal_day <= end
        ])

    # --- day label filters ---

    def day_label(self, label: str, exact: bool = True) -> "SessionQuery":
        if exact:
            return self._filtered([e for e in self.entries if e.day_label == label])
        return self._filtered([e for e in self.entries if e.day_label.startswith(label)])

    def day_label_range(self, start_label: str, end_label: str) -> "SessionQuery":
        """
        Keep entries whose day_label number falls in [start, end] inclusive. E
        and EO are treated as equivalent, e.g. ("E5", "E10") matches E5..E10
        and EO5..EO10 sessions alike.
        """
        start_number = _day_label_number(start_label)
        end_number = _day_label_number(end_label)
        if start_number > end_number:
            raise ValueError(f"day_label range start must be <= end, got {start_label!r} and {end_label!r}")

        return self._filtered([
            e for e in self.entries
            if start_number <= _day_label_number(e.day_label) <= end_number
        ])

    # --- pipeline-step filters (hit disk) ---

    def pending(self, step: PipelineStep) -> "SessionQuery":
        """Keep entries whose recording on disk has not (yet) completed `step`."""
        return self._filtered(self._manager._filter_by_step(self.entries, step, want_complete=False))

    def done(self, step: PipelineStep) -> "SessionQuery":
        """Keep entries whose recording on disk has (already) completed `step`."""
        return self._filtered(self._manager._filter_by_step(self.entries, step, want_complete=True))

    # --- terminal accessors ---

    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def paths(self) -> list[Path]:
        return [self._manager.recording_folder_path(e) for e in self.entries]

    def recordings(self) -> list[tuple[Path, Path | None]]:
        """(recording_folder_path, calibration_toml_path) pairs, ready for full_pipeline/batch_full_pipeline."""
        return self._manager.to_recordings(self.entries)


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

    @staticmethod
    def _to_raw(entry: SessionEntry) -> dict:
        return {
            "name": entry.name,
            "animal_id": entry.animal_id,
            "date": entry.date,
            "postnatal_day": entry.postnatal_day,
            "day_label": entry.day_label,
            "calibration_toml_path": str(entry.calibration_toml_path) if entry.calibration_toml_path else None,
            "notes": entry.notes,
        }

    def save(self) -> None:
        """Write `self.entries` back to `yaml_path`, in the same field order as `_load` expects."""
        raw_entries = [self._to_raw(entry) for entry in self.entries]
        with open(self.yaml_path, "w") as f:
            yaml.safe_dump(raw_entries, f, sort_keys=False, default_flow_style=False)

    def resolve_calibration_paths(self, overwrite: bool = False) -> dict[str, str]:
        """
        Pin each entry's `calibration_toml_path` using the current
        auto-discovery logic (RecordingFolder.calibration_toml_path), so
        triangulation always runs against an explicit, known-good calibration
        file instead of re-discovering (and potentially mis-discovering) one
        at pipeline-run time.

        By default, entries that already have a `calibration_toml_path` set
        are left alone — pass overwrite=True to re-resolve everything (e.g.
        after a batch recalibration).

        Does not write to disk; call save() afterwards once you've reviewed
        the returned report.
        """
        report: dict[str, str] = {}
        for entry in self.entries:
            if entry.calibration_toml_path is not None and not overwrite:
                report[entry.name] = f"already set: {entry.calibration_toml_path}"
                continue

            folder_path = self.recording_folder_path(entry)
            if not folder_path.exists():
                report[entry.name] = f"skipped: {folder_path} does not exist"
                continue

            recording_folder = RecordingFolder.from_folder_path(folder_path)
            try:
                calibration_toml_path = recording_folder.calibration_toml_path
            except ValueError as e:
                report[entry.name] = f"AMBIGUOUS, left unset: {e}"
                continue

            if calibration_toml_path is None:
                report[entry.name] = "not found: no calibration toml in calibration folder"
                continue

            entry.calibration_toml_path = calibration_toml_path
            report[entry.name] = f"resolved: {calibration_toml_path}"
        return report

    def query(self) -> SessionQuery:
        """Entry point for chainable queries, e.g. manager.query().animal_prefix("7").age_range(30, 45)."""
        return SessionQuery(self, list(self.entries))

    def all(self) -> list[SessionEntry]:
        return list(self.entries)

    def get(self, name: str) -> SessionEntry | None:
        """Look up a single entry by its exact session name."""
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def recording_folder_path(self, entry: SessionEntry) -> Path:
        return self.base_recordings_root / entry.name / "full_recording"

    def to_recordings(self, entries: list[SessionEntry]) -> list[tuple[Path, Path | None]]:
        return [
            (self.recording_folder_path(entry), entry.calibration_toml_path)
            for entry in entries
        ]

    def _filter_by_step(
        self,
        entries: list[SessionEntry],
        step: PipelineStep,
        want_complete: bool,
    ) -> list[SessionEntry]:
        """
        Entries whose recording on disk has (or hasn't) completed `step`.
        Entries whose recording folder doesn't exist on disk are skipped and
        reported, not raised.
        """
        is_step_complete = _STEP_CHECKS.get(step)
        if is_step_complete is None:
            raise ValueError(f"No processing check available for step: {step}")

        matches = []
        for entry in entries:
            folder_path = self.recording_folder_path(entry)
            if not folder_path.exists():
                print(f"Skipping {entry.name}: {folder_path} does not exist")
                continue
            recording_folder = RecordingFolder.from_folder_path(folder_path)
            if is_step_complete(recording_folder) == want_complete:
                matches.append(entry)
        return matches

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
    import sys

    session_manager = SessionManager()

    if "--resolve-calibration" in sys.argv:
        overwrite = "--overwrite" in sys.argv
        report = session_manager.resolve_calibration_paths(overwrite=overwrite)
        for name, outcome in report.items():
            print(f"  {name}: {outcome}")
        session_manager.save()
        print(f"\nSaved {session_manager.yaml_path}")
    else:
        issues = session_manager.verify()
        if issues:
            print(f"=== {len(issues)} verification issue(s) ===")
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"All {len(session_manager.all())} sessions verified OK")
