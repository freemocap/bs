# Batch Processing

Tools for running the pipeline (or ad-hoc scripts) across many recording
sessions at once, instead of hand-maintaining lists of paths.

## `sessions.yaml` + `SessionManager`

`sessions.yaml` is the source of truth for every known recording session:
name, animal id, date, day label, and (optionally) a calibration toml path
or notes. `SessionManager` (`session_manager.py`) loads it and answers
queries against it, so scripts select sessions declaratively instead of
copy-pasting folder names.

```python
from python_code.batch_processing.session_manager import SessionManager

session_manager = SessionManager()
```

By default it reads `sessions.yaml` from this folder and resolves recordings
under `/home/scholl-lab/ferret_recordings` (the Scholl Lab machine). Override
either for local testing or a different machine:

```python
session_manager = SessionManager(
    yaml_path=Path("some/other/sessions.yaml"),
    base_recordings_root=Path("/mnt/data/ferret_recordings"),
)
```

### Adding a session

Add an entry to `sessions.yaml`:

```yaml
- name: session_2026-03-14_ferret_407_P47_E14
  animal_id: "407"
  date: 2026-03-14
  day_label: E14
  calibration_toml_path: null
  notes: null
```

`name` must match `SESSION_NAME_PATTERN` in `session_manager.py` (the same
pattern the recording folder itself uses) — `verify()` (below) will catch it
if it doesn't. `day_label` is either `E<n>` or `EO<n>`.

### Querying sessions

All queries return `list[SessionEntry]`, so they compose with plain Python
(list comprehensions, `+`, etc.) before you hand them off:

```python
session_manager.all()
session_manager.by_animal("407")
session_manager.by_date(date(2026, 3, 14))
session_manager.by_date_range(date(2026, 3, 1), date(2026, 3, 31))
session_manager.by_day_label("E14")                 # exact match
session_manager.by_day_label("E1", exact=False)     # prefix match: E1, E10, E11, ...
session_manager.by_day_label_range("EO10", "EO15")  # inclusive; same prefix required (EO here)
```

`by_day_label_range` raises `ValueError` if the two labels don't share a
prefix (`"E5"`/`"EO10"`) or if start > end.

### Turning entries into pipeline inputs

`full_pipeline`/`batch_full_pipeline` want `(recording_folder_path,
calibration_toml_path)` pairs, not `SessionEntry` objects:

```python
from python_code.batch_processing.batch_pipeline import batch_full_pipeline

entries = session_manager.by_animal("407")
recordings = session_manager.to_recordings(entries)
batch_full_pipeline(recordings=recordings, overwrite_dlc=True, ...)
```

To only pick up sessions that haven't finished a given pipeline step yet
(skips anything missing on disk, printing what it skips):

```python
from python_code.utilities.folder_utilities.recording_folder import PipelineStep

recordings = session_manager.not_processed_through(PipelineStep.GAZE_POST_PROCESSED)
batch_full_pipeline(recordings=recordings)
```

This only produces meaningful results on a machine where
`base_recordings_root` actually has the recordings on disk (e.g. the Scholl
Lab machine) — the other queries above work off the YAML alone and don't
touch disk.

### Keeping the YAML honest

`verify()` re-derives animal id / date / day label from each entry's `name`
and flags any mismatch against what's stored (e.g. a typo when the entry was
added by hand):

```bash
python -m python_code.batch_processing.session_manager
```

Run this after editing `sessions.yaml` by hand.

## Selecting sessions to sync (`select_sessions.py`)

`select_sessions.py` is a thin CLI over the same filters, for cases where you
need a plain list of session folder names rather than Python objects — e.g.
feeding `scripts/sync_recordings_to_analysis_pc.sh`'s `SESSION_LIST` when you
only want to transfer a subset of sessions to another machine. Filters
combine with AND; only sessions that exist on disk are printed (others are
skipped with a warning to stderr).

```bash
# Everything for animal 753 in EO10-EO15
python -m python_code.batch_processing.select_sessions --animal 753 --day-label-range EO10-EO15

# A specific date range, any animal
python -m python_code.batch_processing.select_sessions --date-start 2025-06-01 --date-end 2025-06-30
```

## Other scripts in this folder

- `full_pipeline.py` — runs the full pipeline for a single recording.
- `batch_pipeline.py` — runs `full_pipeline` over a list of recordings
  sequentially; see its `__main__` block for a worked example using
  `SessionManager`.
- `postprocess_recording.py` — rigid body solving step for a single
  recording.
- `force_copy_analyzable_output.py` — re-copies `analyzable_output/` to
  Dropbox for recordings where the pipeline's own copy step didn't run.
