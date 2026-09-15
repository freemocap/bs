"""
Filter sessions.yaml down to a list of session folder names, for feeding into
scripts/sync_recordings_to_analysis_pc.sh's SESSION_LIST.

Filters combine with AND. Prints one session folder name per line to stdout;
warns to stderr (and skips) any selected session whose folder doesn't exist
under base_recordings_root.

Examples:
    # EO10 through EO15 inclusive, any animal
    python -m python_code.batch_processing.select_sessions --day-label-range EO10-EO15

    # A specific date range
    python -m python_code.batch_processing.select_sessions --date-start 2025-06-01 --date-end 2025-06-30

    # Combine with animal id
    python -m python_code.batch_processing.select_sessions --day-label-range E5-E10 --animal 753
"""
import argparse
import sys
from datetime import date

from python_code.batch_processing.session_manager import SessionEntry, SessionManager


def select(manager: SessionManager, args: argparse.Namespace) -> list[SessionEntry]:
    entries = manager.all()

    if args.animal:
        entries = [e for e in entries if e.animal_id == args.animal]

    if args.day_label_range:
        start_label, _, end_label = args.day_label_range.partition("-")
        if not end_label:
            raise ValueError(f"--day-label-range must be START-END, e.g. EO10-EO15 (got {args.day_label_range!r})")
        allowed = {e.name for e in manager.by_day_label_range(start_label, end_label)}
        entries = [e for e in entries if e.name in allowed]

    if args.date_start or args.date_end:
        start = date.fromisoformat(args.date_start) if args.date_start else date.min
        end = date.fromisoformat(args.date_end) if args.date_end else date.max
        entries = [e for e in entries if start <= e.date <= end]

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--animal", help="Exact animal_id, e.g. 753")
    parser.add_argument("--day-label-range", help="e.g. EO10-EO15 or E5-E10 — same prefix, inclusive")
    parser.add_argument("--date-start", help="ISO date, inclusive")
    parser.add_argument("--date-end", help="ISO date, inclusive")
    args = parser.parse_args()

    manager = SessionManager()
    selected = select(manager, args)

    for entry in sorted(selected, key=lambda e: e.name):
        folder_path = manager.recording_folder_path(entry).parent  # session dir, not full_recording/
        if not folder_path.exists():
            print(f"Skipping {entry.name}: {folder_path} does not exist", file=sys.stderr)
            continue
        print(entry.name)


if __name__ == "__main__":
    main()
