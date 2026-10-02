"""
Filter sessions.yaml down to a list of session folder names, for feeding into
scripts/sync_recordings_to_analysis_pc.sh's SESSION_LIST.

Filters combine with AND. Prints one session folder name per line to stdout;
warns to stderr (and skips) any selected session whose folder doesn't exist
under base_recordings_root.

Examples:
    # EO10 through EO15 inclusive, any animal (E and EO are treated as
    # equivalent day counts, so this also picks up any matching E10-E15)
    python -m python_code.batch_processing.select_sessions --day-label-range EO10-EO15

    # A specific date range
    python -m python_code.batch_processing.select_sessions --date-start 2025-06-01 --date-end 2025-06-30

    # Postnatal day (age) range, any animal
    python -m python_code.batch_processing.select_sessions --age-range 30-45

    # Combine with animal id, and exclude a known-bad session
    python -m python_code.batch_processing.select_sessions --day-label-range E5-E10 --animal 753
"""
import argparse
import sys
from datetime import date

from python_code.batch_processing.session_manager import SessionEntry, SessionManager


def _parse_range(raw: str, flag: str) -> tuple[str, str]:
    start, _, end = raw.partition("-")
    if not end:
        raise ValueError(f"{flag} must be START-END, e.g. 10-15 (got {raw!r})")
    return start, end


def select(manager: SessionManager, args: argparse.Namespace) -> list[SessionEntry]:
    query = manager.query()

    if args.animal:
        query = query.animal(*args.animal)

    if args.animal_prefix:
        query = query.animal_prefix(*args.animal_prefix)

    if args.exclude_animal:
        query = query.exclude_animal(*args.exclude_animal)

    if args.day_label_range:
        start_label, end_label = _parse_range(args.day_label_range, "--day-label-range")
        query = query.day_label_range(start_label, end_label)

    if args.age_range:
        start_age, end_age = _parse_range(args.age_range, "--age-range")
        query = query.age_range(int(start_age), int(end_age))

    if args.date_start or args.date_end:
        start = date.fromisoformat(args.date_start) if args.date_start else date.min
        end = date.fromisoformat(args.date_end) if args.date_end else date.max
        query = query.date_range(start, end)

    return query.entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--animal", action="append", help="Exact animal_id, e.g. 753. Repeatable.")
    parser.add_argument("--animal-prefix", action="append", help="Animal id prefix, e.g. 7 matches ferrets 700-799. Repeatable.")
    parser.add_argument("--exclude-animal", action="append", help="Exclude this exact animal_id. Repeatable.")
    parser.add_argument("--day-label-range", help="e.g. EO10-EO15 or E5-E10, inclusive — E and EO are treated as equivalent")
    parser.add_argument("--age-range", help="Postnatal day range, inclusive, e.g. 30-45")
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
