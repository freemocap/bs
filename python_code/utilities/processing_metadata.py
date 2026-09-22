import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


def describe_calibration_file(calibration_toml_path: Path | None) -> dict:
    """Capture enough about a calibration toml to identify which calibration ran.

    Path alone isn't sufficient: a calibration folder can be overwritten in place
    by a later recalibration under the same filename, so a content hash + mtime
    are included to detect that.
    """
    if calibration_toml_path is None:
        return {"calibration_toml_path": None}

    calibration_toml_path = Path(calibration_toml_path)
    if not calibration_toml_path.exists():
        return {"calibration_toml_path": str(calibration_toml_path)}

    contents = calibration_toml_path.read_bytes()
    return {
        "calibration_toml_path": str(calibration_toml_path),
        "calibration_toml_hash": hashlib.md5(contents).hexdigest(),
        "calibration_toml_mtime": datetime.fromtimestamp(
            calibration_toml_path.stat().st_mtime
        ).isoformat(),
    }


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def write_step_metadata(
    metadata_path: Path,
    step: str,
    parameters: dict,
    extra: dict | None = None,
) -> None:
    """Read existing metadata JSON (or start fresh), overwrite `step`, and save."""
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    step_data: dict = {
        "timestamp": datetime.now().isoformat(),
        "bs_git_hash": _get_git_hash(),
        "parameters": parameters,
    }
    if extra:
        step_data.update(extra)

    metadata[step] = step_data

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
