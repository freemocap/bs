import json
import subprocess
from datetime import datetime
from pathlib import Path


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
