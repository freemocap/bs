"""Load 3D keypoint tracking data from tidy CSV format."""

from dataclasses import dataclass, field
from pathlib import Path
import csv
import numpy as np


@dataclass(frozen=True)
class Point3D:
    """Immutable 3D point."""
    x: float
    y: float
    z: float


@dataclass
class Trajectory:
    """Time series of 3D positions for a single keypoint."""
    keypoint_name: str
    observations: list[tuple[int, Point3D]] = field(default_factory=list)

    def add_observation(self, *, frame: int, position: Point3D) -> None:
        """Add an observation to this trajectory."""
        self.observations.append((frame, position))

    def to_numpy(self, *, scale_factor: float = 1.0) -> np.ndarray:
        """
        Convert trajectory to numpy array of shape (n_frames, 3).

        Args:
            scale_factor: Multiplier for coordinates (e.g., 0.001 for mm to m)

        Returns:
            Array of shape (n_frames, 3) with x, y, z coordinates
        """
        xyz_array = np.array(
            [[pos.x, pos.y, pos.z] for _, pos in self.observations],
            dtype=np.float32
        )
        return xyz_array * scale_factor

    @property
    def num_frames(self) -> int:
        """Number of frames in this trajectory."""
        return len(self.observations)


def load_trajectories_from_tidy_csv(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Load trajectories from tidy/long-format CSV and convert to numpy arrays.

    Args:
        filepath: Path to CSV with columns: frame, keypoint, x, y, z
        scale_factor: Scale multiplier for coordinates (default: 1.0)

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) numpy arrays
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    print(f"Loading trajectories from: {filepath.name}")

    # First pass: read and organize data
    trajectories: dict[str, Trajectory] = {}

    with filepath.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f=f)

        for row in reader:
            frame = int(row['frame'])
            keypoint_name = row['keypoint']
            position = Point3D(
                x=float(row['x']),
                y=float(row['y']),
                z=float(row['z']) if 'z' in row and row['z'] else 0.0
            )

            if keypoint_name not in trajectories:
                trajectories[keypoint_name] = Trajectory(keypoint_name=keypoint_name)

            trajectories[keypoint_name].add_observation(frame=frame, position=position)

    # Convert to numpy arrays
    trajectory_arrays = {
        name: traj.to_numpy(scale_factor=scale_factor)
        for name, traj in trajectories.items()
    }

    num_keypoints = len(trajectory_arrays)
    num_frames = next(iter(trajectories.values())).num_frames
    print(f"✓ Loaded {num_keypoints} keypoints × {num_frames} frames")

    return trajectory_arrays


def load_trajectories_from_wide_csv(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Load 2D trajectories from wide-format CSV and convert to 3D numpy arrays.

    Expected CSV format:
    - Columns: frame, video (optional), {keypoint}_x, {keypoint}_y, ...
    - Each row represents one frame
    - Keypoint coordinates are in paired x,y columns

    Args:
        filepath: Path to CSV with paired {keypoint}_x, {keypoint}_y columns
        scale_factor: Scale multiplier for coordinates (default: 1.0)
        z_value: Default z-coordinate value for 2D data (default: 0.0)

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) numpy arrays
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    print(f"Loading wide-format trajectories from: {filepath.name}")

    # Read CSV and extract column names
    with filepath.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f=f)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError(f"CSV file has no headers: {filepath}")

        # Identify keypoint columns (those ending in _x or _y)
        keypoint_names: set[str] = set()
        for header in headers:
            if header.endswith('_x'):
                keypoint_name = header[:-2]  # Remove '_x' suffix
                keypoint_names.add(keypoint_name)

        if not keypoint_names:
            raise ValueError(f"No keypoint columns found (expected columns ending in '_x')")

        # Initialize storage for trajectories
        trajectory_data: dict[str, list[list[float]]] = {
            name: [] for name in keypoint_names
        }

        # Read all rows
        f.seek(0)  # Reset file pointer
        reader = csv.DictReader(f=f)

        for row in reader:
            for keypoint_name in keypoint_names:
                x_col = f"{keypoint_name}_x"
                y_col = f"{keypoint_name}_y"

                # Handle missing values
                try:
                    x = float(row[x_col]) if row[x_col] else np.nan
                    y = float(row[y_col]) if row[y_col] else np.nan
                except (KeyError, ValueError):
                    x = np.nan
                    y = np.nan

                trajectory_data[keypoint_name].append([x, y, z_value])

    # Convert to numpy arrays and apply scaling
    trajectory_arrays: dict[str, np.ndarray] = {}
    for keypoint_name, coords in trajectory_data.items():
        array = np.array(coords, dtype=np.float32)
        trajectory_arrays[keypoint_name] = array * scale_factor

    num_keypoints = len(trajectory_arrays)
    num_frames = len(next(iter(trajectory_data.values())))
    print(f"✓ Loaded {num_keypoints} keypoints × {num_frames} frames (2D data)")

    return trajectory_arrays


def detect_csv_format(*, filepath: Path | str) -> str:
    """
    Detect whether CSV is in 'tidy' or 'wide' format.

    Args:
        filepath: Path to CSV file

    Returns:
        Either 'tidy' or 'wide'
    """
    filepath = Path(filepath)

    with filepath.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f=f)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError(f"CSV file has no headers: {filepath}")

        # Tidy format has columns: frame, keypoint, x, y, z
        if 'keypoint' in headers and 'x' in headers and 'y' in headers:
            return 'tidy'

        # Wide format has columns like: frame, {keypoint}_x, {keypoint}_y
        has_x_columns = any(h.endswith('_x') for h in headers)
        has_y_columns = any(h.endswith('_y') for h in headers)

        if has_x_columns and has_y_columns:
            return 'wide'

        raise ValueError(
            f"Unable to detect CSV format. Expected either:\n"
            f"  - Tidy format: columns 'frame', 'keypoint', 'x', 'y', 'z'\n"
            f"  - Wide format: columns 'frame', '{{keypoint}}_x', '{{keypoint}}_y'"
        )


def load_trajectories_auto(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Automatically detect CSV format and load trajectories.

    Args:
        filepath: Path to CSV file (tidy or wide format)
        scale_factor: Scale multiplier for coordinates (default: 1.0)
        z_value: Default z-coordinate for 2D data in wide format (default: 0.0)

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) numpy arrays
    """
    csv_format = detect_csv_format(filepath=filepath)

    if csv_format == 'tidy':
        return load_trajectories_from_tidy_csv(
            filepath=filepath,
            scale_factor=scale_factor
        )
    else:  # wide format
        return load_trajectories_from_wide_csv(
            filepath=filepath,
            scale_factor=scale_factor,
            z_value=z_value
        )