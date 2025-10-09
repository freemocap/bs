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


def load_trajectories_from_dlc_csv(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
        likelihood_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Load trajectories from DeepLabCut multi-header CSV format.

    Expected CSV format (3-row header):
    - Row 0: scorer (metadata, ignored)
    - Row 1: bodypart names (repeated for x, y, likelihood)
    - Row 2: coords ('x', 'y', 'likelihood' for each bodypart)
    - Row 3+: data rows

    Args:
        filepath: Path to DLC CSV file
        scale_factor: Scale multiplier for coordinates (default: 1.0)
        z_value: Default z-coordinate value for 2D data (default: 0.0)
        likelihood_threshold: If provided, set coordinates to NaN when likelihood < threshold

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) numpy arrays
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    print(f"Loading DLC-format trajectories from: {filepath.name}")

    # Read the file to parse the multi-level header
    with filepath.open(mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 4:
        raise ValueError(f"DLC CSV must have at least 4 rows (3 header + 1 data)")

    # Parse the 3-row header structure
    scorer_row = lines[0].strip().split(',')
    bodypart_row = lines[1].strip().split(',')
    coords_row = lines[2].strip().split(',')

    # Build column mapping: determine which columns correspond to which bodypart and coord type
    column_map: dict[str, dict[str, int]] = {}  # {bodypart: {'x': col_idx, 'y': col_idx, 'likelihood': col_idx}}
    
    for col_idx, (bodypart, coord_type) in enumerate(zip(bodypart_row, coords_row)):
        bodypart = bodypart.strip()
        coord_type = coord_type.strip()
        
        if not bodypart or bodypart == 'scorer':  # Skip empty or metadata columns
            continue
            
        if bodypart not in column_map:
            column_map[bodypart] = {}
        
        column_map[bodypart][coord_type] = col_idx

    # Validate that each bodypart has x and y columns
    valid_bodyparts: list[str] = []
    for bodypart, coords in column_map.items():
        if 'x' in coords and 'y' in coords:
            valid_bodyparts.append(bodypart)
        else:
            print(f"Warning: Bodypart '{bodypart}' missing x or y coordinate, skipping")

    if not valid_bodyparts:
        raise ValueError("No valid bodyparts found with both x and y coordinates")

    # Initialize storage
    trajectory_data: dict[str, list[list[float]]] = {
        name: [] for name in valid_bodyparts
    }

    # Parse data rows (skip first 3 header rows)
    for line_idx, line in enumerate(lines[3:], start=3):
        values = line.strip().split(',')
        
        for bodypart in valid_bodyparts:
            coords = column_map[bodypart]
            
            try:
                x_str = values[coords['x']].strip()
                y_str = values[coords['y']].strip()
                
                x = float(x_str) if x_str else np.nan
                y = float(y_str) if y_str else np.nan
                
                # Apply likelihood threshold if specified
                if likelihood_threshold is not None and 'likelihood' in coords:
                    likelihood_str = values[coords['likelihood']].strip()
                    likelihood = float(likelihood_str) if likelihood_str else 0.0
                    
                    if likelihood < likelihood_threshold:
                        x = np.nan
                        y = np.nan
                        
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line {line_idx}, bodypart '{bodypart}': {e}")
                x = np.nan
                y = np.nan
            
            trajectory_data[bodypart].append([x, y, z_value])

    # Convert to numpy arrays and apply scaling
    trajectory_arrays: dict[str, np.ndarray] = {}
    for keypoint_name, coords in trajectory_data.items():
        array = np.array(coords, dtype=np.float32)
        trajectory_arrays[keypoint_name] = array * scale_factor

    num_keypoints = len(trajectory_arrays)
    num_frames = len(next(iter(trajectory_data.values())))
    print(f"✓ Loaded {num_keypoints} keypoints × {num_frames} frames (DLC format)")

    return trajectory_arrays


def detect_csv_format(*, filepath: Path | str) -> str:
    """
    Detect whether CSV is in 'tidy', 'wide', or 'dlc' format.

    Args:
        filepath: Path to CSV file

    Returns:
        Either 'tidy', 'wide', or 'dlc'
    """
    filepath = Path(filepath)

    with filepath.open(mode='r', encoding='utf-8') as f:
        # Read first few lines to detect format
        lines = [f.readline().strip() for _ in range(3)]
    
    if len(lines) < 1:
        raise ValueError(f"CSV file is empty: {filepath}")

    # Check for DLC format: 3-row header with bodyparts in row 2, coords in row 3
    if len(lines) >= 3:
        row2_values = lines[1].split(',')
        row3_values = lines[2].split(',')
        
        # DLC format has 'coords' or coordinate type labels in row 3
        if len(row3_values) > 1 and any(val.strip() in ['x', 'y', 'likelihood', 'coords'] for val in row3_values):
            return 'dlc'

    # Parse as regular CSV with DictReader for tidy/wide detection
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
            f"  - Wide format: columns 'frame', '{{keypoint}}_x', '{{keypoint}}_y'\n"
            f"  - DLC format: 3-row header (scorer/bodyparts/coords)"
        )


def load_trajectories_auto(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
        z_value: float = 0.0,
        likelihood_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Automatically detect CSV format and load trajectories.

    Args:
        filepath: Path to CSV file (tidy, wide, or DLC format)
        scale_factor: Scale multiplier for coordinates (default: 1.0)
        z_value: Default z-coordinate for 2D data in wide/DLC format (default: 0.0)
        likelihood_threshold: For DLC format, filter low-confidence points (default: None)

    Returns:
        Dictionary mapping keypoint names to (n_frames, 3) numpy arrays
    """
    csv_format = detect_csv_format(filepath=filepath)
    
    print(f"Detected format: {csv_format}")

    if csv_format == 'tidy':
        return load_trajectories_from_tidy_csv(
            filepath=filepath,
            scale_factor=scale_factor
        )
    elif csv_format == 'dlc':
        return load_trajectories_from_dlc_csv(
            filepath=filepath,
            scale_factor=scale_factor,
            z_value=z_value,
            likelihood_threshold=likelihood_threshold
        )
    else:  # wide format
        return load_trajectories_from_wide_csv(
            filepath=filepath,
            scale_factor=scale_factor,
            z_value=z_value
        )