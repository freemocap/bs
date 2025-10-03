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


def load_trajectories_from_csv(
        *,
        filepath: Path | str,
        scale_factor: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Load trajectories from CSV and convert directly to numpy arrays.

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
                z=float(row['z'])
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