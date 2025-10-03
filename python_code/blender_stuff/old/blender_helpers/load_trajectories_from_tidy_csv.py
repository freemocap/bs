"""
Dataclasses for reading and organizing 3D keypoint tracking data in tidy format.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import csv
import numpy as np


@dataclass(frozen=True)
class Point3D:
    """Immutable 3D point with x, y, z coordinates."""
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class KeypointObservation:
    """Single observation of a keypoint at a specific frame."""
    frame: int
    keypoint_name: str
    position: Point3D


@dataclass
class Trajectory:
    """Time series of 3D positions for a single keypoint across multiple frames."""
    keypoint_name: str
    observations: list[tuple[int, Point3D]] = field(default_factory=list)

    def add_observation(self, *, frame: int, position: Point3D) -> None:
        """Add an observation to this trajectory."""
        self.observations.append((frame, position))

    def get_position_at_frame(self, *, frame: int) -> Point3D | None:
        """Get the position at a specific frame, or None if not found."""
        for obs_frame, position in self.observations:
            if obs_frame == frame:
                return position
        return None

    def to_numpy(self, *, scale_factor: float = 1.0) -> np.ndarray:
        """Convert trajectory to numpy array of shape (n_frames, 3)."""
        xyz_array = np.array(
            [[pos.x, pos.y, pos.z] for _, pos in self.observations],
            dtype=np.float32
        )
        return xyz_array * scale_factor

    @property
    def frames(self) -> list[int]:
        """Get all frame numbers in this trajectory."""
        return [frame for frame, _ in self.observations]

    @property
    def positions(self) -> list[Point3D]:
        """Get all positions in this trajectory."""
        return [pos for _, pos in self.observations]


def read_trajectory_csv(*, filepath: Path | str) -> dict[str, Trajectory]:
    """
    Read a CSV file in tidy format and return a dictionary of trajectories.

    Args:
        filepath: Path to the CSV file with columns: frame, keypoint, x, y, z

    Returns:
        Dictionary mapping keypoint names to their Trajectory objects
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    trajectories: dict[str, Trajectory] = {}

    with filepath.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

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

            trajectories[keypoint_name].add_observation(
                frame=frame,
                position=position
            )

    return trajectories


def iter_observations(*, filepath: Path | str) -> Iterator[KeypointObservation]:
    """
    Iterate over keypoint observations without loading all data into memory.

    Args:
        filepath: Path to the CSV file

    Yields:
        KeypointObservation objects one at a time
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    with filepath.open(mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            yield KeypointObservation(
                frame=int(row['frame']),
                keypoint_name=row['keypoint'],
                position=Point3D(
                    x=float(row['x']),
                    y=float(row['y']),
                    z=float(row['z'])
                )
            )


# Example usage
if __name__ == "__main__":
    _csv_path =r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\output_data_head_body_eyecam_retrain_test_v2_model_outputs_iteration_1\dlc\dlc_body_rigid_3d_xyz.csv"
    trajectories = read_trajectory_csv(filepath=_csv_path)

    print(f"Found {len(trajectories)} keypoints:")
    for keypoint_name, trajectory in trajectories.items():
        print(f"  {keypoint_name}: {len(trajectory.frames)} frames")


def trajectories_to_numpy_dict(*, trajectories: dict[str, Trajectory], scale_factor: float = 1.0) -> dict[str, np.ndarray]:
    """Convert dict of Trajectory objects to dict of numpy arrays for Blender."""
    return {name: traj.to_numpy(scale_factor=scale_factor) for name, traj in trajectories.items()}