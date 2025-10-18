"""Simplified dataset manager for loading and saving trajectory data.

Handles automatic CSV format detection and conversion to/from TrajectoryDataset.
"""

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from python_code.data_loaders.trajectory_loader.trajectory_dataset import TrajectoryDataset, TrajectoryND, \
    TrajectoryType

logger = logging.getLogger(__name__)


class TrajecotryCSVLoader(BaseModel):
    """Manages loading and saving TrajectoryDataset from/to CSV files.

    Features:
    - Automatic CSV format detection (DLC, Tidy, Wide)
    - Load directly into TrajectoryDataset
    - Save TrajectoryDataset to tidy CSV format
    """

    encoding: str = 'utf-8'
    default_z: float = 0.0
    trajectory_type:TrajectoryType = TrajectoryType.GENERIC
    min_confidence: float | None = None

    def load_csv(self, *, filepath: Path) -> TrajectoryDataset:
        """Load CSV file into TrajectoryDataset with automatic format detection.

        Supports three formats:
        1. DLC: 3-row header with bodyparts and x,y,likelihood
        2. Tidy: frame, keypoint, x, y, z columns
        3. Wide: marker_x, marker_y, marker_z columns

        Args:
            filepath: Path to CSV file

        Returns:
            TrajectoryDataset with loaded trajectories

        Raises:
            ValueError: If file format cannot be detected or parsed
        """
        if not filepath.exists():
            raise ValueError(f"File does not exist: {filepath}")

        if filepath.suffix.lower() != '.csv':
            raise ValueError(f"File must be CSV, got: {filepath.suffix}")

        # Detect format
        format_type = self._detect_format(filepath=filepath)
        logger.info(f"Detected CSV format: {format_type}")

        # Load based on format
        if format_type == "dlc":
            data = self._read_dlc(filepath=filepath)
        elif format_type == "tidy":
            data = self._read_tidy(filepath=filepath)
        elif format_type == "wide":
            data = self._read_wide(filepath=filepath)
        else:
            raise ValueError(f"Unknown CSV format in {filepath}")

        # Convert to TrajectoryDataset
        return self._to_trajectory_dataset(data=data)

    def save_csv(
            self,
            *,
            dataset: TrajectoryDataset,
            filepath: Path
    ) -> None:
        """Save TrajectoryDataset to tidy CSV format.

        Output format:
            frame, keypoint, x, y, z
            0, marker1, 1.0, 2.0, 3.0
            0, marker2, 4.0, 5.0, 6.0
            ...

        Args:
            dataset: TrajectoryDataset to save
            filepath: Output CSV path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for marker_name, trajectory in dataset.data.items():
            for frame_idx, values in zip(dataset.frame_indices, trajectory.data):
                if trajectory.n_dims == 2:
                    x, y = values
                    z = self.default_z
                else:
                    x, y, z = values[:3]

                rows.append({
                    "frame": int(frame_idx),
                    "keypoint": marker_name,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                })

        with open(filepath, mode='w', encoding=self.encoding, newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame", "keypoint", "x", "y", "z"]
            )
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Saved {len(rows)} rows to {filepath}")

    def _detect_format(self, *, filepath: Path) -> str:
        """Detect CSV format by inspecting header.

        Args:
            filepath: Path to CSV file

        Returns:
            Format name: "dlc", "tidy", or "wide"
        """
        with open(filepath, mode='r', encoding=self.encoding) as f:
            # Read first 5 lines for format detection
            lines: list[str] = []
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())

            # Go back to start to read header with csv.reader
            f.seek(0)
            reader = csv.reader(f)
            header: list[str] = next(reader)

        if len(lines) < 2:
            raise ValueError("CSV file has insufficient rows")

        # Check for DLC format (3-row header with x,y,likelihood pattern)
        if len(lines) >= 3:
            row3: list[str] = lines[2].split(',')
            if any(coord.strip().lower() in ['x', 'y', 'likelihood'] for coord in row3):
                return "dlc"

        # Check for Tidy format
        header_lower: list[str] = [col.lower().strip() for col in header]
        if all(col in header_lower for col in ['frame', 'keypoint', 'x', 'y']):
            return "tidy"

        # Check for Wide format
        if any(col.strip().endswith('_x') for col in header):
            return "wide"

        raise ValueError(
            f"Could not detect CSV format. Supported formats:\n"
            f"  - DLC: 3-row header with bodyparts and x,y,likelihood\n"
            f"  - Tidy: columns 'frame', 'keypoint', 'x', 'y', 'z'\n"
            f"  - Wide: columns 'marker_x', 'marker_y', 'marker_z'"
        )

    def _read_dlc(self, *, filepath: Path) -> dict[str, Any]:
        """Read DeepLabCut format CSV."""
        with open(filepath, mode='r', encoding=self.encoding) as f:
            lines = f.readlines()

        if len(lines) < 4:
            raise ValueError("DLC CSV needs at least 4 rows")

        # Parse 3-row header
        bodypart_row = [col.strip() for col in lines[1].split(',')]
        coords_row = [col.strip() for col in lines[2].split(',')]

        # Build column mapping
        column_map: dict[str, dict[str, int]] = {}
        for col_idx, (bodypart, coord) in enumerate(zip(bodypart_row, coords_row)):
            if bodypart and bodypart.lower() != 'scorer':
                if bodypart not in column_map:
                    column_map[bodypart] = {}
                column_map[bodypart][coord] = col_idx

        # Filter valid bodyparts
        valid_bodyparts = [
            bp for bp, coords in column_map.items()
            if 'x' in coords and 'y' in coords
        ]

        if not valid_bodyparts:
            raise ValueError("No valid bodyparts found")

        # Check dimensions
        has_z = any('z' in column_map[bp] for bp in valid_bodyparts)
        n_dims = 3 if has_z else 2
        n_frames = len(lines) - 3

        # Parse data
        trajectories: dict[str, np.ndarray] = {}
        confidence_data: dict[str, np.ndarray] = {}

        for bodypart in valid_bodyparts:
            coords = column_map[bodypart]
            positions = np.zeros((n_frames, n_dims))
            confidence = np.zeros(n_frames) if 'likelihood' in coords else None

            for i, line in enumerate(lines[3:]):
                values = line.split(',')

                try:
                    x = float(values[coords['x']].strip())
                    y = float(values[coords['y']].strip())

                    # Handle confidence
                    if confidence is not None:
                        conf_val = float(values[coords['likelihood']].strip())
                        confidence[i] = conf_val

                        # Filter by threshold
                        if self.min_confidence and conf_val < self.min_confidence:
                            x, y = np.nan, np.nan

                    # Get z if available
                    if has_z and 'z' in coords:
                        z = float(values[coords['z']].strip())
                        positions[i] = [x, y, z]
                    else:
                        positions[i] = [x, y]

                except (ValueError, IndexError):
                    positions[i] = [np.nan] * n_dims

            trajectories[bodypart] = positions
            if confidence is not None:
                confidence_data[bodypart] = confidence

        return {
            "trajectories": trajectories,
            "confidence": confidence_data if confidence_data else None,
            "frame_indices": np.arange(n_frames),
        }

    def _read_tidy(self, *, filepath: Path) -> dict[str, Any]:
        """Read tidy/long format CSV."""
        data_dict: dict[str, dict[int, np.ndarray]] = {}
        frame_set: set[int] = set()

        with open(filepath, mode='r', encoding=self.encoding) as f:
            reader = csv.DictReader(f)

            for row in reader:
                frame = int(row['frame'])
                keypoint = row['keypoint'].strip()
                x = float(row['x'])
                y = float(row['y'])
                z = float(row.get('z', self.default_z))

                frame_set.add(frame)

                if keypoint not in data_dict:
                    data_dict[keypoint] = {}

                data_dict[keypoint][frame] = np.array([x, y, z])

        # Convert to arrays
        frame_indices = np.array(sorted(frame_set))
        n_frames = len(frame_indices)

        trajectories = {}
        for keypoint, frame_data in data_dict.items():
            positions = np.full((n_frames, 3), np.nan)

            for i, frame_idx in enumerate(frame_indices):
                if frame_idx in frame_data:
                    positions[i] = frame_data[frame_idx]

            trajectories[keypoint] = positions

        return {
            "trajectories": trajectories,
            "confidence": None,
            "frame_indices": frame_indices,
        }

    def _read_wide(self, *, filepath: Path) -> dict[str, Any]:
        """Read wide format CSV."""
        with open(filepath, mode='r', encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            if not headers:
                raise ValueError("CSV has no headers")

            # Find marker names
            marker_names = {h[:-2] for h in headers if h.endswith('_x')}

            if not marker_names:
                raise ValueError("No markers found (expected columns ending in '_x')")

            rows = list(reader)

        n_frames = len(rows)
        frame_indices = np.arange(n_frames)

        # Extract trajectories
        trajectories = {}
        for marker in marker_names:
            x_col, y_col, z_col = f"{marker}_x", f"{marker}_y", f"{marker}_z"
            has_z = z_col in headers

            positions = np.zeros((n_frames, 3))

            for i, row in enumerate(rows):
                try:
                    x = float(row[x_col].strip() or 'nan')
                    y = float(row[y_col].strip() or 'nan')
                    z = float(row[z_col].strip() or self.default_z) if has_z else self.default_z

                    positions[i] = [x, y, z]
                except (ValueError, KeyError):
                    positions[i] = [np.nan, np.nan, self.default_z]

            trajectories[marker] = positions

        return {
            "trajectories": trajectories,
            "confidence": None,
            "frame_indices": frame_indices,
        }

    def _to_trajectory_dataset(
            self,
            *,
            data: dict[str, Any]
    ) -> TrajectoryDataset:
        """Convert parsed CSV data to TrajectoryDataset.

        Args:
            data: Dictionary with 'trajectories', 'confidence', 'frame_indices'

        Returns:
            TrajectoryDataset instance
        """
        trajectories = data["trajectories"]
        confidence = data.get("confidence")
        frame_indices = data["frame_indices"]

        # Create TrajectoryND instances
        trajectory_objects = {}
        for name, values in trajectories.items():
            conf = confidence[name] if confidence and name in confidence else None

            trajectory_objects[name] = TrajectoryND(
                name=name,
                data=values,
                confidence=conf,
                trajectory_type=self.trajectory_type,
                metadata={}
            )

        return TrajectoryDataset(
            data=trajectory_objects,
            frame_indices=frame_indices,
            metadata={"format": data.get("format", "unknown")}
        )


# Convenience functions for common use cases

def load_trajectory_csv(
        *,
        filepath: Path,
        trajectory_type:TrajectoryType = TrajectoryType.GENERIC,
        min_confidence: float | None = None
) -> TrajectoryDataset:
    manager = TrajecotryCSVLoader(min_confidence=min_confidence,
                                  trajectory_type=trajectory_type)
    return manager.load_csv(filepath=filepath)


def save_trajectory_csv(
        *,
        dataset: TrajectoryDataset,
        filepath: Path
) -> None:
    """Save TrajectoryDataset to tidy CSV format.

    Args:
        dataset: TrajectoryDataset to save
        filepath: Output CSV path
    """
    manager = TrajecotryCSVLoader()
    manager.save_csv(dataset=dataset, filepath=filepath)

