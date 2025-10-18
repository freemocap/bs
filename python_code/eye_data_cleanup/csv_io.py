"""Simplified CSV loader for 2D trajectory data.

Handles automatic CSV format detection (DLC, Tidy, Wide) for 2D positions only.
"""

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

"""Simplified 2D trajectory data structures for pixel tracking.

Focused on 2D position data only - all complex trajectory types removed.
"""

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class ABaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Trajectory2D(ABaseModel):
    """A 2D trajectory (x, y positions over time).

    Attributes:
        name: Identifier for this trajectory
        data: (n_frames, 2) x,y positions over time
        confidence: Optional (n_frames,) confidence scores [0-1]
        metadata: Optional additional data
    """

    name: str
    data: np.ndarray  # shape: (n_frames, 2)
    confidence: np.ndarray | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate(self) -> Self:
        """Validate trajectory data."""
        # Ensure 2D
        if self.data.ndim != 2 or self.data.shape[1] != 2:
            raise ValueError(f"Data must be (n_frames, 2), got shape {self.data.shape}")

        # Check confidence length matches
        if self.confidence is not None:
            if len(self.confidence) != len(self.data):
                raise ValueError(
                    f"Confidence length {len(self.confidence)} != data length {len(self.data)}"
                )

        return self

    @property
    def n_frames(self) -> int:
        """Number of frames in trajectory."""
        return len(self.data)

    @property
    def x(self) -> np.ndarray:
        """X coordinates (n_frames,)."""
        return self.data[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Y coordinates (n_frames,)."""
        return self.data[:, 1]

    def is_valid(self, *, min_confidence: float | None = None) -> np.ndarray:
        """Get mask of valid frames.

        Args:
            min_confidence: Minimum confidence threshold (default: no threshold)

        Returns:
            (n_frames,) boolean mask
        """
        # Start with not-NaN check
        not_nan = ~np.isnan(self.data[:, 0])

        if self.confidence is None or min_confidence is None:
            return not_nan

        # Valid if confidence above threshold AND not NaN
        above_threshold = self.confidence >= min_confidence
        return above_threshold & not_nan

    def get_valid_data(self, *, min_confidence: float | None = None) -> np.ndarray:
        """Get data for valid frames only.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            (n_valid, 2) valid positions
        """
        mask = self.is_valid(min_confidence=min_confidence)
        return self.data[mask]

    def interpolate_missing(self, *, method: str = "linear") -> "Trajectory2D":
        """Interpolate missing (NaN) values.

        Args:
            method: Interpolation method ("linear" or "cubic")

        Returns:
            New Trajectory2D with interpolated values
        """
        data_interp = self.data.copy()

        for axis in range(2):
            values = self.data[:, axis]
            valid_mask = ~np.isnan(values)

            if np.sum(valid_mask) < 2:
                # Not enough points to interpolate
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_values = values[valid_mask]

            # Interpolate
            interp_func = interp1d(
                valid_indices,
                valid_values,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate"
            )

            # Fill missing values
            missing_mask = np.isnan(values)
            if np.any(missing_mask):
                missing_indices = np.where(missing_mask)[0]
                data_interp[missing_indices, axis] = interp_func(missing_indices)

        return Trajectory2D(
            name=self.name,
            data=data_interp,
            confidence=self.confidence,
            metadata=self.metadata
        )

    def __str__(self) -> str:
        """Human-readable trajectory summary."""
        n_valid = np.sum(self.is_valid())
        validity_pct = (n_valid / self.n_frames * 100) if self.n_frames > 0 else 0

        lines = [
            f"Trajectory '{self.name}':",
            f"  Frames: {self.n_frames}",
            f"  Valid:  {n_valid}/{self.n_frames} ({validity_pct:.1f}%)",
            f"  Confidence: {'Yes' if self.confidence is not None else 'No'}"
        ]
        return "\n".join(lines)


class TrajectoryDataset(ABaseModel):
    """Collection of 2D trajectories with metadata.

    Attributes:
        data: Dictionary mapping names to Trajectory2D instances
        frame_indices: Frame numbers (may not start at 0)
        metadata: Optional dataset-level metadata
    """

    data: dict[str, Trajectory2D]
    frame_indices: np.ndarray
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate(self) -> Self:
        """Validate dataset."""
        if len(self.data) == 0:
            raise ValueError("Dataset must contain at least one trajectory")

        # Check all trajectories have same length
        n_frames_list = [traj.n_frames for traj in self.data.values()]
        if len(set(n_frames_list)) > 1:
            raise ValueError(f"All trajectories must have same length, got {n_frames_list}")

        if len(self.frame_indices) != n_frames_list[0]:
            raise ValueError(
                f"Frame indices length {len(self.frame_indices)} != trajectory length {n_frames_list[0]}"
            )

        return self

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.frame_indices)

    @property
    def marker_names(self) -> list[str]:
        """Get list of marker names."""
        return list(self.data.keys())

    @property
    def n_markers(self) -> int:
        """Number of markers."""
        return len(self.data)

    def slice_frames(self, *, start_frame: int, end_frame: int) -> Self:
        """Extract a slice of frames from the dataset."""
        sliced_data = {}
        for name, traj in self.data.items():
            sliced_values = traj.data[start_frame:end_frame].copy()
            sliced_confidence = (
                traj.confidence[start_frame:end_frame].copy()
                if traj.confidence is not None
                else None
            )

            sliced_data[name] = Trajectory2D(
                name=name,
                data=sliced_values,
                confidence=sliced_confidence,
                metadata=traj.metadata
            )

        return TrajectoryDataset(
            data=sliced_data,
            frame_indices=self.frame_indices[start_frame:end_frame].copy(),
            metadata=self.metadata
        )

    def to_array(self, *, marker_names: list[str] | None = None) -> np.ndarray:
        """Convert to numpy array.

        Args:
            marker_names: Optional list of markers to include (default: all)

        Returns:
            (n_frames, n_markers, 2) array of x,y positions
        """
        if marker_names is None:
            marker_names = self.marker_names

        # Check all requested markers exist
        missing = set(marker_names) - set(self.marker_names)
        if missing:
            raise ValueError(f"Markers not in dataset: {missing}")

        # Stack data
        arrays = [self.data[name].data for name in marker_names]
        return np.stack(arrays, axis=1)

    def get_valid_frames(
        self,
        *,
        min_confidence: float | None = None,
        min_valid_markers: int | None = None
    ) -> np.ndarray:
        """Get mask of frames with sufficient valid markers.

        Args:
            min_confidence: Minimum confidence threshold
            min_valid_markers: Minimum number of valid markers required (default: all)

        Returns:
            (n_frames,) boolean mask
        """
        if min_valid_markers is None:
            min_valid_markers = self.n_markers

        # Count valid markers per frame
        valid_counts = np.zeros(self.n_frames, dtype=int)
        for traj in self.data.values():
            valid_counts += traj.is_valid(min_confidence=min_confidence).astype(int)

        return valid_counts >= min_valid_markers

    def filter_by_confidence(
        self,
        *,
        min_confidence: float,
        min_valid_markers: int | None = None
    ) -> "TrajectoryDataset":
        """Filter dataset to keep only high-confidence frames.

        Args:
            min_confidence: Minimum confidence threshold
            min_valid_markers: Minimum valid markers per frame

        Returns:
            New filtered dataset
        """
        valid_mask = self.get_valid_frames(
            min_confidence=min_confidence,
            min_valid_markers=min_valid_markers
        )

        if not np.any(valid_mask):
            raise ValueError("No valid frames after filtering")

        # Filter each trajectory
        filtered_data = {}
        for name, traj in self.data.items():
            filtered_values = traj.data[valid_mask]
            filtered_confidence = traj.confidence[valid_mask] if traj.confidence is not None else None

            filtered_data[name] = Trajectory2D(
                name=name,
                data=filtered_values,
                confidence=filtered_confidence,
                metadata=traj.metadata
            )

        return TrajectoryDataset(
            data=filtered_data,
            frame_indices=self.frame_indices[valid_mask],
            metadata=self.metadata
        )

    def interpolate_missing(self, *, method: str = "linear") -> "TrajectoryDataset":
        """Interpolate missing data in all trajectories.

        Args:
            method: Interpolation method ("linear" or "cubic")

        Returns:
            New dataset with interpolated trajectories
        """
        interpolated_data = {
            name: traj.interpolate_missing(method=method)
            for name, traj in self.data.items()
        }

        return TrajectoryDataset(
            data=interpolated_data,
            frame_indices=self.frame_indices,
            metadata=self.metadata
        )

    def get_mean_position(self, *, min_confidence: float | None = None) -> np.ndarray:
        """Compute mean position across all markers over time.

        Args:
            min_confidence: Minimum confidence for valid data

        Returns:
            (n_frames, 2) mean x,y positions
        """
        all_values = []
        for traj in self.data.values():
            valid_mask = traj.is_valid(min_confidence=min_confidence)
            values_copy = traj.data.copy()
            values_copy[~valid_mask] = np.nan
            all_values.append(values_copy)

        # Stack and compute mean ignoring NaN
        stacked = np.stack(all_values, axis=1)  # (n_frames, n_markers, 2)
        return np.nanmean(stacked, axis=1)  # (n_frames, 2)

    def __str__(self) -> str:
        """Human-readable dataset summary."""
        lines = [
            f"TrajectoryDataset:",
            f"  Frames:  {self.n_frames}",
            f"  Markers: {self.n_markers}",
            f"  Marker names: {', '.join(self.marker_names[:5])}"
        ]
        if self.n_markers > 5:
            lines.append(f"                ... and {self.n_markers - 5} more")

        return "\n".join(lines)
class TrajectoryCSVLoader(BaseModel):
    """Loads 2D trajectory data from CSV files.

    Features:
    - Automatic CSV format detection (DLC 3-row, DLC 2-row, Tidy, Wide)
    - Confidence filtering
    - Load directly into TrajectoryDataset
    - Save TrajectoryDataset to tidy CSV format
    """

    encoding: str = 'utf-8'
    min_confidence: float | None = None

    def load_csv(self, *, filepath: Path) -> TrajectoryDataset:
        """Load CSV file into TrajectoryDataset with automatic format detection.

        Supports four formats:
        1. DLC: 3-row header with scorer, bodyparts, and x,y,likelihood
        2. DLC (no scorer): 2-row header with bodyparts and x,y,likelihood
        3. Tidy: frame, keypoint, x, y columns
        4. Wide: marker_x, marker_y columns

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
        elif format_type == "dlc_no_scorer":
            data = self._read_dlc_no_scorer(filepath=filepath)
        elif format_type == "tidy":
            data = self._read_tidy(filepath=filepath)
        elif format_type == "wide":
            data = self._read_wide(filepath=filepath)
        else:
            raise ValueError(f"Unknown CSV format in {filepath}")

        # Convert to TrajectoryDataset
        return self._to_trajectory_dataset(data=data)

    def save_csv(self, *, dataset: TrajectoryDataset, filepath: Path) -> None:
        """Save TrajectoryDataset to tidy CSV format.

        Output format:
            frame, keypoint, x, y
            0, marker1, 1.0, 2.0
            0, marker2, 4.0, 5.0
            ...

        Args:
            dataset: TrajectoryDataset to save
            filepath: Output CSV path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for marker_name, trajectory in dataset.data.items():
            for frame_idx, values in zip(dataset.frame_indices, trajectory.data):
                x, y = values

                rows.append({
                    "frame": int(frame_idx),
                    "keypoint": marker_name,
                    "x": float(x),
                    "y": float(y),
                })

        with open(filepath, mode='w', encoding=self.encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "keypoint", "x", "y"])
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Saved {len(rows)} rows to {filepath}")

    def _detect_format(self, *, filepath: Path) -> str:
        """Detect CSV format by inspecting header.

        Args:
            filepath: Path to CSV file

        Returns:
            Format name: "dlc", "dlc_no_scorer", "tidy", or "wide"
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

        # Check for 2-row DLC format (no scorer row)
        if len(lines) >= 2:
            row2: list[str] = lines[1].split(',')
            # Check if second row has x,y,likelihood/confidence pattern
            row2_lower = [col.strip().lower() for col in row2]
            if any(coord in ['x', 'y', 'likelihood', 'confidence'] for coord in row2_lower):
                return "dlc_no_scorer"

        # Check for Tidy format
        header_lower: list[str] = [col.lower().strip() for col in header]
        if all(col in header_lower for col in ['frame', 'keypoint', 'x', 'y']):
            return "tidy"

        # Check for Wide format
        if any(col.strip().endswith('_x') for col in header):
            return "wide"

        raise ValueError(
            f"Could not detect CSV format. Supported formats:\n"
            f"  - DLC: 3-row header with scorer, bodyparts, and x,y,likelihood\n"
            f"  - DLC (no scorer): 2-row header with bodyparts and x,y,likelihood\n"
            f"  - Tidy: columns 'frame', 'keypoint', 'x', 'y'\n"
            f"  - Wide: columns 'marker_x', 'marker_y'"
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

        # Filter valid bodyparts (must have x and y)
        valid_bodyparts = [
            bp for bp, coords in column_map.items()
            if 'x' in coords and 'y' in coords
        ]

        if not valid_bodyparts:
            raise ValueError("No valid bodyparts found with x,y coordinates")

        n_frames = len(lines) - 3

        # Parse data
        trajectories: dict[str, np.ndarray] = {}
        confidence_data: dict[str, np.ndarray] = {}

        for bodypart in valid_bodyparts:
            coords = column_map[bodypart]
            positions = np.zeros((n_frames, 2))
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

                    positions[i] = [x, y]

                except (ValueError, IndexError):
                    positions[i] = [np.nan, np.nan]

            trajectories[bodypart] = positions
            if confidence is not None:
                confidence_data[bodypart] = confidence

        return {
            "trajectories": trajectories,
            "confidence": confidence_data if confidence_data else None,
            "frame_indices": np.arange(n_frames),
        }

    def _read_dlc_no_scorer(self, *, filepath: Path) -> dict[str, Any]:
        """Read 2-row DLC format CSV (without scorer row).
        
        Format:
        - Row 0: marker/bodypart names
        - Row 1: x, y, likelihood/confidence labels
        - Row 2+: data values
        """
        with open(filepath, mode='r', encoding=self.encoding) as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError("2-row DLC CSV needs at least 3 rows (2 header + 1 data)")

        # Parse 2-row header
        bodypart_row = [col.strip() for col in lines[0].split(',')]
        coords_row = [col.strip() for col in lines[1].split(',')]

        # Build column mapping
        column_map: dict[str, dict[str, int]] = {}
        for col_idx, (bodypart, coord) in enumerate(zip(bodypart_row, coords_row)):
            if bodypart:  # Non-empty marker name
                if bodypart not in column_map:
                    column_map[bodypart] = {}
                coord_lower = coord.lower()
                # Map both 'likelihood' and 'confidence' to same key
                if coord_lower in ['likelihood', 'confidence']:
                    column_map[bodypart]['likelihood'] = col_idx
                else:
                    column_map[bodypart][coord_lower] = col_idx

        # Filter valid bodyparts (must have x and y)
        valid_bodyparts = [
            bp for bp, coords in column_map.items()
            if 'x' in coords and 'y' in coords
        ]

        if not valid_bodyparts:
            raise ValueError("No valid bodyparts found with x,y coordinates")

        n_frames = len(lines) - 2

        # Parse data
        trajectories: dict[str, np.ndarray] = {}
        confidence_data: dict[str, np.ndarray] = {}

        for bodypart in valid_bodyparts:
            coords = column_map[bodypart]
            positions = np.zeros((n_frames, 2))
            confidence = np.zeros(n_frames) if 'likelihood' in coords else None

            for i, line in enumerate(lines[2:]):
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

                    positions[i] = [x, y]

                except (ValueError, IndexError):
                    positions[i] = [np.nan, np.nan]

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

                frame_set.add(frame)

                if keypoint not in data_dict:
                    data_dict[keypoint] = {}

                data_dict[keypoint][frame] = np.array([x, y])

        # Convert to arrays
        frame_indices = np.array(sorted(frame_set))
        n_frames = len(frame_indices)

        trajectories = {}
        for keypoint, frame_data in data_dict.items():
            positions = np.full((n_frames, 2), np.nan)

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

            # Find marker names (columns ending in _x)
            marker_names = {h[:-2] for h in headers if h.endswith('_x')}

            if not marker_names:
                raise ValueError("No markers found (expected columns ending in '_x')")

            rows = list(reader)

        n_frames = len(rows)
        frame_indices = np.arange(n_frames)

        # Extract trajectories
        trajectories = {}
        for marker in marker_names:
            x_col, y_col = f"{marker}_x", f"{marker}_y"

            # Check if y column exists
            if y_col not in headers:
                logger.warning(f"Skipping marker '{marker}': missing {y_col} column")
                continue

            positions = np.zeros((n_frames, 2))

            for i, row in enumerate(rows):
                try:
                    x = float(row[x_col].strip() or 'nan')
                    y = float(row[y_col].strip() or 'nan')
                    positions[i] = [x, y]
                except (ValueError, KeyError):
                    positions[i] = [np.nan, np.nan]

            trajectories[marker] = positions

        return {
            "trajectories": trajectories,
            "confidence": None,
            "frame_indices": frame_indices,
        }

    def _to_trajectory_dataset(self, *, data: dict[str, Any]) -> TrajectoryDataset:
        """Convert parsed CSV data to TrajectoryDataset.

        Args:
            data: Dictionary with 'trajectories', 'confidence', 'frame_indices'

        Returns:
            TrajectoryDataset instance
        """
        trajectories = data["trajectories"]
        confidence = data.get("confidence")
        frame_indices = data["frame_indices"]

        # Create Trajectory2D instances
        trajectory_objects = {}
        for name, values in trajectories.items():
            conf = confidence[name] if confidence and name in confidence else None

            trajectory_objects[name] = Trajectory2D(
                name=name,
                data=values,
                confidence=conf,
                metadata={}
            )

        return TrajectoryDataset(
            data=trajectory_objects,
            frame_indices=frame_indices,
            metadata={"format": data.get("format", "unknown")}
        )


# Convenience functions
def load_trajectory_csv(
    *,
    filepath: Path,
    min_confidence: float | None = None
) -> TrajectoryDataset:
    """Load 2D trajectory CSV file.

    Args:
        filepath: Path to CSV file
        min_confidence: Optional confidence threshold

    Returns:
        TrajectoryDataset with 2D trajectories
    """
    loader = TrajectoryCSVLoader(min_confidence=min_confidence)
    return loader.load_csv(filepath=filepath)


def save_trajectory_csv(*, dataset: TrajectoryDataset, filepath: Path) -> None:
    """Save TrajectoryDataset to tidy CSV format.

    Args:
        dataset: TrajectoryDataset to save
        filepath: Output CSV path
    """
    loader = TrajectoryCSVLoader()
    loader.save_csv(dataset=dataset, filepath=filepath)
