"""Simplified CSV loader for 2D trajectory data with raw/cleaned versions.

Handles automatic CSV format detection (DLC, Tidy, Wide) for 2D positions only.
Automatically creates both raw and cleaned versions upon loading.
"""

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

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
        if self.data.ndim != 2 or self.data.shape[1] != 2:
            raise ValueError(f"Data must be (n_frames, 2), got shape {self.data.shape}")

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
        """Get mask of valid frames."""
        not_nan = ~np.isnan(self.data[:, 0])

        if self.confidence is None or min_confidence is None:
            return not_nan

        above_threshold = self.confidence >= min_confidence
        return above_threshold & not_nan

    def interpolate_missing(self, *, method: str = "linear") -> "Trajectory2D":
        """Interpolate missing (NaN) values."""
        data_interp = self.data.copy()

        for axis in range(2):
            values = self.data[:, axis]
            valid_mask = ~np.isnan(values)

            if np.sum(valid_mask) < 2:
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_values = values[valid_mask]

            interp_func = interp1d(
                valid_indices,
                valid_values,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate"
            )

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

    def apply_butterworth_filter(
        self,
        *,
        cutoff_freq: float = 6.0,
        sampling_rate: float = 30.0,
        order: int = 4
    ) -> "Trajectory2D":
        """Apply Butterworth low-pass filter to trajectory."""
        nyquist = sampling_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False)

        filtered_data = self.data.copy()

        for axis in range(2):
            values = self.data[:, axis]
            valid_mask = ~np.isnan(values)
            n_valid = np.sum(valid_mask)

            if n_valid > 2 * order:
                if not np.any(np.isnan(values)):
                    filtered_data[:, axis] = filtfilt(b=b, a=a, x=values)
                else:
                    logger.warning(f"Cannot filter '{self.name}' axis {axis}: contains NaN")

        return Trajectory2D(
            name=self.name,
            data=filtered_data,
            confidence=self.confidence,
            metadata={**self.metadata, 'filtered': True}
        )

    def create_cleaned(
        self,
        *,
        interpolation_method: str = "linear",
        butterworth_cutoff: float = 6.0,
        butterworth_sampling_rate: float = 30.0,
        butterworth_order: int = 4
    ) -> "Trajectory2D":
        """Create cleaned version: interpolate + Butterworth filter."""
        interpolated = self.interpolate_missing(method=interpolation_method)
        cleaned = interpolated.apply_butterworth_filter(
            cutoff_freq=butterworth_cutoff,
            sampling_rate=butterworth_sampling_rate,
            order=butterworth_order
        )
        return cleaned


class TrajectoryPair(ABaseModel):
    """Holds both raw and cleaned versions of a trajectory.

    Attributes:
        raw: Raw trajectory data
        cleaned: Cleaned trajectory data (interpolated + filtered)
    """

    raw: Trajectory2D
    cleaned: Trajectory2D

    @property
    def name(self) -> str:
        """Trajectory name."""
        return self.raw.name

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self.raw.n_frames


class TrajectoryDataset(ABaseModel):
    """Collection of 2D trajectories with both raw and cleaned versions.

    Access patterns:
        dataset.raw['p1']  # Raw Trajectory2D for p1
        dataset.cleaned['p1']  # Cleaned Trajectory2D for p1
        dataset.pairs['p1']  # TrajectoryPair with both versions
        dataset.pairs['p1'].raw  # Raw version
        dataset.pairs['p1'].cleaned  # Cleaned version

    Attributes:
        pairs: Dictionary mapping marker names to TrajectoryPair instances
        frame_indices: Frame numbers (may not start at 0)
        metadata: Optional dataset-level metadata
    """

    pairs: dict[str, TrajectoryPair]
    frame_indices: np.ndarray
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate(self) -> Self:
        """Validate dataset."""
        if len(self.pairs) == 0:
            raise ValueError("Dataset must contain at least one trajectory")

        n_frames_list = [pair.n_frames for pair in self.pairs.values()]
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
        return list(self.pairs.keys())

    @property
    def n_markers(self) -> int:
        """Number of markers."""
        return len(self.pairs)

    @property
    def raw(self) -> dict[str, Trajectory2D]:
        """Access raw trajectories: dataset.raw['marker_name']"""
        return {name: pair.raw for name, pair in self.pairs.items()}

    @property
    def cleaned(self) -> dict[str, Trajectory2D]:
        """Access cleaned trajectories: dataset.cleaned['marker_name']"""
        return {name: pair.cleaned for name, pair in self.pairs.items()}

    @property
    def data(self) -> dict[str, Trajectory2D]:
        """Alias for raw data (backward compatibility)."""
        return self.raw

    def to_array(self, *, marker_names: list[str] | None = None, use_cleaned: bool = False) -> np.ndarray:
        """Convert to numpy array.

        Args:
            marker_names: Optional list of markers to include (default: all)
            use_cleaned: If True, use cleaned data; otherwise use raw

        Returns:
            (n_frames, n_markers, 2) array of x,y positions
        """
        if marker_names is None:
            marker_names = self.marker_names

        missing = set(marker_names) - set(self.marker_names)
        if missing:
            raise ValueError(f"Markers not in dataset: {missing}")

        if use_cleaned:
            arrays = [self.pairs[name].cleaned.data for name in marker_names]
        else:
            arrays = [self.pairs[name].raw.data for name in marker_names]

        return np.stack(arrays, axis=1)

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

    Automatically creates both raw and cleaned versions.
    """

    encoding: str = 'utf-8'
    min_confidence: float | None = None
    butterworth_cutoff: float = 6.0
    butterworth_sampling_rate: float = 30.0
    butterworth_order: int = 4

    def load_csv(self, *, filepath: Path) -> TrajectoryDataset:
        """Load CSV file into TrajectoryDataset with both raw and cleaned versions.

        Supports four formats:
        1. DLC: 3-row header with scorer, bodyparts, and x,y,likelihood
        2. DLC (no scorer): 2-row header with bodyparts and x,y,likelihood
        3. Tidy: frame, keypoint, x, y columns
        4. Wide: marker_x, marker_y columns

        Args:
            filepath: Path to CSV file

        Returns:
            TrajectoryDataset with both raw and cleaned trajectories
        """
        if not filepath.exists():
            raise ValueError(f"File does not exist: {filepath}")

        if filepath.suffix.lower() != '.csv':
            raise ValueError(f"File must be CSV, got: {filepath.suffix}")

        format_type = self._detect_format(filepath=filepath)
        logger.info(f"Detected CSV format: {format_type}")

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

        return self._to_trajectory_dataset(data=data)

    def _detect_format(self, *, filepath: Path) -> str:
        """Detect CSV format by inspecting header."""
        with open(filepath, mode='r', encoding=self.encoding) as f:
            lines: list[str] = []
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())

            f.seek(0)
            reader = csv.reader(f)
            header: list[str] = next(reader)

        if len(lines) < 2:
            raise ValueError("CSV file has insufficient rows")

        if len(lines) >= 3:
            row3: list[str] = lines[2].split(',')
            if any(coord.strip().lower() in ['x', 'y', 'likelihood'] for coord in row3):
                return "dlc"

        if len(lines) >= 2:
            row2: list[str] = lines[1].split(',')
            row2_lower = [col.strip().lower() for col in row2]
            if any(coord in ['x', 'y', 'likelihood', 'confidence'] for coord in row2_lower):
                return "dlc_no_scorer"

        header_lower: list[str] = [col.lower().strip() for col in header]
        if all(col in header_lower for col in ['frame', 'keypoint', 'x', 'y']):
            return "tidy"

        if any(col.strip().endswith('_x') for col in header):
            return "wide"

        raise ValueError("Could not detect CSV format")

    def _read_dlc(self, *, filepath: Path) -> dict[str, Any]:
        """Read DeepLabCut format CSV."""
        with open(filepath, mode='r', encoding=self.encoding) as f:
            lines = f.readlines()

        if len(lines) < 4:
            raise ValueError("DLC CSV needs at least 4 rows")

        bodypart_row = [col.strip() for col in lines[1].split(',')]
        coords_row = [col.strip() for col in lines[2].split(',')]

        column_map: dict[str, dict[str, int]] = {}
        for col_idx, (bodypart, coord) in enumerate(zip(bodypart_row, coords_row)):
            if bodypart and bodypart.lower() != 'scorer':
                if bodypart not in column_map:
                    column_map[bodypart] = {}
                column_map[bodypart][coord] = col_idx

        valid_bodyparts = [
            bp for bp, coords in column_map.items()
            if 'x' in coords and 'y' in coords
        ]

        if not valid_bodyparts:
            raise ValueError("No valid bodyparts found")

        n_frames = len(lines) - 3

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

                    if confidence is not None:
                        conf_val = float(values[coords['likelihood']].strip())
                        confidence[i] = conf_val

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
        """Read 2-row DLC format CSV."""
        with open(filepath, mode='r', encoding=self.encoding) as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError("2-row DLC CSV needs at least 3 rows")

        bodypart_row = [col.strip() for col in lines[0].split(',')]
        coords_row = [col.strip() for col in lines[1].split(',')]

        column_map: dict[str, dict[str, int]] = {}
        for col_idx, (bodypart, coord) in enumerate(zip(bodypart_row, coords_row)):
            if bodypart:
                if bodypart not in column_map:
                    column_map[bodypart] = {}
                coord_lower = coord.lower()
                if coord_lower in ['likelihood', 'confidence']:
                    column_map[bodypart]['likelihood'] = col_idx
                else:
                    column_map[bodypart][coord_lower] = col_idx

        valid_bodyparts = [
            bp for bp, coords in column_map.items()
            if 'x' in coords and 'y' in coords
        ]

        if not valid_bodyparts:
            raise ValueError("No valid bodyparts found")

        n_frames = len(lines) - 2

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

                    if confidence is not None:
                        conf_val = float(values[coords['likelihood']].strip())
                        confidence[i] = conf_val

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

            marker_names = {h[:-2] for h in headers if h.endswith('_x')}

            if not marker_names:
                raise ValueError("No markers found")

            rows = list(reader)

        n_frames = len(rows)
        frame_indices = np.arange(n_frames)

        trajectories = {}
        for marker in marker_names:
            x_col, y_col = f"{marker}_x", f"{marker}_y"

            if y_col not in headers:
                logger.warning(f"Skipping marker '{marker}': missing {y_col}")
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
        """Convert parsed CSV data to TrajectoryDataset with both raw and cleaned versions."""
        trajectories = data["trajectories"]
        confidence = data.get("confidence")
        frame_indices = data["frame_indices"]

        pairs = {}
        for name, values in trajectories.items():
            conf = confidence[name] if confidence and name in confidence else None

            # Create raw trajectory
            raw_traj = Trajectory2D(
                name=name,
                data=values,
                confidence=conf,
                metadata={}
            )

            # Create cleaned trajectory
            cleaned_traj = raw_traj.create_cleaned(
                butterworth_cutoff=self.butterworth_cutoff,
                butterworth_sampling_rate=self.butterworth_sampling_rate,
                butterworth_order=self.butterworth_order
            )

            pairs[name] = TrajectoryPair(raw=raw_traj, cleaned=cleaned_traj)

        logger.info(f"Created dataset with {len(pairs)} trajectory pairs (raw + cleaned)")

        return TrajectoryDataset(
            pairs=pairs,
            frame_indices=frame_indices,
            metadata={"format": data.get("format", "unknown")}
        )


def load_trajectory_csv(
    *,
    filepath: Path,
    min_confidence: float | None = None,
    butterworth_cutoff: float = 6.0,
    butterworth_sampling_rate: float = 30.0
) -> TrajectoryDataset:
    """Load 2D trajectory CSV file with both raw and cleaned versions.

    Args:
        filepath: Path to CSV file
        min_confidence: Optional confidence threshold
        butterworth_cutoff: Butterworth filter cutoff frequency (Hz)
        butterworth_sampling_rate: Video sampling rate (Hz)

    Returns:
        TrajectoryDataset with both raw and cleaned trajectories
    """
    loader = TrajectoryCSVLoader(
        min_confidence=min_confidence,
        butterworth_cutoff=butterworth_cutoff,
        butterworth_sampling_rate=butterworth_sampling_rate
    )
    return loader.load_csv(filepath=filepath)