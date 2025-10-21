import logging
from functools import cached_property
from typing import Any, Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from python_code.eye_analysis.data_models.abase_model import FrozenABaseModel

DEFAULT_MIN_CONFIDENCE: float = 0.3
DEFAULT_BUTTERWORTH_CUTOFF: float = 6.0
DEFAULT_BUTTERWORTH_ORDER: int = 4

logger = logging.getLogger(__name__)


class Trajectory2D(FrozenABaseModel):

    name: str
    data: np.ndarray  # shape: (n_frames, 2)
    timestamps: np.ndarray
    confidence: np.ndarray
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate(self) -> Self:
        """Validate trajectory data."""
        if self.data.ndim != 2 or self.data.shape[1] != 2:
            raise ValueError(f"Data must be (n_frames, 2), got shape {self.data.shape}")

        if len(self.confidence) != len(self.data):
            raise ValueError(
                f"Confidence length {len(self.confidence)} != data length {len(self.data)}"
            )

        if len(self.timestamps) != len(self.data):
            raise ValueError(
                f"Timestamps length {len(self.timestamps)} != data length {len(self.data)}"
            )
        if not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("Timestamps must be strictly increasing")

        return self

    @cached_property
    def framerate(self) -> float:
        """Estimated framerate from timestamps."""
        if self.timestamps is None:
            raise ValueError("Timestamps are required to compute framerate")

        diffs = np.diff(self.timestamps)
        median_diff = np.median(diffs)

        if median_diff <= 0:
            raise ValueError(
                "Timestamps must be strictly increasing to compute framerate"
            )

        return 1.0 / float(median_diff)

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
                fill_value="extrapolate",
            )

            missing_mask = np.isnan(values)
            if np.any(missing_mask):
                missing_indices = np.where(missing_mask)[0]
                data_interp[missing_indices, axis] = interp_func(missing_indices)

        return Trajectory2D(
            name=self.name,
            data=data_interp,
            confidence=self.confidence,
            timestamps=self.timestamps,
            metadata=self.metadata,
        )

    def apply_butterworth_filter(
        self, *, cutoff_freq: float = 6.0, order: int = 4
    ) -> "Trajectory2D":
        """Apply Butterworth low-pass filter to trajectory."""
        nyquist = self.framerate / 2.0
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(N=order, Wn=normal_cutoff, btype="low", analog=False)

        filtered_data = self.data.copy()

        for axis in range(2):
            values = self.data[:, axis]
            valid_mask = ~np.isnan(values)
            n_valid = np.sum(valid_mask)

            if n_valid > 2 * order:
                if not np.any(np.isnan(values)):
                    filtered_data[:, axis] = filtfilt(b=b, a=a, x=values)
                else:
                    logger.warning(
                        f"Cannot filter '{self.name}' axis {axis}: contains NaN"
                    )

        return Trajectory2D(
            name=self.name,
            data=filtered_data,
            confidence=self.confidence,
            timestamps=self.timestamps,
            metadata={**self.metadata, "filtered": True},
        )

    def create_cleaned(
        self,
        *,
        interpolation_method: str = "linear",
        butterworth_cutoff: float = 6.0,
        butterworth_order: int = 4,
    ) -> "Trajectory2D":
        """Create cleaned version: interpolate + Butterworth filter."""
        interpolated = self.interpolate_missing(method=interpolation_method)
        return interpolated.apply_butterworth_filter(
            cutoff_freq=butterworth_cutoff, order=butterworth_order
        )


class ProcessedTrajectory(FrozenABaseModel):
    """Holds both raw and cleaned versions of a trajectory.

    Attributes:
        raw: Raw trajectory data
        cleaned: Interpolated + filtered trajectory data
        aligned: Anatomically aligned trajectory data (analyzable output)
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

    @classmethod
    def create(
        cls,
        *,
        name: str,
        data: np.ndarray,
        confidence: np.ndarray,
        timestamps: list[float],
        butterworth_cutoff: float,
        butterworth_order: int,
        metadata: dict[str, Any] | None = None,
    ):
        """Convert parsed CSV data to TrajectoryDataset with both raw and cleaned versions."""

        raw_traj = Trajectory2D(
            name=name,
            data=data,
            confidence=confidence,
            timestamps=np.array(timestamps),
            metadata=metadata or {},
        )

        # Create cleaned trajectory
        cleaned_traj = raw_traj.create_cleaned(
            butterworth_cutoff=butterworth_cutoff, butterworth_order=butterworth_order
        )

        return cls(
            raw=raw_traj,
            cleaned=cleaned_traj,
        )


class TrajectoryDataset(FrozenABaseModel):

    name: str
    trajectories: dict[str, ProcessedTrajectory]
    frame_indices: np.ndarray
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        trajectories: dict[str, np.ndarray],
        confidence: dict[str, np.ndarray],
        frame_indices: np.ndarray,
        timestamps: list[float] | np.ndarray,
        butterworth_cutoff: float,
        butterworth_order: int,
        metadata: dict[str, Any] | None = None,
    ) -> "TrajectoryDataset":

        pairs = {}
        for trajectory_name, values in trajectories.items():

            pairs[trajectory_name] = ProcessedTrajectory.create(
                name=trajectory_name,
                data=values,
                timestamps=timestamps,
                butterworth_cutoff=butterworth_cutoff,
                butterworth_order=butterworth_order,
                confidence=confidence[trajectory_name],
            )

        logger.info(
            f"Created dataset with {len(pairs)} trajectory pairs (raw + cleaned)"
        )

        return cls(
            name=dataset_name,
            trajectories=pairs,
            frame_indices=frame_indices,
            metadata=metadata or {},
        )

    @model_validator(mode="after")
    def validate(self) -> Self:
        """Validate dataset."""
        if len(self.trajectories) == 0:
            raise ValueError("Dataset must contain at least one trajectory")

        n_frames_list = [pair.n_frames for pair in self.trajectories.values()]
        if len(set(n_frames_list)) > 1:
            raise ValueError(
                f"All trajectories must have same length, got {n_frames_list}"
            )

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
        return list(self.trajectories.keys())

    @property
    def n_markers(self) -> int:
        """Number of markers."""
        return len(self.trajectories)

    @property
    def raw(self) -> dict[str, Trajectory2D]:
        """Access raw trajectories: dataset.raw['marker_name']"""
        return {name: pair.raw for name, pair in self.trajectories.items()}

    @property
    def cleaned(self) -> dict[str, Trajectory2D]:
        """Access cleaned trajectories: dataset.cleaned['marker_name']"""
        return {name: pair.cleaned for name, pair in self.trajectories.items()}

    def to_array(
        self, *, marker_names: list[str] | None = None, use_cleaned: bool = True
    ) -> np.ndarray:
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
            arrays = [self.trajectories[name].cleaned.data for name in marker_names]
        else:
            arrays = [self.trajectories[name].raw.data for name in marker_names]

        return np.stack(arrays, axis=1)

    def to_tidy_dataset(self, eye_name: str | None = None, include_raw: bool = True) -> pd.DataFrame:
        """Convert to tidy dataset."""
        dataframes = []

        for marker, trajectory_dataset in self.trajectories.items():
            data_types = [("cleaned", trajectory_dataset.cleaned)]
            if include_raw:
                data_types.append(("raw", trajectory_dataset.raw))
            for processing_level, trajectory in data_types:
                df = self._construct_marker_dataframe(eye_name, marker, processing_level, trajectory)
                dataframes.append(df)

        return pd.concat(dataframes).sort_values(["frame", "keypoint"])

    def _construct_marker_dataframe(
        self,
        eye_name: str | None,
        marker: str,
        processing_level: str,
        trajectory: Trajectory2D,
    ):
        df = pd.DataFrame()
        df["frame"] = self.frame_indices
        df["keypoint"] = np.array([marker] * self.n_frames)
        if eye_name is not None:
            df["eye"] = np.array([eye_name] * self.n_frames)
        df["timestamp"] = trajectory.timestamps
        df["x"] = trajectory.data[:, 0]
        df["y"] = trajectory.data[:, 1]
        df["processing_level"] = processing_level
        return df

    def get_frame_points(
        self, *, frame_idx: int, include_raw: bool = True, include_cleaned: bool = True
    ) -> dict[str, dict[str, np.ndarray]]:

        if frame_idx < 0 or frame_idx >= self.n_frames:
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {self.n_frames})"
            )

        points: dict[str, dict[str, np.ndarray]] = {}

        for trajectory_name, trajectory in self.trajectories.items():

            if include_raw:
                if "raw" not in points:
                    points["raw"] = {}
                points["raw"][trajectory_name] = trajectory.raw.data[frame_idx]
            if include_cleaned:
                if "cleaned" not in points:
                    points["cleaned"] = {}
                points["cleaned"][trajectory_name] = trajectory.cleaned.data[frame_idx]

        return points

    def __str__(self) -> str:
        """Human-readable dataset summary."""
        lines = [
            f"TrajectoryDataset:",
            f"  Frames:  {self.n_frames}",
            f"  Markers: {self.n_markers}",
            f"  Marker names: {', '.join(self.marker_names[:5])}",
        ]
        if self.n_markers > 5:
            lines.append(f"                ... and {self.n_markers - 5} more")

        return "\n".join(lines)
