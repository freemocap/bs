"""Base data structures for all SkellySolver pipelines.

This module provides fundamental data structures used across all pipelines:
- TrajectoryType: Enum defining the mathematical domain of trajectory data
- TrajectoryND: N-dimensional trajectories over time with domain specification
- TrajectoryDataset: Collection of trajectories with metadata

These replace scattered data handling across multiple files.
"""

import logging
from enum import Enum
from typing import Any

import numpy as np
from pydantic import model_validator, Field, BaseModel, ConfigDict
from typing_extensions import Self
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

class ABaseModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

logger = logging.getLogger(__name__)


class TrajectoryType(str, Enum):
    """Mathematical domain of trajectory data.

    Defines how data should be interpolated, blended, and validated.

    Attributes:
        SCALAR: Single scalar values (e.g., joint angles, confidences)
        POSITION_2D: 2D positions in Euclidean space (x, y)
        POSITION_3D: 3D positions in Euclidean space (x, y, z)
        ROTATION_MATRIX: SO(3) rotations as flattened 3x3 matrices (9 values)
        QUATERNION: SO(3) rotations as unit quaternions (w, x, y, z) - 4 values
        ROTATION_VECTOR: SO(3) rotations as axis-angle representation (3 values)
        SE3_MATRIX: SE(3) rigid transforms as flattened 4x4 matrices (16 values)
        SE3_RT: SE(3) as rotation matrix + translation (12 values: 9 for R, 3 for t)
        GENERIC: Arbitrary N-dimensional data with linear interpolation
    """
    SCALAR = "scalar"
    POSITION_3D = "position_3d"
    POSITION_2D = "position_2d"
    ROTATION_MATRIX = "rotation_matrix"
    QUATERNION = "quaternion"
    ROTATION_VECTOR = "rotation_vector"
    SE3_MATRIX = "se3_matrix"
    SE3_RT = "se3_rt"
    GENERIC = "generic"

    def expected_dims(self) -> int | None:
        """Expected number of dimensions for this trajectory type.

        Returns:
            Expected dimension count, or None if variable
        """
        dim_map = {
            TrajectoryType.SCALAR: 1,
            TrajectoryType.POSITION_3D: 3,
            TrajectoryType.POSITION_2D: 2,
            TrajectoryType.ROTATION_MATRIX: 9,
            TrajectoryType.QUATERNION: 4,
            TrajectoryType.ROTATION_VECTOR: 3,
            TrajectoryType.SE3_MATRIX: 16,
            TrajectoryType.SE3_RT: 12,
            TrajectoryType.GENERIC: None,
        }
        return dim_map[self]

    def column_names(self) -> list[str] | None:
        if self == TrajectoryType.SCALAR:
            return ["value"]
        elif self == TrajectoryType.POSITION_2D:
            return ["x", "y"]
        elif self == TrajectoryType.POSITION_3D:
            return ["x", "y", "z"]
        elif self == TrajectoryType.ROTATION_MATRIX:
            return [f"r{i}{j}" for i in range(1, 4) for j in range(1, 4)]
        elif self == TrajectoryType.QUATERNION:
            return ["w", "x", "y", "z"]
        elif self == TrajectoryType.ROTATION_VECTOR:
            return ["rx", "ry", "rz"]
        elif self == TrajectoryType.SE3_MATRIX:
            return [f"t{i}{j}" for i in range(1, 5) for j in range(1, 5)]
        elif self == TrajectoryType.SE3_RT:
            return [f"r{i}{j}" for i in range(1, 4) for j in range(1, 4)] + ["tx", "ty", "tz"]
        else:
            return None


    def requires_manifold_blending(self) -> bool:
        """Whether this type requires manifold-aware blending (not linear).

        Returns:
            True if SLERP or other manifold interpolation is needed
        """
        return self in {
            TrajectoryType.ROTATION_MATRIX,
            TrajectoryType.QUATERNION,
            TrajectoryType.ROTATION_VECTOR,
            TrajectoryType.SE3_MATRIX,
            TrajectoryType.SE3_RT,
        }


class TrajectoryND(ABaseModel):
    """An N-dimensional vector over time with domain specification.

    Attributes:
        name: Identifier for this trajectory
        data: (n_frames, n_dims) values over time
        trajectory_type: Mathematical domain of the data
        confidence: Optional (n_frames,) confidence scores [0-1]
        metadata: Optional additional data
    """

    name: str
    data: np.ndarray
    trajectory_type: TrajectoryType = TrajectoryType.GENERIC
    confidence: np.ndarray | None = None
    column_names: list[str] | None = None
    dimension_names: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate(self) -> Self:
        """Validate trajectory data."""
        # Check confidence length matches values
        if self.confidence is not None:
            if len(self.confidence) != len(self.data):
                raise ValueError(
                    f"Confidence length {len(self.confidence)} != values length {len(self.data)}"
                )

        # Check dimensions match trajectory type
        expected_dims = self.trajectory_type.expected_dims()
        if expected_dims is not None and self.n_dims != expected_dims:
            raise ValueError(
                f"Trajectory type {self.trajectory_type.value} expects {expected_dims} dimensions, "
                f"but got {self.n_dims}"
            )

        # Validate quaternions are normalized
        if self.trajectory_type == TrajectoryType.QUATERNION:
            norms = np.linalg.norm(self.data, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                logger.warning(
                    f"Quaternion trajectory '{self.name}' contains non-normalized quaternions. "
                    f"Norms range: [{norms.min():.6f}, {norms.max():.6f}]"
                )

        # Set default column names if not provided
        if self.column_names is None:
            self.column_names = self.trajectory_type.column_names()
            if self.column_names is None:
                self.column_names = [f"dim_{i}" for i in range(self.n_dims)]

        return self

    @property
    def n_frames(self) -> int:
        """Number of frames in trajectory."""
        return len(self.data)

    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return self.data.shape[1] if len(self.data.shape) > 1 else 1

    def is_valid(self, *, min_confidence: float = None) -> np.ndarray:
        """Get mask of valid frames.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            (n_frames,) boolean mask
        """
        if self.confidence is None:
            # If no confidence, check for NaN
            return ~np.isnan(self.data[:, 0])

        # Valid if confidence above threshold AND not NaN
        above_threshold = self.confidence >= min_confidence
        not_nan = ~np.isnan(self.data[:, 0])
        return above_threshold & not_nan

    def get_valid_values(self, *, min_confidence: float = None) -> np.ndarray:
        """Get values for valid frames only.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            (n_valid, n_dims) valid values
        """
        mask = self.is_valid(min_confidence=min_confidence)
        return self.data[mask]

    def get_centroid(self, *, min_confidence: float = None) -> np.ndarray:
        """Compute centroid of valid values.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            (n_dims,) centroid position
        """
        valid_pos = self.get_valid_values(min_confidence=min_confidence)
        if len(valid_pos) == 0:
            return np.zeros(self.n_dims)
        return np.mean(valid_pos, axis=0)

    def interpolate_missing(self, *, method: str = "linear") -> "TrajectoryND":
        """Interpolate missing (NaN) values.

        Args:
            method: Interpolation method ("linear", "cubic")

        Returns:
            New TrajectoryND with interpolated values
        """
        values_interp = self.data.copy()

        for axis in range(self.n_dims):
            data = self.data[:, axis]
            valid_mask = ~np.isnan(data)

            if np.sum(valid_mask) < 2:
                # Not enough points to interpolate
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_values = data[valid_mask]

            # Interpolate
            interp_func = interp1d(
                valid_indices,
                valid_values,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate"
            )

            # Fill missing values
            missing_mask = np.isnan(data)
            if np.any(missing_mask):
                missing_indices = np.where(missing_mask)[0]
                values_interp[missing_indices, axis] = interp_func(missing_indices)

        return TrajectoryND(
            name=self.name,
            data=values_interp,
            trajectory_type=self.trajectory_type,
            confidence=self.confidence,
            metadata=self.metadata
        )

    def __str__(self) -> str:
        """Human-readable trajectory summary."""
        n_valid = np.sum(self.is_valid())
        validity_pct = (n_valid / self.n_frames * 100) if self.n_frames > 0 else 0

        has_conf = "Yes" if self.confidence is not None else "No"

        lines = [
            f"Trajectory '{self.name}':",
            f"  Type:       {self.trajectory_type.value}",
            f"  Frames:     {self.n_frames}",
            f"  Dimensions: {self.n_dims}",
            f"  Valid:      {n_valid}/{self.n_frames} ({validity_pct:.1f}%)",
            f"  Confidence: {has_conf}"
        ]
        return "\n".join(lines)


class TrajectoryDataset(ABaseModel):
    """Collection of trajectories or observations with metadata.

    Can contain N-D trajectories (TrajectoryND).
    Used by all pipelines to manage multiple markers/points.

    Attributes:
        data: Dictionary mapping names to TrajectoryND instances
        frame_indices: Frame numbers (may not start at 0)
        metadata: Optional dataset-level metadata
    """

    data: dict[str, TrajectoryND]
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
        """Get list of marker/point names."""
        return list(self.data.keys())

    @property
    def n_markers(self) -> int:
        """Number of markers/points."""
        return len(self.data)

    @property
    def n_dims(self) -> int:
        """Number of dimensions (assumes all trajectories have same dims)."""
        if not self.data:
            return 0
        first_traj = next(iter(self.data.values()))
        return first_traj.n_dims

    def slice_frames(self, *, start_frame: int, end_frame: int) -> Self:
        """Extract a slice of frames from the dataset.

        Creates COPIES, not views, to ensure independence.
        """
        sliced_data = {}
        for name, traj in self.data.items():
            sliced_values = traj.data[start_frame:end_frame].copy()
            sliced_confidence = (
                traj.confidence[start_frame:end_frame].copy()
                if traj.confidence is not None
                else None
            )

            sliced_data[name] = TrajectoryND(
                name=name,
                data=sliced_values,
                trajectory_type=traj.trajectory_type,
                confidence=sliced_confidence,
                metadata=traj.metadata
            )

        return TrajectoryDataset(
            data=sliced_data,
            frame_indices=self.frame_indices[start_frame:end_frame].copy(),
            metadata=self.metadata
        )

    def to_array(
        self,
        *,
        marker_names: list[str] | None = None,
        fill_missing: bool = False
    ) -> np.ndarray:
        """Convert to numpy array.

        Args:
            marker_names: Optional list of markers to include (default: all)
            fill_missing: If True, fill NaN values with zeros

        Returns:
            (n_frames, n_markers, n_dims) array
        """
        if marker_names is None:
            marker_names = self.marker_names

        # Check all requested markers exist
        missing = set(marker_names) - set(self.marker_names)
        if missing:
            raise ValueError(f"Markers not in dataset: {missing}")

        # Stack values
        arrays = [self.data[name].data for name in marker_names]
        result = np.stack(arrays, axis=1)

        if fill_missing:
            result = np.nan_to_num(result, nan=0.0)

        return result

    def get_valid_frames(
        self,
        *,
        min_confidence: float = 0.3,
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
        min_confidence: float = 0.3,
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

            filtered_data[name] = TrajectoryND(
                name=name,
                data=filtered_values,
                trajectory_type=traj.trajectory_type,
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
            method: Interpolation method

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

    def get_centroid(self, *, min_confidence: float = 0.3) -> np.ndarray:
        """Compute centroid across all markers over time.

        Args:
            min_confidence: Minimum confidence for valid data

        Returns:
            (n_frames, n_dims) centroid trajectory
        """
        # Get all valid values
        all_values = []
        for traj in self.data.values():
            valid_mask = traj.is_valid(min_confidence=min_confidence)
            values_copy = traj.data.copy()
            values_copy[~valid_mask] = np.nan
            all_values.append(values_copy)

        # Stack and compute mean ignoring NaN
        stacked = np.stack(all_values, axis=1)  # (n_frames, n_markers, n_dims)
        centroids = np.nanmean(stacked, axis=1)  # (n_frames, n_dims)

        return centroids

    @classmethod
    def stitch_with_blending(
        cls,
        *,
        datasets: list["TrajectoryDataset"],
        chunk_ranges: list[tuple[int, int]],
        overlap_size: int,
        blend_window: int
    ) -> "TrajectoryDataset":
        """Stitch multiple datasets together with domain-aware blending.

        Uses trajectory type to determine appropriate blending method:
        - ROTATION_MATRIX, QUATERNION, ROTATION_VECTOR: SLERP
        - SE3_*: Special handling for rigid transforms
        - Others: Linear interpolation

        Args:
            datasets: List of trajectory datasets to stitch
            chunk_ranges: List of (start, end) frame ranges for each chunk
            overlap_size: Number of overlapping frames between chunks
            blend_window: Size of blending window within overlap

        Returns:
            Complete stitched dataset
        """
        if not datasets:
            raise ValueError("Cannot stitch empty dataset list")

        if len(datasets) != len(chunk_ranges):
            raise ValueError(
                f"Number of datasets ({len(datasets)}) must match "
                f"number of chunk ranges ({len(chunk_ranges)})"
            )

        # Get dimensions from first dataset
        first_dataset = datasets[0]
        marker_names = first_dataset.marker_names

        # Determine total number of frames
        total_frames = max(end for _, end in chunk_ranges)

        # Initialize output arrays for each marker
        stitched_data = {}
        for marker_name in marker_names:
            # Get trajectory type from first dataset
            traj_type = first_dataset.data[marker_name].trajectory_type
            n_dims = first_dataset.data[marker_name].n_dims

            stitched_values = np.zeros((total_frames, n_dims))

            # Stitch this marker across all chunks
            for chunk_idx, dataset in enumerate(datasets):
                chunk_start, chunk_end = chunk_ranges[chunk_idx]
                chunk_traj = dataset.data[marker_name]

                if chunk_idx == 0:
                    # First chunk: copy directly (no blending)
                    blend_end = chunk_end - overlap_size if chunk_idx < len(datasets) - 1 else chunk_end
                    stitched_values[chunk_start:blend_end] = chunk_traj.data[:blend_end - chunk_start]

                    logger.debug(f"Marker {marker_name}, Chunk 0: Copied frames {chunk_start}-{blend_end}")

                else:
                    # Subsequent chunks: blend overlap region
                    prev_dataset = datasets[chunk_idx - 1]
                    prev_start, prev_end = chunk_ranges[chunk_idx - 1]
                    prev_traj = prev_dataset.data[marker_name]

                    overlap_start = chunk_start
                    overlap_end = min(chunk_start + overlap_size, chunk_end)
                    blend_size = min(blend_window, overlap_end - overlap_start)

                    blend_global_start = overlap_start
                    blend_global_end = overlap_start + blend_size

                    # Extract data from both chunks
                    prev_local_start = blend_global_start - prev_start
                    prev_local_end = blend_global_end - prev_start
                    curr_local_start = blend_global_start - chunk_start
                    curr_local_end = blend_global_end - chunk_start

                    prev_values = prev_traj.data[prev_local_start:prev_local_end]
                    curr_values = chunk_traj.data[curr_local_start:curr_local_end]

                    # Blend using trajectory type-specific method
                    blended = cls._blend_by_type(
                        data1=prev_values,
                        data2=curr_values,
                        trajectory_type=traj_type,
                        blend_size=blend_size
                    )

                    stitched_values[blend_global_start:blend_global_end] = blended

                    logger.debug(
                        f"Marker {marker_name} ({traj_type.value}), Chunk {chunk_idx}: "
                        f"Blended frames {blend_global_start}-{blend_global_end}"
                    )

                    # Copy non-overlapping region
                    copy_start = blend_global_end
                    copy_end = chunk_end - (overlap_size if chunk_idx < len(datasets) - 1 else 0)

                    if copy_start < copy_end:
                        local_copy_start = copy_start - chunk_start
                        local_copy_end = copy_end - chunk_start

                        stitched_values[copy_start:copy_end] = (
                            chunk_traj.data[local_copy_start:local_copy_end]
                        )

                        logger.debug(
                            f"Marker {marker_name}, Chunk {chunk_idx}: "
                            f"Copied frames {copy_start}-{copy_end}"
                        )

            # Create stitched trajectory for this marker
            stitched_data[marker_name] = TrajectoryND(
                name=marker_name,
                data=stitched_values,
                trajectory_type=traj_type,
                confidence=None,  # Confidence not preserved during stitching
                metadata=first_dataset.data[marker_name].metadata
            )

        logger.info(f"Stitching complete: {total_frames} frames")

        return TrajectoryDataset(
            data=stitched_data,
            frame_indices=np.arange(total_frames),
            metadata=first_dataset.metadata
        )

    @staticmethod
    def _blend_by_type(
        *,
        data1: np.ndarray,
        data2: np.ndarray,
        trajectory_type: TrajectoryType,
        blend_size: int
    ) -> np.ndarray:
        """Blend data using trajectory type-appropriate method.

        Args:
            data1: First dataset values
            data2: Second dataset values
            trajectory_type: Type of trajectory data
            blend_size: Size of blend region

        Returns:
            Blended values
        """
        if trajectory_type == TrajectoryType.ROTATION_MATRIX:
            return TrajectoryDataset._blend_rotation_matrices(
                R1=data1.reshape(blend_size, 3, 3),
                R2=data2.reshape(blend_size, 3, 3),
                blend_size=blend_size
            ).reshape(blend_size, 9)

        elif trajectory_type == TrajectoryType.QUATERNION:
            return TrajectoryDataset._blend_quaternions(
                q1=data1,
                q2=data2,
                blend_size=blend_size
            )

        elif trajectory_type == TrajectoryType.ROTATION_VECTOR:
            return TrajectoryDataset._blend_rotation_vectors(
                rv1=data1,
                rv2=data2,
                blend_size=blend_size
            )

        elif trajectory_type == TrajectoryType.SE3_RT:
            # SE(3) as [R (9), t (3)]
            R1 = data1[:, :9].reshape(blend_size, 3, 3)
            R2 = data2[:, :9].reshape(blend_size, 3, 3)
            t1 = data1[:, 9:12]
            t2 = data2[:, 9:12]

            R_blended = TrajectoryDataset._blend_rotation_matrices(
                R1=R1,
                R2=R2,
                blend_size=blend_size
            )
            t_blended = TrajectoryDataset._blend_linear(
                data1=t1,
                data2=t2,
                blend_size=blend_size
            )

            return np.concatenate([R_blended.reshape(blend_size, 9), t_blended], axis=1)

        elif trajectory_type == TrajectoryType.SE3_MATRIX:
            # SE(3) as flattened 4x4 matrix
            T1 = data1.reshape(blend_size, 4, 4)
            T2 = data2.reshape(blend_size, 4, 4)

            # Extract rotation and translation
            R1 = T1[:, :3, :3]
            R2 = T2[:, :3, :3]
            t1 = T1[:, :3, 3]
            t2 = T2[:, :3, 3]

            R_blended = TrajectoryDataset._blend_rotation_matrices(
                R1=R1,
                R2=R2,
                blend_size=blend_size
            )
            t_blended = TrajectoryDataset._blend_linear(
                data1=t1,
                data2=t2,
                blend_size=blend_size
            )

            # Reconstruct 4x4 matrices
            T_blended = np.zeros((blend_size, 4, 4))
            T_blended[:, :3, :3] = R_blended
            T_blended[:, :3, 3] = t_blended
            T_blended[:, 3, 3] = 1.0

            return T_blended.reshape(blend_size, 16)

        else:
            # Default: linear interpolation for SCALAR, POSITION, GENERIC
            return TrajectoryDataset._blend_linear(
                data1=data1,
                data2=data2,
                blend_size=blend_size
            )

    @staticmethod
    def _create_blend_weights(*, blend_size: int) -> np.ndarray:
        """Create smooth cosine blending weights from 0 to 1.

        Args:
            blend_size: Number of frames in blend region

        Returns:
            (blend_size,) weights transitioning from 0 to 1
        """
        t = np.linspace(start=0.0, stop=1.0, num=blend_size)
        return (1.0 - np.cos(t * np.pi)) / 2.0

    @staticmethod
    def _blend_linear(
        *,
        data1: np.ndarray,
        data2: np.ndarray,
        blend_size: int
    ) -> np.ndarray:
        """Blend data using linear interpolation.

        Args:
            data1: First dataset values
            data2: Second dataset values
            blend_size: Size of blend region

        Returns:
            Blended values
        """
        weights = TrajectoryDataset._create_blend_weights(blend_size=blend_size)
        weights_expanded = weights[:, np.newaxis]
        return (1.0 - weights_expanded) * data1 + weights_expanded * data2

    @staticmethod
    def _blend_rotation_matrices(
        *,
        R1: np.ndarray,
        R2: np.ndarray,
        blend_size: int
    ) -> np.ndarray:
        """Blend rotation matrices using SLERP on SO(3).

        Args:
            R1: (blend_size, 3, 3) rotation matrices from chunk 1
            R2: (blend_size, 3, 3) rotation matrices from chunk 2
            blend_size: Size of blend region

        Returns:
            (blend_size, 3, 3) blended rotations
        """
        weights = TrajectoryDataset._create_blend_weights(blend_size=blend_size)
        blended = np.zeros((blend_size, 3, 3))

        for i in range(blend_size):
            if weights[i] <= 0.0:
                blended[i] = R1[i]
            elif weights[i] >= 1.0:
                blended[i] = R2[i]
            else:
                # Convert to quaternions for SLERP
                q1 = Rotation.from_matrix(matrix=R1[i]).as_quat()
                q2 = Rotation.from_matrix(matrix=R2[i]).as_quat()

                # Ensure shortest path
                if np.dot(q1, q2) < 0:
                    q2 = -q2

                # Spherical linear interpolation
                rot_interp = Slerp(
                    times=np.array([0.0, 1.0]),
                    rotations=Rotation.from_quat(quat=[q1, q2])
                )
                blended[i] = rot_interp(times=weights[i]).as_matrix()

        return blended

    @staticmethod
    def _blend_quaternions(
        *,
        q1: np.ndarray,
        q2: np.ndarray,
        blend_size: int
    ) -> np.ndarray:
        """Blend unit quaternions using SLERP on SO(3).

        Args:
            q1: (blend_size, 4) quaternions from chunk 1
            q2: (blend_size, 4) quaternions from chunk 2
            blend_size: Size of blend region

        Returns:
            (blend_size, 4) blended quaternions
        """
        weights = TrajectoryDataset._create_blend_weights(blend_size=blend_size)
        blended = np.zeros((blend_size, 4))

        for i in range(blend_size):
            if weights[i] <= 0.0:
                blended[i] = q1[i]
            elif weights[i] >= 1.0:
                blended[i] = q2[i]
            else:
                # Ensure shortest path
                q1_norm = q1[i] / np.linalg.norm(q1[i])
                q2_norm = q2[i] / np.linalg.norm(q2[i])

                if np.dot(q1_norm, q2_norm) < 0:
                    q2_norm = -q2_norm

                # SLERP
                rot_interp = Slerp(
                    times=np.array([0.0, 1.0]),
                    rotations=Rotation.from_quat(quat=[q1_norm, q2_norm])
                )
                blended[i] = rot_interp(times=weights[i]).as_quat()

        return blended

    @staticmethod
    def _blend_rotation_vectors(
        *,
        rv1: np.ndarray,
        rv2: np.ndarray,
        blend_size: int
    ) -> np.ndarray:
        """Blend rotation vectors (axis-angle) using SLERP on SO(3).

        Args:
            rv1: (blend_size, 3) rotation vectors from chunk 1
            rv2: (blend_size, 3) rotation vectors from chunk 2
            blend_size: Size of blend region

        Returns:
            (blend_size, 3) blended rotation vectors
        """
        weights = TrajectoryDataset._create_blend_weights(blend_size=blend_size)
        blended = np.zeros((blend_size, 3))

        for i in range(blend_size):
            if weights[i] <= 0.0:
                blended[i] = rv1[i]
            elif weights[i] >= 1.0:
                blended[i] = rv2[i]
            else:
                # Convert to quaternions for SLERP
                rot1 = Rotation.from_rotvec(rotvec=rv1[i])
                rot2 = Rotation.from_rotvec(rotvec=rv2[i])

                q1 = rot1.as_quat()
                q2 = rot2.as_quat()

                # Ensure shortest path
                if np.dot(q1, q2) < 0:
                    q2 = -q2
                    rot2 = Rotation.from_quat(quat=q2)

                # SLERP
                rot_interp = Slerp(
                    times=np.array([0.0, 1.0]),
                    rotations=Rotation.from_quat(quat=[q1, q2])
                )
                blended[i] = rot_interp(times=weights[i]).as_rotvec()

        return blended

    def get_summary(self) -> dict[str, Any]:
        """Get dataset summary statistics.

        Returns:
            Dictionary with summary information
        """
        summary = {
            "n_frames": self.n_frames,
            "n_markers": self.n_markers,
            "n_dims": self.n_dims,
            "marker_names": self.marker_names,
        }

        # Compute validity statistics
        if self.data:
            first_traj = next(iter(self.data.values()))
            if first_traj.confidence is not None:
                all_valid = self.get_valid_frames(
                    min_confidence=0.3,
                    min_valid_markers=self.n_markers
                )
                summary["n_fully_valid_frames"] = int(np.sum(all_valid))
                summary["percent_fully_valid"] = float(np.mean(all_valid) * 100)

                # Per-marker validity
                marker_validity = {}
                for name, traj in self.data.items():
                    n_valid = np.sum(traj.is_valid(min_confidence=0.3))
                    marker_validity[name] = {
                        "n_valid": int(n_valid),
                        "percent_valid": float(n_valid / traj.n_frames * 100)
                    }
                summary["marker_validity"] = marker_validity

        return summary

    def __str__(self) -> str:
        """Human-readable dataset summary."""
        lines = [
            f"TrajectoryDataset:",
            f"  Frames:  {self.n_frames}",
            f"  Markers: {self.n_markers}",
            f"  Dims:    {self.n_dims}",
        ]

        # Add validity info if confidence available
        summary = self.get_summary()
        if "percent_fully_valid" in summary:
            lines.append(f"  Valid:   {summary['percent_fully_valid']:.1f}% of frames")

        # List markers
        lines.append(f"  Marker names: {', '.join(self.marker_names[:5])}")
        if self.n_markers > 5:
            lines.append(f"                ... and {self.n_markers - 5} more")

        return "\n".join(lines)