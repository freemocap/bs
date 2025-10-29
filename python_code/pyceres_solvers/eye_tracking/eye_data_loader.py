"""Load and preprocess eye tracking data from CSV."""

import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, computed_field
from numpydantic import NDArray, Shape
from typing import Any


class EyeTrackingData(BaseModel):
    """Loaded eye tracking observations."""

    model_config = {"arbitrary_types_allowed": True}

    pupil_points_px: NDArray[Shape["*, 8, 2"], float]
    tear_ducts_px: NDArray[Shape["*, 2"], float]
    frame_indices: NDArray[Shape["* frames_number"], int]
    n_valid_points: NDArray[Shape["* frame_number"], int]

    @computed_field
    @property
    def n_frames(self) -> int:
        """Number of frames in dataset."""
        return len(self.frame_indices)

    @classmethod
    def load_from_dlc_csv(
        cls,
        *,
        filepath: Path,
        min_confidence: float = 0.3
    ) -> "EyeTrackingData":
        """
        Load DeepLabCut CSV with pupil points (p1-p8) and tear duct.

        Expected columns:
        - p1_x, p1_y, p1_likelihood
        - p2_x, p2_y, p2_likelihood
        - ...
        - p8_x, p8_y, p8_likelihood
        - tear_duct_x, tear_duct_y, tear_duct_likelihood

        Args:
            filepath: Path to CSV file
            min_confidence: Minimum likelihood to consider a point valid

        Returns:
            EyeTrackingData with observations
        """
        df = pd.read_csv(filepath_or_buffer=filepath, header=[0, 1, 2])

        # Get number of frames
        n_frames = len(df)

        # Extract pupil points (p1-p8)
        pupil_points = np.zeros(shape=(n_frames, 8, 2))
        pupil_points[:] = np.nan

        point_names = [f"p{i}" for i in range(1, 9)]

        for i, name in enumerate(point_names):
            try:
                x_col = (name, "x")
                y_col = (name, "y")
                conf_col = (name, "likelihood")

                x = df[x_col].values
                y = df[y_col].values
                conf = df[conf_col].values

                valid = conf >= min_confidence
                pupil_points[valid, i, 0] = x[valid]
                pupil_points[valid, i, 1] = y[valid]
            except KeyError:
                # Try alternative column naming
                try:
                    x = df[f"{name}_x"].values
                    y = df[f"{name}_y"].values
                    conf = df[f"{name}_likelihood"].values

                    valid = conf >= min_confidence
                    pupil_points[valid, i, 0] = x[valid]
                    pupil_points[valid, i, 1] = y[valid]
                except KeyError:
                    print(f"Warning: Could not find columns for {name}")

        # Extract tear duct
        tear_ducts = np.zeros(shape=(n_frames, 2))
        tear_ducts[:] = np.nan

        try:
            x_col = ("tear_duct", "x")
            y_col = ("tear_duct", "y")
            conf_col = ("tear_duct", "likelihood")

            x = df[x_col].values
            y = df[y_col].values
            conf = df[conf_col].values

            valid = conf >= min_confidence
            tear_ducts[valid, 0] = x[valid]
            tear_ducts[valid, 1] = y[valid]
        except KeyError:
            try:
                x = df["tear_duct_x"].values
                y = df["tear_duct_y"].values
                conf = df["tear_duct_likelihood"].values

                valid = conf >= min_confidence
                tear_ducts[valid, 0] = x[valid]
                tear_ducts[valid, 1] = y[valid]
            except KeyError:
                print("Warning: Could not find tear_duct columns")

        # Count valid points per frame
        n_valid_points = np.sum(~np.isnan(pupil_points[:, :, 0]), axis=1)

        frame_indices = np.arange(n_frames)

        return cls(
            pupil_points_px=pupil_points,
            tear_ducts_px=tear_ducts,
            frame_indices=frame_indices,
            n_valid_points=n_valid_points
        )

    def filter_bad_frames(
        self,
        *,
        min_pupil_points: int = 6,
        require_tear_duct: bool = True
    ) -> "EyeTrackingData":
        """
        Filter out frames with insufficient data.

        Args:
            min_pupil_points: Minimum number of valid pupil points required
            require_tear_duct: Whether tear duct must be present

        Returns:
            Filtered data
        """
        # Find valid frames
        valid = self.n_valid_points >= min_pupil_points

        if require_tear_duct:
            tear_duct_valid = ~np.isnan(self.tear_ducts_px[:, 0])
            valid = valid & tear_duct_valid

        n_before = self.n_frames
        n_after = valid.sum()

        print(f"Filtering: {n_before} → {n_after} frames ({n_before - n_after} removed)")

        return EyeTrackingData(
            pupil_points_px=self.pupil_points_px[valid],
            tear_ducts_px=self.tear_ducts_px[valid],
            frame_indices=self.frame_indices[valid],
            n_valid_points=self.n_valid_points[valid]
        )

    def interpolate_missing_pupil_points(self) -> "EyeTrackingData":
        """
        Interpolate missing pupil points from valid neighbors.

        For frames with some missing points, interpolate from the mean
        of valid points (approximate).

        Returns:
            New EyeTrackingData with interpolated points
        """
        result = self.pupil_points_px.copy()

        for i in range(len(result)):
            frame = result[i]
            valid_mask = ~np.isnan(frame[:, 0])

            if valid_mask.sum() == 0:
                continue

            if valid_mask.sum() < 8:
                # Compute center from valid points
                center = frame[valid_mask].mean(axis=0)

                # For missing points, place them at the center
                # (This is a simple approximation; could use ellipse fitting instead)
                result[i, ~valid_mask] = center

        return EyeTrackingData(
            pupil_points_px=result,
            tear_ducts_px=self.tear_ducts_px,
            frame_indices=self.frame_indices,
            n_valid_points=self.n_valid_points
        )


def load_dlc_csv(*, filepath: Path, min_confidence: float = 0.3) -> EyeTrackingData:
    """
    Load DeepLabCut CSV with pupil points (p1-p8) and tear duct.

    Convenience function that wraps EyeTrackingData.load_from_dlc_csv().

    Args:
        filepath: Path to CSV file
        min_confidence: Minimum likelihood to consider a point valid

    Returns:
        EyeTrackingData with observations
    """
    return EyeTrackingData.load_from_dlc_csv(filepath=filepath, min_confidence=min_confidence)


def filter_bad_frames(
    *,
    data: EyeTrackingData,
    min_pupil_points: int = 6,
    require_tear_duct: bool = True
) -> EyeTrackingData:
    """
    Filter out frames with insufficient data.

    Convenience function that wraps data.filter_bad_frames().

    Args:
        data: Input data
        min_pupil_points: Minimum number of valid pupil points required
        require_tear_duct: Whether tear duct must be present

    Returns:
        Filtered data
    """
    return data.filter_bad_frames(
        min_pupil_points=min_pupil_points,
        require_tear_duct=require_tear_duct
    )


def interpolate_missing_points(*, pupil_points: np.ndarray) -> np.ndarray:
    """
    Interpolate missing pupil points from valid neighbors.

    Args:
        pupil_points: (N, 8, 2) with NaNs for missing points

    Returns:
        (N, 8, 2) with interpolated points
    """
    result = pupil_points.copy()

    for i in range(len(result)):
        frame = result[i]
        valid_mask = ~np.isnan(frame[:, 0])

        if valid_mask.sum() == 0:
            continue

        if valid_mask.sum() < 8:
            # Compute center from valid points
            center = frame[valid_mask].mean(axis=0)

            # For missing points, place them at the center
            result[i, ~valid_mask] = center

    return result