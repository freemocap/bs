"""Save eye tracking optimization results to CSV."""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from pydantic import BaseModel, computed_field
from numpydantic import NDArray, Shape

logger = logging.getLogger(__name__)


class ReprojectionErrorStats(BaseModel):
    """Reprojection error statistics."""

    mean_error_px: float
    std_error_px: float
    median_error_px: float
    max_error_px: float
    min_error_px: float

    @classmethod
    def compute_from_errors(
        cls,
        *,
        reprojection_errors_px: NDArray[Shape["* frame_numbers"], float]
    ) -> "ReprojectionErrorStats":
        """
        Compute reprojection error statistics from error array.

        Args:
            reprojection_errors_px: (n_frames,) reprojection errors in pixels

        Returns:
            Computed statistics
        """
        return cls(
            mean_error_px=float(np.mean(reprojection_errors_px)),
            std_error_px=float(np.std(reprojection_errors_px)),
            median_error_px=float(np.median(reprojection_errors_px)),
            max_error_px=float(np.max(reprojection_errors_px)),
            min_error_px=float(np.min(reprojection_errors_px))
        )


class EyeTrackingResults(BaseModel):
    """Complete eye tracking results for saving."""

    model_config = {"arbitrary_types_allowed": True}

    frame_indices: NDArray[Shape["* frame_numbers"], int]
    pupil_centers_observed_px: NDArray[Shape["*, 2"], float]
    pupil_centers_reprojected_px: NDArray[Shape["*, 2"], float]
    pupil_centers_3d_mm: NDArray[Shape["*, 3"], float]
    eyeball_centers_mm: NDArray[Shape["*, 3"], float]
    gaze_directions: NDArray[Shape["*, 3"], float]
    reprojection_errors_px: NDArray[Shape["* frame_numbers"], float]

    @computed_field
    @property
    def gaze_azimuth_rad(self) -> NDArray[Shape["* frame_numbers"], float]:
        """Azimuth angle in XZ plane from +Z axis (radians)."""
        return np.arctan2(self.gaze_directions[:, 0], self.gaze_directions[:, 2])

    @computed_field
    @property
    def gaze_elevation_rad(self) -> NDArray[Shape["* frame_numbers"], float]:
        """Elevation angle from XY plane (radians)."""
        return np.arcsin(self.gaze_directions[:, 1])

    @computed_field
    @property
    def gaze_azimuth_deg(self) -> NDArray[Shape["* frame_numbers"], float]:
        """Azimuth angle in degrees."""
        return np.rad2deg(self.gaze_azimuth_rad)

    @computed_field
    @property
    def gaze_elevation_deg(self) -> NDArray[Shape["* frame_numbers"], float]:
        """Elevation angle in degrees."""
        return np.rad2deg(self.gaze_elevation_rad)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            'frame': self.frame_indices,

            # Observed pupil center
            'pupil_obs_x_px': self.pupil_centers_observed_px[:, 0],
            'pupil_obs_y_px': self.pupil_centers_observed_px[:, 1],

            # Reprojected pupil center
            'pupil_repr_x_px': self.pupil_centers_reprojected_px[:, 0],
            'pupil_repr_y_px': self.pupil_centers_reprojected_px[:, 1],

            # 3D pupil center
            'pupil_3d_x_mm': self.pupil_centers_3d_mm[:, 0],
            'pupil_3d_y_mm': self.pupil_centers_3d_mm[:, 1],
            'pupil_3d_z_mm': self.pupil_centers_3d_mm[:, 2],

            # Eyeball center
            'eyeball_x_mm': self.eyeball_centers_mm[:, 0],
            'eyeball_y_mm': self.eyeball_centers_mm[:, 1],
            'eyeball_z_mm': self.eyeball_centers_mm[:, 2],

            # Gaze direction
            'gaze_x': self.gaze_directions[:, 0],
            'gaze_y': self.gaze_directions[:, 1],
            'gaze_z': self.gaze_directions[:, 2],

            # Gaze angles
            'gaze_azimuth_rad': self.gaze_azimuth_rad,
            'gaze_elevation_rad': self.gaze_elevation_rad,
            'gaze_azimuth_deg': self.gaze_azimuth_deg,
            'gaze_elevation_deg': self.gaze_elevation_deg,

            # Reprojection error
            'reprojection_error_px': self.reprojection_errors_px
        }

        return pd.DataFrame(data=data)

    def save_to_csv(self, *, filepath: Path) -> None:
        """Save results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path_or_buf=filepath, index=False)

        logger.info(f"Saved eye tracking results: {filepath}")
        logger.info(f"  Columns: {len(df.columns)}")
        logger.info(f"  Rows: {len(df)}")


class SummaryStatistics(BaseModel):
    """Summary statistics for eye tracking optimization."""

    # Reprojection errors
    mean_reprojection_error_px: float
    median_reprojection_error_px: float
    std_reprojection_error_px: float
    max_reprojection_error_px: float

    # Eyeball position
    mean_eyeball_x_mm: float
    mean_eyeball_y_mm: float
    mean_eyeball_z_mm: float
    std_eyeball_x_mm: float
    std_eyeball_y_mm: float
    std_eyeball_z_mm: float

    # Gaze angles
    mean_azimuth_deg: float
    mean_elevation_deg: float
    std_azimuth_deg: float
    std_elevation_deg: float
    azimuth_range_deg: float
    elevation_range_deg: float

    # Optimization info
    optimization_time_sec: float
    n_frames: int

    @classmethod
    def compute_from_results(
        cls,
        *,
        reprojection_errors: ReprojectionErrorStats,
        eyeball_centers_mm: NDArray[Shape["*, 3"], float],
        gaze_directions: NDArray[Shape["*, 3"], float],
        optimization_time_sec: float,
        n_frames: int
    ) -> "SummaryStatistics":
        """
        Compute summary statistics from optimization results.

        Args:
            reprojection_errors: Error statistics
            eyeball_centers_mm: (n_frames, 3) eyeball positions
            gaze_directions: (n_frames, 3) gaze direction vectors
            optimization_time_sec: Time in seconds
            n_frames: Number of frames

        Returns:
            Computed summary statistics
        """
        # Compute gaze angles
        azimuth_rad = np.arctan2(gaze_directions[:, 0], gaze_directions[:, 2])
        elevation_rad = np.arcsin(gaze_directions[:, 1])

        azimuth_deg = np.rad2deg(azimuth_rad)
        elevation_deg = np.rad2deg(elevation_rad)

        return cls(
            mean_reprojection_error_px=reprojection_errors.mean_error_px,
            median_reprojection_error_px=reprojection_errors.median_error_px,
            std_reprojection_error_px=reprojection_errors.std_error_px,
            max_reprojection_error_px=reprojection_errors.max_error_px,
            mean_eyeball_x_mm=float(np.mean(eyeball_centers_mm[:, 0])),
            mean_eyeball_y_mm=float(np.mean(eyeball_centers_mm[:, 1])),
            mean_eyeball_z_mm=float(np.mean(eyeball_centers_mm[:, 2])),
            std_eyeball_x_mm=float(np.std(eyeball_centers_mm[:, 0])),
            std_eyeball_y_mm=float(np.std(eyeball_centers_mm[:, 1])),
            std_eyeball_z_mm=float(np.std(eyeball_centers_mm[:, 2])),
            mean_azimuth_deg=float(np.mean(azimuth_deg)),
            mean_elevation_deg=float(np.mean(elevation_deg)),
            std_azimuth_deg=float(np.std(azimuth_deg)),
            std_elevation_deg=float(np.std(elevation_deg)),
            azimuth_range_deg=float(np.ptp(azimuth_deg)),
            elevation_range_deg=float(np.ptp(elevation_deg)),
            optimization_time_sec=optimization_time_sec,
            n_frames=n_frames
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with metric/value columns."""
        data = {
            'metric': list(self.model_fields.keys()),
            'value': [getattr(self, field) for field in self.model_fields.keys()]
        }
        return pd.DataFrame(data=data)

    def save_to_csv(self, *, filepath: Path) -> None:
        """Save summary statistics to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path_or_buf=filepath, index=False)
        logger.info(f"Saved summary statistics: {filepath}")


def save_eye_tracking_results(
    *,
    filepath: Path,
    frame_indices: NDArray[Shape["* frame_numbers"], int],
    pupil_centers_observed_px: NDArray[Shape["*, 2"], float],
    pupil_centers_reprojected_px: NDArray[Shape["*, 2"], float],
    pupil_centers_3d_mm: NDArray[Shape["*, 3"], float],
    eyeball_centers_mm: NDArray[Shape["*, 3"], float],
    gaze_directions: NDArray[Shape["*, 3"], float],
    reprojection_errors_px: NDArray[Shape["* frame_numbers"], float]
) -> None:
    """
    Save complete eye tracking results to CSV.

    Args:
        filepath: Output CSV path
        frame_indices: (n_frames,) frame numbers
        pupil_centers_observed_px: (n_frames, 2) observed centers in pixels
        pupil_centers_reprojected_px: (n_frames, 2) reprojected centers in pixels
        pupil_centers_3d_mm: (n_frames, 3) 3D pupil centers in mm
        eyeball_centers_mm: (n_frames, 3) eyeball center positions in mm
        gaze_directions: (n_frames, 3) normalized gaze direction vectors
        reprojection_errors_px: (n_frames,) reprojection errors in pixels
    """
    results = EyeTrackingResults(
        frame_indices=frame_indices,
        pupil_centers_observed_px=pupil_centers_observed_px,
        pupil_centers_reprojected_px=pupil_centers_reprojected_px,
        pupil_centers_3d_mm=pupil_centers_3d_mm,
        eyeball_centers_mm=eyeball_centers_mm,
        gaze_directions=gaze_directions,
        reprojection_errors_px=reprojection_errors_px
    )

    results.save_to_csv(filepath=filepath)


def compute_reprojection_error_stats(
    *,
    reprojection_errors_px: NDArray[Shape["* frame_numbers"], float]
) -> ReprojectionErrorStats:
    """
    Compute reprojection error statistics.

    Args:
        reprojection_errors_px: (n_frames,) reprojection errors in pixels

    Returns:
        Error statistics
    """
    return ReprojectionErrorStats.compute_from_errors(
        reprojection_errors_px=reprojection_errors_px
    )


def save_summary_stats(
    *,
    filepath: Path,
    reprojection_errors: ReprojectionErrorStats,
    eyeball_centers_mm: NDArray[Shape["*, 3"], float],
    gaze_directions: NDArray[Shape["*, 3"], float],
    optimization_time_sec: float,
    n_frames: int
) -> None:
    """
    Save summary statistics to CSV.

    Args:
        filepath: Output CSV path
        reprojection_errors: Error statistics
        eyeball_centers_mm: (n_frames, 3) eyeball positions
        gaze_directions: (n_frames, 3) gaze direction vectors
        optimization_time_sec: Time in seconds
        n_frames: Number of frames
    """
    stats = SummaryStatistics.compute_from_results(
        reprojection_errors=reprojection_errors,
        eyeball_centers_mm=eyeball_centers_mm,
        gaze_directions=gaze_directions,
        optimization_time_sec=optimization_time_sec,
        n_frames=n_frames
    )

    stats.save_to_csv(filepath=filepath)


def print_summary(
    *,
    reprojection_errors: ReprojectionErrorStats,
    eyeball_centers_mm: NDArray[Shape["*, 3"], float],
    gaze_directions: NDArray[Shape["*, 3"], float]
) -> None:
    """
    Print summary statistics to console.

    Args:
        reprojection_errors: Error statistics
        eyeball_centers_mm: (n_frames, 3) eyeball positions
        gaze_directions: (n_frames, 3) gaze direction vectors
    """
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)

    logger.info("\nReprojection Errors:")
    logger.info(f"  Mean:   {reprojection_errors.mean_error_px:.2f} px")
    logger.info(f"  Median: {reprojection_errors.median_error_px:.2f} px")
    logger.info(f"  Std:    {reprojection_errors.std_error_px:.2f} px")
    logger.info(f"  Max:    {reprojection_errors.max_error_px:.2f} px")

    logger.info("\nEyeball Position (mm):")
    mean_center = np.mean(eyeball_centers_mm, axis=0)
    std_center = np.std(eyeball_centers_mm, axis=0)
    logger.info(f"  Mean: [{mean_center[0]:.2f}, {mean_center[1]:.2f}, {mean_center[2]:.2f}]")
    logger.info(f"  Std:  [{std_center[0]:.2f}, {std_center[1]:.2f}, {std_center[2]:.2f}]")

    logger.info("\nGaze Angles:")
    azimuth_rad = np.arctan2(gaze_directions[:, 0], gaze_directions[:, 2])
    elevation_rad = np.arcsin(gaze_directions[:, 1])

    azimuth_deg = np.rad2deg(azimuth_rad)
    elevation_deg = np.rad2deg(elevation_rad)

    logger.info(f"  Azimuth (horizontal):")
    logger.info(f"    Range: {np.min(azimuth_deg):.1f}° to {np.max(azimuth_deg):.1f}°")
    logger.info(f"    Mean:  {np.mean(azimuth_deg):.1f}° ± {np.std(azimuth_deg):.1f}°")

    logger.info(f"  Elevation (vertical):")
    logger.info(f"    Range: {np.min(elevation_deg):.1f}° to {np.max(elevation_deg):.1f}°")
    logger.info(f"    Mean:  {np.mean(elevation_deg):.1f}° ± {np.std(elevation_deg):.1f}°")