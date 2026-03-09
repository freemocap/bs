"""Save eye tracking optimization results to CSV."""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_eye_tracking_results(
    *,
    filepath: Path,
    frame_indices: np.ndarray,
    pupil_centers_observed_px: np.ndarray,
    pupil_centers_reprojected_px: np.ndarray,
    pupil_centers_3d_mm: np.ndarray,
    eyeball_centers_mm: np.ndarray,
    gaze_directions: np.ndarray,
    reprojection_errors_px: np.ndarray
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
    n_frames = len(frame_indices)

    # Compute gaze angles from direction vectors
    # Azimuth: angle in XZ plane from +Z axis
    # Elevation: angle from XY plane
    azimuth_rad = np.arctan2(gaze_directions[:, 0], gaze_directions[:, 2])
    elevation_rad = np.arcsin(gaze_directions[:, 1])

    data: dict[str, np.ndarray] = {
        'frame': frame_indices,

        # Observed pupil center
        'pupil_obs_x_px': pupil_centers_observed_px[:, 0],
        'pupil_obs_y_px': pupil_centers_observed_px[:, 1],

        # Reprojected pupil center
        'pupil_repr_x_px': pupil_centers_reprojected_px[:, 0],
        'pupil_repr_y_px': pupil_centers_reprojected_px[:, 1],

        # 3D pupil center
        'pupil_3d_x_mm': pupil_centers_3d_mm[:, 0],
        'pupil_3d_y_mm': pupil_centers_3d_mm[:, 1],
        'pupil_3d_z_mm': pupil_centers_3d_mm[:, 2],

        # Eyeball center
        'eyeball_x_mm': eyeball_centers_mm[:, 0],
        'eyeball_y_mm': eyeball_centers_mm[:, 1],
        'eyeball_z_mm': eyeball_centers_mm[:, 2],

        # Gaze direction
        'gaze_x': gaze_directions[:, 0],
        'gaze_y': gaze_directions[:, 1],
        'gaze_z': gaze_directions[:, 2],

        # Gaze angles
        'gaze_azimuth_rad': azimuth_rad,
        'gaze_elevation_rad': elevation_rad,
        'gaze_azimuth_deg': np.rad2deg(azimuth_rad),
        'gaze_elevation_deg': np.rad2deg(elevation_rad),

        # Reprojection error
        'reprojection_error_px': reprojection_errors_px
    }

    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=filepath, index=False)

    logger.info(f"Saved eye tracking results: {filepath}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Rows: {len(df)}")


def compute_reprojection_error_stats(
    *,
    reprojection_errors_px: np.ndarray
) -> dict[str, float]:
    """
    Compute reprojection error statistics.

    Args:
        reprojection_errors_px: (n_frames,) reprojection errors in pixels

    Returns:
        Dictionary with error statistics
    """
    return {
        'mean_error_px': float(np.mean(reprojection_errors_px)),
        'std_error_px': float(np.std(reprojection_errors_px)),
        'median_error_px': float(np.median(reprojection_errors_px)),
        'max_error_px': float(np.max(reprojection_errors_px)),
        'min_error_px': float(np.min(reprojection_errors_px))
    }


def save_summary_stats(
    *,
    filepath: Path,
    reprojection_errors: dict[str, float],
    eyeball_centers_mm: np.ndarray,
    gaze_directions: np.ndarray,
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
    # Compute gaze angles
    azimuth_rad = np.arctan2(gaze_directions[:, 0], gaze_directions[:, 2])
    elevation_rad = np.arcsin(gaze_directions[:, 1])

    azimuth_deg = np.rad2deg(azimuth_rad)
    elevation_deg = np.rad2deg(elevation_rad)

    stats = {
        'metric': [
            'mean_reprojection_error_px',
            'median_reprojection_error_px',
            'std_reprojection_error_px',
            'max_reprojection_error_px',
            'mean_eyeball_x_mm',
            'mean_eyeball_y_mm',
            'mean_eyeball_z_mm',
            'std_eyeball_x_mm',
            'std_eyeball_y_mm',
            'std_eyeball_z_mm',
            'mean_azimuth_deg',
            'mean_elevation_deg',
            'std_azimuth_deg',
            'std_elevation_deg',
            'azimuth_range_deg',
            'elevation_range_deg',
            'optimization_time_sec',
            'n_frames'
        ],
        'value': [
            reprojection_errors['mean_error_px'],
            reprojection_errors['median_error_px'],
            reprojection_errors['std_error_px'],
            reprojection_errors['max_error_px'],
            float(np.mean(eyeball_centers_mm[:, 0])),
            float(np.mean(eyeball_centers_mm[:, 1])),
            float(np.mean(eyeball_centers_mm[:, 2])),
            float(np.std(eyeball_centers_mm[:, 0])),
            float(np.std(eyeball_centers_mm[:, 1])),
            float(np.std(eyeball_centers_mm[:, 2])),
            float(np.mean(azimuth_deg)),
            float(np.mean(elevation_deg)),
            float(np.std(azimuth_deg)),
            float(np.std(elevation_deg)),
            float(np.ptp(azimuth_deg)),
            float(np.ptp(elevation_deg)),
            optimization_time_sec,
            n_frames
        ]
    }

    df = pd.DataFrame(data=stats)
    df.to_csv(path_or_buf=filepath, index=False)

    logger.info(f"Saved summary statistics: {filepath}")


def print_summary(
    *,
    reprojection_errors: dict[str, float],
    eyeball_centers_mm: np.ndarray,
    gaze_directions: np.ndarray
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
    logger.info(f"  Mean:   {reprojection_errors['mean_error_px']:.2f} px")
    logger.info(f"  Median: {reprojection_errors['median_error_px']:.2f} px")
    logger.info(f"  Std:    {reprojection_errors['std_error_px']:.2f} px")
    logger.info(f"  Max:    {reprojection_errors['max_error_px']:.2f} px")

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