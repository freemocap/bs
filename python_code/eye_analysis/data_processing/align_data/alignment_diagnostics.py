from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from python_code.eye_analysis.data_models.trajectory_dataset import TrajectoryDataset, Trajectory2D


def get_correction_summary(
        *,
        original_dataset: TrajectoryDataset,
        corrected_dataset: TrajectoryDataset,
        stabilized_on: str = "tear_duct",
        aligned_to: str = "outer_eye"
) -> dict[str, float]:
    """Get summary statistics of the spatial correction applied.

    Args:
        original_dataset: Original (uncorrected) dataset
        corrected_dataset: Spatially corrected dataset
        stabilized_on: Name of tear duct landmark
        aligned_to: Name of outer eye corner landmark

    Returns:
        Dictionary with correction statistics
    """
    # Compute statistics on cleaned data
    orig_tear: Trajectory2D = original_dataset.trajectories[stabilized_on].cleaned
    corr_tear: Trajectory2D = corrected_dataset.trajectories[stabilized_on].cleaned

    orig_outer: Trajectory2D = original_dataset.trajectories[aligned_to].cleaned
    corr_outer: Trajectory2D = corrected_dataset.trajectories[aligned_to].cleaned

    # Compute average positions
    orig_tear_mean: np.ndarray = np.nanmean(a=orig_tear.data, axis=0)
    corr_tear_mean: np.ndarray = np.nanmean(a=corr_tear.data, axis=0)

    orig_outer_mean: np.ndarray = np.nanmean(a=orig_outer.data, axis=0)
    corr_outer_mean: np.ndarray = np.nanmean(a=corr_outer.data, axis=0)

    # Compute pupil center
    pupil_names: list[str] = [f'p{i}' for i in range(1, 9)]
    orig_pupil_points: list[np.ndarray] = [original_dataset.trajectories[name].cleaned.data for name in pupil_names]
    orig_pupil_center: np.ndarray = np.nanmean(a=np.stack(arrays=orig_pupil_points, axis=1), axis=1)

    corr_pupil_points: list[np.ndarray] = [corrected_dataset.trajectories[name].cleaned.data for name in pupil_names]
    corr_pupil_center: np.ndarray = np.nanmean(a=np.stack(arrays=corr_pupil_points, axis=1), axis=1)

    orig_pupil_mean: np.ndarray = np.nanmean(a=orig_pupil_center, axis=0)
    corr_pupil_mean: np.ndarray = np.nanmean(a=corr_pupil_center, axis=0)

    return {
        'original_tear_duct_mean_x': float(orig_tear_mean[0]),
        'original_tear_duct_mean_y': float(orig_tear_mean[1]),
        'corrected_tear_duct_mean_x': float(corr_tear_mean[0]),
        'corrected_tear_duct_mean_y': float(corr_tear_mean[1]),
        'original_outer_eye_mean_x': float(orig_outer_mean[0]),
        'original_outer_eye_mean_y': float(orig_outer_mean[1]),
        'corrected_outer_eye_mean_x': float(corr_outer_mean[0]),
        'corrected_outer_eye_mean_y': float(corr_outer_mean[1]),
        'original_pupil_center_mean_x': float(orig_pupil_mean[0]),
        'original_pupil_center_mean_y': float(orig_pupil_mean[1]),
        'corrected_pupil_center_mean_x': float(corr_pupil_mean[0]),
        'corrected_pupil_center_mean_y': float(corr_pupil_mean[1]),
    }


def plot_correction_comparison(
        *,
        original_dataset: TrajectoryDataset,
        corrected_dataset: TrajectoryDataset,
        output_path: Path | None = None
) -> None:
    """Plot before/after comparison of spatial correction.

    Args:
        original_dataset: Original trajectory dataset
        corrected_dataset: Spatially corrected dataset
        output_path: Optional path to save figure
    """
    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Get pupil center trajectories
    pupil_names: list[str] = [f'p{i}' for i in range(1, 9)]

    # Original data
    orig_pupil_points: list[np.ndarray] = [original_dataset.trajectories[name].cleaned.data for name in pupil_names]
    orig_pupil_center: np.ndarray = np.nanmean(a=np.stack(arrays=orig_pupil_points, axis=1), axis=1)

    orig_tear: np.ndarray = original_dataset.trajectories['tear_duct'].cleaned.data
    orig_outer: np.ndarray = original_dataset.trajectories['outer_eye'].cleaned.data

    # Corrected data
    corr_pupil_points: list[np.ndarray] = [corrected_dataset.trajectories[name].cleaned.data for name in pupil_names]
    corr_pupil_center: np.ndarray = np.nanmean(a=np.stack(arrays=corr_pupil_points, axis=1), axis=1)

    corr_tear: np.ndarray = corrected_dataset.trajectories['tear_duct'].cleaned.data
    corr_outer: np.ndarray = corrected_dataset.trajectories['outer_eye'].cleaned.data

    # Plot original
    ax: plt.Axes = axes[0]
    ax.plot(
        orig_pupil_center[:, 0],
        orig_pupil_center[:, 1],
        color='blue',
        alpha=0.5,
        linewidth=1,
        marker='o',
        markersize=2,
        label='Pupil center'
    )
    ax.plot(
        orig_tear[:, 0],
        orig_tear[:, 1],
        color='red',
        linewidth=1,
        marker='s',
        markersize=3,
        label='Tear duct',
        alpha=0.6
    )
    ax.plot(
        orig_outer[:, 0],
        orig_outer[:, 1],
        color='green',
        linewidth=1,
        marker='^',
        markersize=3,
        label='Outer eye',
        alpha=0.6
    )

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Original Coordinates')
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    ax.axis('equal')

    # Plot corrected
    ax = axes[1]
    ax.plot(
        corr_pupil_center[:, 0],
        corr_pupil_center[:, 1],
        color='blue',
        alpha=0.5,
        linewidth=1,
        marker='o',
        markersize=2,
        label='Pupil center'
    )
    ax.plot(
        corr_tear[:, 0],
        corr_tear[:, 1],
        color='red',
        linewidth=1,
        marker='s',
        markersize=3,
        label='Tear duct',
        alpha=0.6
    )
    ax.plot(
        corr_outer[:, 0],
        corr_outer[:, 1],
        color='green',
        linewidth=1,
        marker='^',
        markersize=3,
        label='Outer eye',
        alpha=0.6
    )

    # Add axis labels for anatomical reference
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    ax.set_xlabel('X (lateral → | ← nasal)')
    ax.set_ylabel('Y (superior ↑ | ↓ inferior)')
    ax.set_title('Spatially Corrected Coordinates\n(Anatomical Reference Frame)')
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()

    if output_path:
        plt.savefig(fname=output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")

    # plt.show()
