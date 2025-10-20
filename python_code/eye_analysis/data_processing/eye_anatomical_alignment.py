"""Spatial correction for eye tracking data to establish anatomical coordinate system.

Applies transformations to align data with anatomical axes:
1. Translate: tear_duct → origin (0,0)
2. Rotate: eye_outer → X-axis (lateral-nasal alignment)
3. Center: mode of pupil center → origin (resting position)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from python_code.eye_analysis.data_models.eye_video_dataset import EyeVideoData
from python_code.eye_analysis.data_models.trajectory_dataset import Trajectory2D, TrajectoryPair, TrajectoryDataset


def compute_histogram_mode(*, data: np.ndarray, n_bins: int = 50) -> float:
    """Compute mode of continuous data using histogram binning.

    Args:
        data: 1D array of values
        n_bins: Number of histogram bins

    Returns:
        Mode value (center of most frequent bin)
    """
    # Remove NaN values
    valid_data: np.ndarray = data[~np.isnan(data)]

    if len(valid_data) == 0:
        return 0.0

    hist: np.ndarray
    edges: np.ndarray
    hist, edges = np.histogram(a=valid_data, bins=n_bins)
    mode_idx: int = int(np.argmax(a=hist))

    # Return center of the modal bin
    return float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)


def apply_spatial_correction_to_trajectory(
    *,
    trajectory: Trajectory2D,
    tear_duct_positions: np.ndarray,
    rotation_angles: np.ndarray,
    mode_offset: np.ndarray
) -> Trajectory2D:
    """Apply spatial correction to a single trajectory.

    Args:
        trajectory: Input trajectory to correct
        tear_duct_positions: (n_frames, 2) tear duct positions for translation
        rotation_angles: (n_frames,) rotation angles for each frame (radians)
        mode_offset: (2,) offset to center by mode

    Returns:
        Spatially corrected trajectory
    """
    n_frames: int = len(trajectory.data)
    corrected_data: np.ndarray = np.zeros_like(a=trajectory.data)

    for i in range(n_frames):
        # Step 1: Translate by tear duct
        translated: np.ndarray = trajectory.data[i] - tear_duct_positions[i]

        # Step 2: Rotate to align eye_outer with X-axis
        angle: float = float(rotation_angles[i])
        cos_a: float = float(np.cos(angle))
        sin_a: float = float(np.sin(angle))
        rotation_matrix: np.ndarray = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        rotated: np.ndarray = rotation_matrix @ translated

        # Step 3: Shift by mode offset
        corrected_data[i] = rotated - mode_offset

    return Trajectory2D(
        name=trajectory.name,
        data=corrected_data,
        timestamps=trajectory.timestamps,
        confidence=trajectory.confidence,
        metadata={**trajectory.metadata, 'spatially_corrected': True}
    )


def compute_spatial_correction_parameters(
    *,
    tear_duct_trajectory: Trajectory2D,
    outer_eye_trajectory: Trajectory2D,
    pupil_trajectories: list[Trajectory2D],
    n_bins: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute parameters for spatial correction.

    Args:
        tear_duct_trajectory: Tear duct positions
        outer_eye_trajectory: Outer eye corner positions
        pupil_trajectories: List of pupil landmark trajectories (p1-p8)
        n_bins: Number of bins for mode computation

    Returns:
        Tuple of (tear_duct_positions, rotation_angles, mode_offset)
    """
    n_frames: int = len(tear_duct_trajectory.data)

    # Step 1: Get tear duct positions for translation
    tear_duct_positions: np.ndarray = tear_duct_trajectory.data.copy()

    # Step 2: Compute rotation angles
    # After translating by tear duct, find angle of eye_outer
    outer_eye_translated: np.ndarray = outer_eye_trajectory.data - tear_duct_positions
    rotation_angles: np.ndarray = -np.arctan2(outer_eye_translated[:, 1], outer_eye_translated[:, 0])

    # Step 3: Compute mode offset from pupil centers
    # First translate and rotate all pupil trajectories
    pupil_centers_rotated: list[np.ndarray] = []

    for frame_idx in range(n_frames):
        frame_points: list[np.ndarray] = []
        for pupil_traj in pupil_trajectories:
            # Translate
            translated: np.ndarray = pupil_traj.data[frame_idx] - tear_duct_positions[frame_idx]

            # Rotate
            angle: float = float(rotation_angles[frame_idx])
            cos_a: float = float(np.cos(angle))
            sin_a: float = float(np.sin(angle))
            rotation_matrix: np.ndarray = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            rotated: np.ndarray = rotation_matrix @ translated

            frame_points.append(rotated)

        # Average the pupil points for this frame
        if frame_points:
            pupil_centers_rotated.append(np.mean(a=frame_points, axis=0))

    pupil_centers_rotated_array: np.ndarray = np.array(pupil_centers_rotated)  # (n_frames, 2)

    # Compute mode for X and Y separately
    x_mode: float = compute_histogram_mode(data=pupil_centers_rotated_array[:, 0], n_bins=n_bins)
    y_mode: float = compute_histogram_mode(data=pupil_centers_rotated_array[:, 1], n_bins=n_bins)
    mode_offset: np.ndarray = np.array([x_mode, y_mode])

    return tear_duct_positions, rotation_angles, mode_offset


def apply_spatial_correction_to_dataset(
    *,
    dataset: TrajectoryDataset,
    tear_duct_name: str = "tear_duct",
    outer_eye_name: str = "outer_eye",
    pupil_names: list[str] | None = None,
    apply_to_raw: bool = True,
    apply_to_cleaned: bool = True,
    n_bins: int = 50
) -> TrajectoryDataset:
    """Apply spatial correction to entire trajectory dataset.

    Creates a new dataset with spatially corrected trajectories establishing
    an anatomical coordinate system:
    - Origin at resting pupil position
    - X-axis along lateral-nasal axis (tear_duct → eye_outer)
    - Y-axis along superior-inferior axis

    Args:
        dataset: Input trajectory dataset
        tear_duct_name: Name of tear duct landmark
        outer_eye_name: Name of outer eye corner landmark
        pupil_names: Names of pupil landmarks (default: p1-p8)
        apply_to_raw: Whether to correct raw trajectories
        apply_to_cleaned: Whether to correct cleaned trajectories
        n_bins: Number of bins for mode computation

    Returns:
        New TrajectoryDataset with spatially corrected trajectories
    """
    if pupil_names is None:
        pupil_names = [f'p{i}' for i in range(1, 9)]

    # Validate required landmarks exist
    required: list[str] = [tear_duct_name, outer_eye_name] + pupil_names
    missing: list[str] = [name for name in required if name not in dataset.pairs]
    if missing:
        raise ValueError(f"Missing required landmarks: {missing}")

    corrected_pairs: dict[str, TrajectoryPair] = {}

    for name, pair in dataset.pairs.items():
        # Process raw trajectory
        if apply_to_raw:
            # Compute correction parameters from raw data
            tear_duct_traj: Trajectory2D = dataset.pairs[tear_duct_name].raw
            outer_eye_traj: Trajectory2D = dataset.pairs[outer_eye_name].raw
            pupil_trajs: list[Trajectory2D] = [dataset.pairs[pname].raw for pname in pupil_names]

            tear_duct_pos: np.ndarray
            rotation_angles: np.ndarray
            mode_offset: np.ndarray
            tear_duct_pos, rotation_angles, mode_offset = compute_spatial_correction_parameters(
                tear_duct_trajectory=tear_duct_traj,
                outer_eye_trajectory=outer_eye_traj,
                pupil_trajectories=pupil_trajs,
                n_bins=n_bins
            )

            corrected_raw: Trajectory2D = apply_spatial_correction_to_trajectory(
                trajectory=pair.raw,
                tear_duct_positions=tear_duct_pos,
                rotation_angles=rotation_angles,
                mode_offset=mode_offset
            )
        else:
            corrected_raw = pair.raw

        # Process cleaned trajectory
        if apply_to_cleaned:
            # Compute correction parameters from cleaned data
            tear_duct_traj: Trajectory2D = dataset.pairs[tear_duct_name].cleaned
            outer_eye_traj: Trajectory2D = dataset.pairs[outer_eye_name].cleaned
            pupil_trajs: list[Trajectory2D] = [dataset.pairs[pname].cleaned for pname in pupil_names]

            tear_duct_pos: np.ndarray
            rotation_angles: np.ndarray
            mode_offset: np.ndarray
            tear_duct_pos, rotation_angles, mode_offset = compute_spatial_correction_parameters(
                tear_duct_trajectory=tear_duct_traj,
                outer_eye_trajectory=outer_eye_traj,
                pupil_trajectories=pupil_trajs,
                n_bins=n_bins
            )

            corrected_cleaned: Trajectory2D = apply_spatial_correction_to_trajectory(
                trajectory=pair.cleaned,
                tear_duct_positions=tear_duct_pos,
                rotation_angles=rotation_angles,
                mode_offset=mode_offset
            )
        else:
            corrected_cleaned = pair.cleaned

        corrected_pairs[name] = TrajectoryPair(
            raw=corrected_raw,
            cleaned=corrected_cleaned
        )

    return TrajectoryDataset(
        name=dataset.name,
        pairs=corrected_pairs,
        frame_indices=dataset.frame_indices,
        metadata={
            **dataset.metadata,
            'spatially_corrected': True,
            'correction_params': {
                'tear_duct_name': tear_duct_name,
                'outer_eye_name': outer_eye_name,
                'pupil_names': pupil_names,
                'n_bins': n_bins
            }
        }
    )


def get_correction_summary(
    *,
    original_dataset: TrajectoryDataset,
    corrected_dataset: TrajectoryDataset,
    tear_duct_name: str = "tear_duct",
    outer_eye_name: str = "outer_eye"
) -> dict[str, float]:
    """Get summary statistics of the spatial correction applied.

    Args:
        original_dataset: Original (uncorrected) dataset
        corrected_dataset: Spatially corrected dataset
        tear_duct_name: Name of tear duct landmark
        outer_eye_name: Name of outer eye corner landmark

    Returns:
        Dictionary with correction statistics
    """
    # Compute statistics on cleaned data
    orig_tear: Trajectory2D = original_dataset.pairs[tear_duct_name].cleaned
    corr_tear: Trajectory2D = corrected_dataset.pairs[tear_duct_name].cleaned

    orig_outer: Trajectory2D = original_dataset.pairs[outer_eye_name].cleaned
    corr_outer: Trajectory2D = corrected_dataset.pairs[outer_eye_name].cleaned

    # Compute average positions
    orig_tear_mean: np.ndarray = np.nanmean(a=orig_tear.data, axis=0)
    corr_tear_mean: np.ndarray = np.nanmean(a=corr_tear.data, axis=0)

    orig_outer_mean: np.ndarray = np.nanmean(a=orig_outer.data, axis=0)
    corr_outer_mean: np.ndarray = np.nanmean(a=corr_outer.data, axis=0)

    # Compute pupil center
    pupil_names: list[str] = [f'p{i}' for i in range(1, 9)]
    orig_pupil_points: list[np.ndarray] = [original_dataset.pairs[name].cleaned.data for name in pupil_names]
    orig_pupil_center: np.ndarray = np.nanmean(a=np.stack(arrays=orig_pupil_points, axis=1), axis=1)

    corr_pupil_points: list[np.ndarray] = [corrected_dataset.pairs[name].cleaned.data for name in pupil_names]
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
    orig_pupil_points: list[np.ndarray] = [original_dataset.pairs[name].cleaned.data for name in pupil_names]
    orig_pupil_center: np.ndarray = np.nanmean(a=np.stack(arrays=orig_pupil_points, axis=1), axis=1)

    orig_tear: np.ndarray = original_dataset.pairs['tear_duct'].cleaned.data
    orig_outer: np.ndarray = original_dataset.pairs['outer_eye'].cleaned.data

    # Corrected data
    corr_pupil_points: list[np.ndarray] = [corrected_dataset.pairs[name].cleaned.data for name in pupil_names]
    corr_pupil_center: np.ndarray = np.nanmean(a=np.stack(arrays=corr_pupil_points, axis=1), axis=1)

    corr_tear: np.ndarray = corrected_dataset.pairs['tear_duct'].cleaned.data
    corr_outer: np.ndarray = corrected_dataset.pairs['outer_eye'].cleaned.data

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

    plt.show()


def eye_correction_main() -> None:
    """Run spatial correction example."""
    # Setup paths
    base_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37"
    )
    video_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4"
    )
    timestamps_npy_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy"
    )
    csv_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )

    # Load dataset
    print("Loading eye tracking dataset...")
    eye_dataset: EyeVideoData = EyeVideoData.create(
        data_name="ferret_757_eye_tracking",
        recording_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
        butterworth_cutoff=6.0,
    )

    # Apply spatial correction
    print("\nApplying spatial correction...")
    print("  Step 1: Translating by tear duct position...")
    print("  Step 2: Rotating to align outer eye with X-axis...")
    print("  Step 3: Centering by pupil mode...")

    corrected_dataset: TrajectoryDataset = apply_spatial_correction_to_dataset(
        dataset=eye_dataset.dataset,
        tear_duct_name="tear_duct",
        outer_eye_name="outer_eye",
        pupil_names=[f'p{i}' for i in range(1, 9)],
        apply_to_raw=True,
        apply_to_cleaned=True,
        n_bins=50
    )

    # Get correction summary
    print("\n" + "=" * 60)
    print("SPATIAL CORRECTION SUMMARY")
    print("=" * 60)

    summary: dict[str, float] = get_correction_summary(
        original_dataset=eye_dataset.dataset,
        corrected_dataset=corrected_dataset,
        tear_duct_name="tear_duct",
        outer_eye_name="outer_eye"
    )

    print("\nOriginal coordinates (cleaned data):")
    print(
        f"  Tear duct mean:   ({summary['original_tear_duct_mean_x']:.2f}, {summary['original_tear_duct_mean_y']:.2f})")
    print(
        f"  Outer eye mean:   ({summary['original_outer_eye_mean_x']:.2f}, {summary['original_outer_eye_mean_y']:.2f})")
    print(
        f"  Pupil center mean: ({summary['original_pupil_center_mean_x']:.2f}, {summary['original_pupil_center_mean_y']:.2f})")

    print("\nCorrected coordinates (anatomical frame):")
    print(
        f"  Tear duct mean:   ({summary['corrected_tear_duct_mean_x']:.2f}, {summary['corrected_tear_duct_mean_y']:.2f})")
    print(
        f"  Outer eye mean:   ({summary['corrected_outer_eye_mean_x']:.2f}, {summary['corrected_outer_eye_mean_y']:.2f})")
    print(
        f"  Pupil center mean: ({summary['corrected_pupil_center_mean_x']:.2f}, {summary['corrected_pupil_center_mean_y']:.2f})")

    print("\nAnatomical coordinate system established:")
    print("  - Origin: Resting pupil position (mode)")
    print("  - X-axis: Lateral (+) ← → Nasal (-)")
    print("  - Y-axis: Superior (+) ↑ ↓ Inferior (-)")
    print("=" * 60)

    # Plot comparison
    print("\nGenerating comparison plot...")
    plot_correction_comparison(
        original_dataset=eye_dataset.dataset,
        corrected_dataset=corrected_dataset,
        output_path=base_path / "spatial_correction_comparison.png"
    )

    # Access corrected data
    print("\nAccessing corrected trajectories:")
    print("  Raw data:")
    print(f"    corrected_dataset.raw['p1']  # Raw corrected trajectory")
    print(f"    corrected_dataset.pairs['p1'].raw  # Also raw")

    print("  Cleaned data:")
    print(f"    corrected_dataset.cleaned['p1']  # Cleaned corrected trajectory")
    print(f"    corrected_dataset.pairs['p1'].cleaned  # Also cleaned")

    # Example: Get pupil center in anatomical coordinates
    pupil_names: list[str] = [f'p{i}' for i in range(1, 9)]
    pupil_points: list[np.ndarray] = [corrected_dataset.pairs[name].cleaned.data for name in pupil_names]
    pupil_center_anatomical: np.ndarray = np.nanmean(a=np.stack(arrays=pupil_points, axis=1), axis=1)

    print(f"\nPupil center in anatomical frame (first 5 frames):")
    print(f"  Shape: {pupil_center_anatomical.shape}")
    for i in range(min(5, len(pupil_center_anatomical))):
        x: float = float(pupil_center_anatomical[i, 0])
        y: float = float(pupil_center_anatomical[i, 1])
        print(f"    Frame {i}: ({x:6.2f}, {y:6.2f})")

    print("\nSpatial correction complete!")


if __name__ == "__main__":
    eye_correction_main()