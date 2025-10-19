"""Example script demonstrating spatial correction for eye tracking data."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from python_code.eye_data_cleanup.eye_analysis.spatial_correction import apply_spatial_correction_to_dataset, \
    get_correction_summary
from python_code.eye_data_cleanup.eye_viewer import EyeVideoDataset


def plot_correction_comparison(
    *,
    original_dataset,
    corrected_dataset,
    output_path: Path | None = None
) -> None:
    """Plot before/after comparison of spatial correction.
    
    Args:
        original_dataset: Original trajectory dataset
        corrected_dataset: Spatially corrected dataset
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    
    # Get pupil center trajectories
    pupil_names = [f'p{i}' for i in range(1, 9)]
    
    # Original data
    orig_pupil_points = [original_dataset.pairs[name].cleaned.data for name in pupil_names]
    orig_pupil_center = np.nanmean(np.stack(orig_pupil_points, axis=1), axis=1)
    
    orig_tear = original_dataset.pairs['tear_duct'].cleaned.data
    orig_outer = original_dataset.pairs['outer_eye'].cleaned.data
    
    # Corrected data
    corr_pupil_points = [corrected_dataset.pairs[name].cleaned.data for name in pupil_names]
    corr_pupil_center = np.nanmean(np.stack(corr_pupil_points, axis=1), axis=1)
    
    corr_tear = corrected_dataset.pairs['tear_duct'].cleaned.data
    corr_outer = corrected_dataset.pairs['outer_eye'].cleaned.data
    
    # Plot original
    ax = axes[0]
    ax.scatter(orig_pupil_center[:, 0], orig_pupil_center[:, 1], 
               c='blue', alpha=0.3, s=10, label='Pupil center')
    ax.scatter(orig_tear[:, 0], orig_tear[:, 1],
               c='red', s=50, marker='s', label='Tear duct', alpha=0.5)
    ax.scatter(orig_outer[:, 0], orig_outer[:, 1],
               c='green', s=50, marker='^', label='Outer eye', alpha=0.5)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Original Coordinates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot corrected
    ax = axes[1]
    ax.scatter(corr_pupil_center[:, 0], corr_pupil_center[:, 1],
               c='blue', alpha=0.3, s=10, label='Pupil center')
    ax.scatter(corr_tear[:, 0], corr_tear[:, 1],
               c='red', s=50, marker='s', label='Tear duct', alpha=0.5)
    ax.scatter(corr_outer[:, 0], corr_outer[:, 1],
               c='green', s=50, marker='^', label='Outer eye', alpha=0.5)
    
    # Add axis labels for anatomical reference
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('X (lateral → | ← nasal)')
    ax.set_ylabel('Y (superior ↑ | ↓ inferior)')
    ax.set_title('Spatially Corrected Coordinates\n(Anatomical Reference Frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(fname=output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")
    
    plt.show()


def main() -> None:
    """Run spatial correction example."""
    # Setup paths
    base_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37"
    )
    video_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4"
    )
    timestamps_npy_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy"
    )
    csv_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )

    # Load dataset
    print("Loading eye tracking dataset...")
    eye_dataset = EyeVideoDataset.create(
        data_name="ferret_757_eye_tracking",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
        butterworth_cutoff=6.0,
        butterworth_sampling_rate=90.0
    )

    # Apply spatial correction
    print("\nApplying spatial correction...")
    print("  Step 1: Translating by tear duct position...")
    print("  Step 2: Rotating to align outer eye with X-axis...")
    print("  Step 3: Centering by pupil mode...")
    
    corrected_dataset = apply_spatial_correction_to_dataset(
        dataset=eye_dataset.pixel_trajectories,
        tear_duct_name="tear_duct",
        outer_eye_name="outer_eye",
        pupil_names=[f'p{i}' for i in range(1, 9)],
        apply_to_raw=True,
        apply_to_cleaned=True,
        n_bins=50
    )

    # Get correction summary
    print("\n" + "="*60)
    print("SPATIAL CORRECTION SUMMARY")
    print("="*60)
    
    summary = get_correction_summary(
        original_dataset=eye_dataset.pixel_trajectories,
        corrected_dataset=corrected_dataset,
        tear_duct_name="tear_duct",
        outer_eye_name="outer_eye"
    )
    
    print("\nOriginal coordinates (cleaned data):")
    print(f"  Tear duct mean:   ({summary['original_tear_duct_mean_x']:.2f}, {summary['original_tear_duct_mean_y']:.2f})")
    print(f"  Outer eye mean:   ({summary['original_outer_eye_mean_x']:.2f}, {summary['original_outer_eye_mean_y']:.2f})")
    print(f"  Pupil center mean: ({summary['original_pupil_center_mean_x']:.2f}, {summary['original_pupil_center_mean_y']:.2f})")
    
    print("\nCorrected coordinates (anatomical frame):")
    print(f"  Tear duct mean:   ({summary['corrected_tear_duct_mean_x']:.2f}, {summary['corrected_tear_duct_mean_y']:.2f})")
    print(f"  Outer eye mean:   ({summary['corrected_outer_eye_mean_x']:.2f}, {summary['corrected_outer_eye_mean_y']:.2f})")
    print(f"  Pupil center mean: ({summary['corrected_pupil_center_mean_x']:.2f}, {summary['corrected_pupil_center_mean_y']:.2f})")
    
    print("\nAnatomical coordinate system established:")
    print("  - Origin: Resting pupil position (mode)")
    print("  - X-axis: Lateral (+) ← → Nasal (-)")
    print("  - Y-axis: Superior (+) ↑ ↓ Inferior (-)")
    print("="*60)

    # Plot comparison
    print("\nGenerating comparison plot...")
    plot_correction_comparison(
        original_dataset=eye_dataset.pixel_trajectories,
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
    pupil_names = [f'p{i}' for i in range(1, 9)]
    pupil_points = [corrected_dataset.pairs[name].cleaned.data for name in pupil_names]
    pupil_center_anatomical = np.nanmean(np.stack(pupil_points, axis=1), axis=1)
    
    print(f"\nPupil center in anatomical frame (first 5 frames):")
    print(f"  Shape: {pupil_center_anatomical.shape}")
    for i in range(min(5, len(pupil_center_anatomical))):
        x, y = pupil_center_anatomical[i]
        print(f"    Frame {i}: ({x:6.2f}, {y:6.2f})")
    
    print("\nSpatial correction complete!")


if __name__ == "__main__":
    main()
