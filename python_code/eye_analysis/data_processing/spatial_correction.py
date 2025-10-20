"""Spatial correction for eye tracking data to establish anatomical coordinate system.

Applies transformations to align data with anatomical axes:
1. Translate: tear_duct → origin (0,0)
2. Rotate: eye_outer → X-axis (lateral-nasal alignment)
3. Center: mode of pupil center → origin (resting position)
"""

import numpy as np
from scipy.stats import mode as scipy_mode

from python_code.eye_analysis.trajectory_dataset import Trajectory2D, TrajectoryPair, TrajectoryDataset


def compute_histogram_mode(*, data: np.ndarray, n_bins: int = 50) -> float:
    """Compute mode of continuous data using histogram binning.
    
    Args:
        data: 1D array of values
        n_bins: Number of histogram bins
        
    Returns:
        Mode value (center of most frequent bin)
    """
    # Remove NaN values
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return 0.0
    
    hist, edges = np.histogram(valid_data, bins=n_bins)
    mode_idx = np.argmax(hist)
    
    # Return center of the modal bin
    return (edges[mode_idx] + edges[mode_idx + 1]) / 2.0


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
    n_frames = len(trajectory.data)
    corrected_data = np.zeros_like(trajectory.data)
    
    for i in range(n_frames):
        # Step 1: Translate by tear duct
        translated = trajectory.data[i] - tear_duct_positions[i]
        
        # Step 2: Rotate to align eye_outer with X-axis
        angle = rotation_angles[i]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        rotated = rotation_matrix @ translated
        
        # Step 3: Shift by mode offset
        corrected_data[i] = rotated - mode_offset
    
    return Trajectory2D(
        name=trajectory.name,
        data=corrected_data,
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
    n_frames = len(tear_duct_trajectory.data)
    
    # Step 1: Get tear duct positions for translation
    tear_duct_positions = tear_duct_trajectory.data.copy()
    
    # Step 2: Compute rotation angles
    # After translating by tear duct, find angle of eye_outer
    outer_eye_translated = outer_eye_trajectory.data - tear_duct_positions
    rotation_angles = -np.arctan2(outer_eye_translated[:, 1], outer_eye_translated[:, 0])
    
    # Step 3: Compute mode offset from pupil centers
    # First translate and rotate all pupil trajectories
    pupil_centers_rotated = []
    
    for frame_idx in range(n_frames):
        frame_points = []
        for pupil_traj in pupil_trajectories:
            # Translate
            translated = pupil_traj.data[frame_idx] - tear_duct_positions[frame_idx]
            
            # Rotate
            angle = rotation_angles[frame_idx]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            rotated = rotation_matrix @ translated
            
            frame_points.append(rotated)
        
        # Average the pupil points for this frame
        if frame_points:
            pupil_centers_rotated.append(np.mean(frame_points, axis=0))
    
    pupil_centers_rotated = np.array(pupil_centers_rotated)  # (n_frames, 2)
    
    # Compute mode for X and Y separately
    x_mode = compute_histogram_mode(data=pupil_centers_rotated[:, 0], n_bins=n_bins)
    y_mode = compute_histogram_mode(data=pupil_centers_rotated[:, 1], n_bins=n_bins)
    mode_offset = np.array([x_mode, y_mode])
    
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
    required = [tear_duct_name, outer_eye_name] + pupil_names
    missing = [name for name in required if name not in dataset.pairs]
    if missing:
        raise ValueError(f"Missing required landmarks: {missing}")
    
    corrected_pairs = {}
    
    for name, pair in dataset.pairs.items():
        # Process raw trajectory
        if apply_to_raw:
            # Compute correction parameters from raw data
            tear_duct_traj = dataset.pairs[tear_duct_name].raw
            outer_eye_traj = dataset.pairs[outer_eye_name].raw
            pupil_trajs = [dataset.pairs[pname].raw for pname in pupil_names]
            
            tear_duct_pos, rotation_angles, mode_offset = compute_spatial_correction_parameters(
                tear_duct_trajectory=tear_duct_traj,
                outer_eye_trajectory=outer_eye_traj,
                pupil_trajectories=pupil_trajs,
                n_bins=n_bins
            )
            
            corrected_raw = apply_spatial_correction_to_trajectory(
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
            tear_duct_traj = dataset.pairs[tear_duct_name].cleaned
            outer_eye_traj = dataset.pairs[outer_eye_name].cleaned
            pupil_trajs = [dataset.pairs[pname].cleaned for pname in pupil_names]
            
            tear_duct_pos, rotation_angles, mode_offset = compute_spatial_correction_parameters(
                tear_duct_trajectory=tear_duct_traj,
                outer_eye_trajectory=outer_eye_traj,
                pupil_trajectories=pupil_trajs,
                n_bins=n_bins
            )
            
            corrected_cleaned = apply_spatial_correction_to_trajectory(
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
    orig_tear = original_dataset.pairs[tear_duct_name].cleaned
    corr_tear = corrected_dataset.pairs[tear_duct_name].cleaned
    
    orig_outer = original_dataset.pairs[outer_eye_name].cleaned
    corr_outer = corrected_dataset.pairs[outer_eye_name].cleaned
    
    # Compute average positions
    orig_tear_mean = np.nanmean(orig_tear.data, axis=0)
    corr_tear_mean = np.nanmean(corr_tear.data, axis=0)
    
    orig_outer_mean = np.nanmean(orig_outer.data, axis=0)
    corr_outer_mean = np.nanmean(corr_outer.data, axis=0)
    
    # Compute pupil center
    pupil_names = [f'p{i}' for i in range(1, 9)]
    orig_pupil_points = [original_dataset.pairs[name].cleaned.data for name in pupil_names]
    orig_pupil_center = np.nanmean(np.stack(orig_pupil_points, axis=1), axis=1)
    
    corr_pupil_points = [corrected_dataset.pairs[name].cleaned.data for name in pupil_names]
    corr_pupil_center = np.nanmean(np.stack(corr_pupil_points, axis=1), axis=1)
    
    orig_pupil_mean = np.nanmean(orig_pupil_center, axis=0)
    corr_pupil_mean = np.nanmean(corr_pupil_center, axis=0)
    
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
