"""Reference geometry saving and trajectory verification utilities."""

import json
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_reference_geometry_json(
        *,
        filepath: Path,
        reference_geometry: np.ndarray,
        marker_names: list[str],
        units: str = "mm"
) -> None:
    """
    Save reference geometry as JSON mapping marker names to xyz coordinates.

    Args:
        filepath: Output JSON file path
        reference_geometry: (n_markers, 3) reference positions in body frame (meters)
        marker_names: List of marker names
        units: Units to save ("mm" or "m")
    """
    scale = 1000.0 if units == "mm" else 1.0

    geometry_dict = {}
    for marker_name, position in zip(marker_names, reference_geometry):
        geometry_dict[marker_name] = {
            "x": float(position[0] * scale),
            "y": float(position[1] * scale),
            "z": float(position[2] * scale),
            "units": units
        }

    output = {
        "units": units,
        "coordinate_system": {
            "description": "Body-fixed frame",
            "origin": "Mean of origin markers",
            "x_axis": "Points from origin to x_axis_marker",
            "y_axis": "Points generally toward y_axis_marker (orthogonalized)",
            "z_axis": "Cross product of y and x axes"
        },
        "markers": geometry_dict,
        "n_markers": len(marker_names)
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved reference geometry to: {filepath}")
    logger.info(f"  Units: {units}")
    logger.info(f"  Markers: {len(marker_names)}")


def verify_trajectory_reconstruction(
        *,
        reference_geometry: np.ndarray,
        rotations: np.ndarray,
        translations: np.ndarray,
        reconstructed: np.ndarray,
        marker_names: list[str],
        n_frames_to_check: int = 5,
        tolerance: float = 1e-6
) -> bool:
    """
    Verify that reconstructed trajectories match: world = R @ reference + t

    Args:
        reference_geometry: (n_markers, 3) reference positions in body frame
        rotations: (n_frames, 3, 3) rotation matrices
        translations: (n_frames, 3) translation vectors
        reconstructed: (n_frames, n_markers, 3) reconstructed positions
        marker_names: List of marker names
        n_frames_to_check: Number of frames to verify
        tolerance: Maximum allowed error (meters)

    Returns:
        True if verification passes, False otherwise
    """
    n_frames = min(n_frames_to_check, rotations.shape[0])
    n_markers = reference_geometry.shape[0]

    logger.info("=" * 80)
    logger.info("VERIFYING TRAJECTORY RECONSTRUCTION")
    logger.info("=" * 80)
    logger.info(f"Formula: world = R @ reference + t")
    logger.info(f"Checking {n_frames} frames, {n_markers} markers")
    logger.info(f"Tolerance: {tolerance * 1000:.4f} mm")

    max_error = 0.0
    max_error_frame = -1
    max_error_marker = -1

    for frame_idx in range(n_frames):
        R = rotations[frame_idx]
        t = translations[frame_idx]

        # Recompute reconstruction: world = R @ reference + t
        expected = (R @ reference_geometry.T).T + t
        actual = reconstructed[frame_idx]

        # Compute errors
        errors = np.linalg.norm(expected - actual, axis=1)

        # Track maximum error
        frame_max_error = errors.max()
        if frame_max_error > max_error:
            max_error = frame_max_error
            max_error_frame = frame_idx
            max_error_marker = errors.argmax()

        # Log first few frames
        if frame_idx < 3:
            logger.info(f"\nFrame {frame_idx}:")
            logger.info(f"  Max error: {frame_max_error * 1000:.6f} mm (marker: {marker_names[errors.argmax()]})")
            logger.info(f"  Mean error: {errors.mean() * 1000:.6f} mm")

    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Frames checked: {n_frames}")
    logger.info(f"Max error: {max_error * 1000:.6f} mm")
    logger.info(f"  Frame: {max_error_frame}")
    logger.info(f"  Marker: {marker_names[max_error_marker]}")

    if max_error < tolerance:
        logger.info(f"✓ PASS: All errors < {tolerance * 1000:.4f} mm")
        return True
    else:
        logger.warning(f"✗ FAIL: Max error {max_error * 1000:.6f} mm exceeds tolerance {tolerance * 1000:.4f} mm")
        return False


def print_reference_geometry_summary(
        *,
        reference_geometry: np.ndarray,
        marker_names: list[str],
        units: str = "mm"
) -> None:
    """
    Print a summary of the reference geometry.

    Args:
        reference_geometry: (n_markers, 3) positions
        marker_names: List of marker names
        units: Display units ("mm" or "m")
    """
    scale = 1000.0 if units == "mm" else 1.0

    logger.info("=" * 80)
    logger.info("REFERENCE GEOMETRY SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Units: {units}")
    logger.info(f"Coordinate system: Body-fixed frame")
    logger.info("\nMarker positions:")
    logger.info(f"{'Name':<20} {'X':>10} {'Y':>10} {'Z':>10}")
    logger.info("-" * 80)

    for name, pos in zip(marker_names, reference_geometry):
        x, y, z = pos * scale
        logger.info(f"{name:<20} {x:>10.3f} {y:>10.3f} {z:>10.3f}")

    # Compute bounding box
    min_vals = reference_geometry.min(axis=0) * scale
    max_vals = reference_geometry.max(axis=0) * scale
    size = max_vals - min_vals

    logger.info("\nBounding box:")
    logger.info(f"  X: [{min_vals[0]:.3f}, {max_vals[1]:.3f}] {units} (size: {size[0]:.3f} {units})")
    logger.info(f"  Y: [{min_vals[1]:.3f}, {max_vals[1]:.3f}] {units} (size: {size[1]:.3f} {units})")
    logger.info(f"  Z: [{min_vals[2]:.3f}, {max_vals[2]:.3f}] {units} (size: {size[2]:.3f} {units})")
