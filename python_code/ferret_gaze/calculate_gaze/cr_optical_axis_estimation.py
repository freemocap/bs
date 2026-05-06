"""
CR-Based Optical Axis Estimation (Guestrin et al. 2006)
=======================================================

Standalone analysis script that uses tracked pupil centers and corneal reflections
(CRs) to estimate the optical axis of the eye per-frame.

Method (Guestrin & Eizenman, IEEE TBME 2006):
    The cornea acts as a convex mirror. When a point light source at known position l
    illuminates the eye, it creates a glint (CR) at image position q. The 3D glint g
    lies on the ray from the camera through q:

        g = λ * k    (k = unit ray, camera at origin)

    The surface normal at the glint bisects the (glint→camera) and (glint→light) directions:

        n = normalize(normalize(l - g) + normalize(-k))

    The corneal center o_c lies along the inward normal at distance R_c:

        o_c = g - R_c * n

    With 2 light sources (cr_top, cr_bottom), we solve for (λ₀, λ₁) minimizing
    |o_c(λ₀) - o_c(λ₁)|². Once o_c is found, the pupil ray is intersected with the
    corneal sphere to get the 3D pupil position P, and the optical axis is
    normalize(P - o_c).

IMPORTANT: Results are only meaningful once CR_LED_POSITIONS_MM is filled in with
measured values from the physical camera/LED setup.
"""
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import least_squares

from python_code.ferret_gaze.calculate_gaze.calculate_ferret_gaze import (
    EYE_AZIMUTH_DEG,
    EYE_ELEVATION_DEG,
    get_eye_to_skull_rotation_matrix,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_functions import (
    compute_camera_to_eye_rotation,
    extract_frame_data,
    eye_camera_distance_from_skull_geometry,
    get_camera_centered_positions,
    load_eye_trajectories_csv,
)
from python_code.ferret_gaze.eye_kinematics.ferret_eyeball_reference_geometry import (
    FERRET_EYE_OPENING_MM,
)
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry
from python_code.utilities.folder_utilities.recording_folder import (
    PipelineStep,
    RecordingFolder,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Corneal sphere radius for ferrets (estimated; adjust from CT/MRI measurements).
# Humans: ~7.8 mm. Ferret eyes are much smaller (eyeball radius ~3 mm).
FERRET_CORNEAL_RADIUS_MM: float = 2.5

# LED (light source) positions in the camera frame.
# Camera is at the origin; the eye center is at [0, 0, camera_distance_mm].
# Axis convention matches pixels_to_camera_3d (no axis flips applied):
#   +X = pixel rightward, +Y = pixel downward, +Z = toward eye
# CR_KEYPOINT_NAMES = ["cr_top", "cr_bottom"], so index 0 = cr_top, index 1 = cr_bottom.
# THESE ARE PLACEHOLDER VALUES — measure the actual LED offsets from the hardware setup.
CR_LED_POSITIONS_MM: NDArray[np.float64] = np.array(
    [
        [0.0, -10.0, 0.0],  # cr_top LED: above camera → negative Y in image coords
        [0.0, 10.0, 0.0],   # cr_bottom LED: below camera → positive Y in image coords
    ],
    dtype=np.float64,
)


def unproject_pixel_to_ray(
    pixel_xy: NDArray[np.float64],
    eye_center_px: NDArray[np.float64],
    px_to_mm_scale: float,
    camera_distance_mm: float,
) -> NDArray[np.float64]:
    """Convert a 2D pixel position to a unit ray direction in the camera frame.

    Camera is at origin; eye center is at [0, 0, camera_distance_mm].
    Uses the same pixel-to-mm convention as pixels_to_camera_3d (no axis flips).

    Args:
        pixel_xy: (2,) pixel coordinates [x, y]
        eye_center_px: (2,) principal point in pixels
        px_to_mm_scale: mm per pixel (at the eye's focal plane)
        camera_distance_mm: distance from camera to eye center in mm

    Returns:
        (3,) unit ray direction in camera frame
    """
    dx = (pixel_xy[0] - eye_center_px[0]) * px_to_mm_scale
    dy = (pixel_xy[1] - eye_center_px[1]) * px_to_mm_scale
    ray = np.array([dx, dy, camera_distance_mm])
    return ray / np.linalg.norm(ray)


def _corneal_centers_from_lambdas(
    lambdas: NDArray[np.float64],
    k: NDArray[np.float64],
    led_positions: NDArray[np.float64],
    corneal_radius_mm: float,
) -> NDArray[np.float64]:
    """Compute corneal center estimates from glint depth scalars.

    Args:
        lambdas: (2,) depth scalars along each CR ray
        k: (2, 3) unit ray directions for each CR
        led_positions: (2, 3) LED positions in camera frame
        corneal_radius_mm: corneal sphere radius in mm

    Returns:
        (2, 3) corneal center estimates, one per glint
    """
    o_c_estimates = np.zeros((2, 3), dtype=np.float64)
    for i in range(2):
        g = lambdas[i] * k[i]
        to_led = led_positions[i] - g
        to_led_norm = np.linalg.norm(to_led)
        u = to_led / to_led_norm if to_led_norm > 1e-10 else np.array([0.0, 0.0, 1.0])
        v = -k[i]  # direction from glint toward camera (camera at origin)
        bisector = u + v
        bisector_norm = np.linalg.norm(bisector)
        n = bisector / bisector_norm if bisector_norm > 1e-10 else k[i]
        o_c_estimates[i] = g - corneal_radius_mm * n
    return o_c_estimates


def find_corneal_center(
    cr_pixels: NDArray[np.float64],
    eye_center_px: NDArray[np.float64],
    px_to_mm_scale: float,
    camera_distance_mm: float,
    led_positions: NDArray[np.float64],
    corneal_radius_mm: float,
) -> tuple[NDArray[np.float64], bool]:
    """Solve for the corneal center using the Guestrin dual-glint method.

    Args:
        cr_pixels: (2, 2) CR pixel positions [cr_top, cr_bottom]
        eye_center_px: (2,) principal point in pixels
        px_to_mm_scale: mm per pixel at eye plane
        camera_distance_mm: camera-to-eye distance in mm
        led_positions: (2, 3) LED positions in camera frame
        corneal_radius_mm: corneal sphere radius in mm

    Returns:
        o_c: (3,) corneal center in camera frame (averaged over both glints)
        converged: True if optimization residual is within 1 mm
    """
    k = np.array(
        [
            unproject_pixel_to_ray(cr_pixels[i], eye_center_px, px_to_mm_scale, camera_distance_mm)
            for i in range(2)
        ]
    )

    def residuals(lambdas: NDArray) -> NDArray:
        o_c_pair = _corneal_centers_from_lambdas(lambdas, k, led_positions, corneal_radius_mm)
        return (o_c_pair[0] - o_c_pair[1]).ravel()

    x0 = np.array([camera_distance_mm, camera_distance_mm])
    result = least_squares(
        residuals, x0, bounds=([0.0, 0.0], [np.inf, np.inf]), method="trf", max_nfev=100
    )
    converged = bool(result.success and np.linalg.norm(result.fun) < 1.0)

    o_c_pair = _corneal_centers_from_lambdas(result.x, k, led_positions, corneal_radius_mm)
    o_c = np.mean(o_c_pair, axis=0)
    return o_c, converged


def find_pupil_on_cornea(
    pupil_px: NDArray[np.float64],
    eye_center_px: NDArray[np.float64],
    px_to_mm_scale: float,
    camera_distance_mm: float,
    o_c: NDArray[np.float64],
    corneal_radius_mm: float,
) -> NDArray[np.float64] | None:
    """Find the 3D pupil position by intersecting the pupil ray with the corneal sphere.

    Args:
        pupil_px: (2,) pupil center pixel position
        eye_center_px: (2,) principal point in pixels
        px_to_mm_scale: mm per pixel at eye plane
        camera_distance_mm: camera-to-eye distance in mm
        o_c: (3,) corneal center in camera frame
        corneal_radius_mm: corneal sphere radius in mm

    Returns:
        (3,) 3D pupil position in camera frame, or None if no valid intersection
    """
    p_ray = unproject_pixel_to_ray(pupil_px, eye_center_px, px_to_mm_scale, camera_distance_mm)

    # Solve |t * p_ray - o_c|² = R_c² (quadratic in t, with |p_ray|=1 so a=1)
    b = -2.0 * np.dot(p_ray, o_c)
    c = float(np.dot(o_c, o_c)) - corneal_radius_mm**2
    discriminant = b**2 - 4.0 * c

    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    # Take the smallest positive root (front of cornea, closest to camera)
    if t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    else:
        return None

    return t * p_ray


def estimate_optical_axis_per_frame(
    pupil_pixels: NDArray[np.float64],
    cr_pixels: NDArray[np.float64],
    eye_center_px: NDArray[np.float64],
    px_to_mm_scale: float,
    camera_distance_mm: float,
    led_positions: NDArray[np.float64] = CR_LED_POSITIONS_MM,
    corneal_radius_mm: float = FERRET_CORNEAL_RADIUS_MM,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Estimate the optical axis per frame using the Guestrin dual-glint method.

    Args:
        pupil_pixels: (N, 2) pupil center pixel positions
        cr_pixels: (N, 2, 2) CR pixel positions [cr_top, cr_bottom] per frame
        eye_center_px: (2,) principal point in pixels
        px_to_mm_scale: mm per pixel at eye plane
        camera_distance_mm: camera-to-eye distance in mm
        led_positions: (2, 3) LED positions in camera frame
        corneal_radius_mm: corneal sphere radius in mm

    Returns:
        optical_axes_cam: (N, 3) unit optical axis vectors in camera frame
        corneal_centers_cam: (N, 3) corneal center positions in camera frame
        converged: (N,) bool, True where optimization converged within tolerance
    """
    n_frames = len(pupil_pixels)
    optical_axes = np.zeros((n_frames, 3), dtype=np.float64)
    corneal_centers = np.zeros((n_frames, 3), dtype=np.float64)
    converged = np.zeros(n_frames, dtype=bool)

    fallback_axis = np.array([0.0, 0.0, 1.0])  # straight toward camera

    for i in range(n_frames):
        try:
            o_c, did_converge = find_corneal_center(
                cr_pixels[i],
                eye_center_px,
                px_to_mm_scale,
                camera_distance_mm,
                led_positions,
                corneal_radius_mm,
            )
            corneal_centers[i] = o_c
            converged[i] = did_converge

            P = find_pupil_on_cornea(
                pupil_pixels[i],
                eye_center_px,
                px_to_mm_scale,
                camera_distance_mm,
                o_c,
                corneal_radius_mm,
            )

            if P is None:
                optical_axes[i] = fallback_axis
                converged[i] = False
            else:
                axis = P - o_c
                norm = np.linalg.norm(axis)
                optical_axes[i] = axis / norm if norm > 1e-10 else fallback_axis

        except Exception:
            optical_axes[i] = fallback_axis
            converged[i] = False

    return optical_axes, corneal_centers, converged


def compute_azimuth_elevation_deg(
    direction_skull: NDArray[np.float64],
) -> tuple[float, float]:
    """Compute azimuth and elevation of a skull-frame direction vector.

    Skull frame: +X = nose, +Y = toward left eye, +Z = superior.
    Azimuth is measured from +X in the horizontal (XY) plane toward the lateral side
    (always positive, 0–90°, symmetric for both eyes via |Y|).
    Elevation is the angle above horizontal.

    Args:
        direction_skull: (3,) unit vector in skull frame

    Returns:
        azimuth_deg: degrees from nose (+X) in horizontal plane
        elevation_deg: degrees above horizontal
    """
    x, y, z = float(direction_skull[0]), float(direction_skull[1]), float(direction_skull[2])
    azimuth_deg = float(np.degrees(np.arctan2(abs(y), x)))
    elevation_deg = float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))
    return azimuth_deg, elevation_deg


def run_cr_optical_axis_analysis(
    recording_folder: RecordingFolder,
    eye_name: Literal["left_eye", "right_eye"],
    led_positions_mm: NDArray[np.float64] = CR_LED_POSITIONS_MM,
    corneal_radius_mm: float = FERRET_CORNEAL_RADIUS_MM,
) -> pl.DataFrame:
    """Run CR-based optical axis estimation and save per-frame results to CSV.

    Loads the raw DLC eye tracking CSV, applies the Guestrin dual-glint method
    per frame, transforms optical axes to skull coordinates, and computes angular
    deviation from the anatomical mounting constants (EYE_AZIMUTH_DEG / EYE_ELEVATION_DEG).

    Output CSV is written to:
        <recording_folder>/analyzable_output/cr_optical_axis/<eye_name>_cr_optical_axis.csv

    Args:
        recording_folder: RecordingFolder at SKULL_POST_PROCESSED stage or later
        eye_name: "left_eye" or "right_eye"
        led_positions_mm: (2, 3) LED positions in camera frame (index 0=cr_top, 1=cr_bottom)
        corneal_radius_mm: ferret corneal sphere radius in mm

    Returns:
        polars DataFrame with per-frame results (also written to disk)
    """
    logger.warning(
        "CR_LED_POSITIONS_MM contains placeholder values. "
        "Measure actual LED offsets from the physical camera setup before interpreting results."
    )

    eye_side: Literal["left", "right"] = "left" if eye_name == "left_eye" else "right"

    eye_csv_path = (
        recording_folder.left_eye_data_csv
        if eye_name == "left_eye"
        else recording_folder.right_eye_data_csv
    )
    if eye_csv_path is None:
        raise ValueError(f"Eye data CSV not found for {eye_name} in {recording_folder.folder_path}")

    skull_reference_geometry_path = recording_folder.skull_reference_geometry
    if skull_reference_geometry_path is None:
        raise ValueError(
            f"skull_reference_geometry.json not found in {recording_folder.folder_path}"
        )

    output_csv_path = (
        recording_folder.folder_path
        / "analyzable_output"
        / "cr_optical_axis"
        / f"{eye_name}_cr_optical_axis.csv"
    )

    skull_reference_geometry = ReferenceGeometry.from_json_file(skull_reference_geometry_path)
    eye_camera_distance_mm = eye_camera_distance_from_skull_geometry(skull_reference_geometry, eye_side)
    logger.info(f"Eye-camera distance ({eye_side}): {eye_camera_distance_mm:.1f} mm")

    df = load_eye_trajectories_csv(eye_csv_path, eye_side, video_name=eye_name)

    # Get camera-frame data (gaze directions, socket landmarks) from existing pipeline
    (
        _,
        gaze_directions_cam,
        _,
        _,
        _,
        outer_eye_cam,
        tear_duct_cam,
        timestamps,
    ) = get_camera_centered_positions(
        df=df, eye_name=eye_name, eye_camera_distance_mm=eye_camera_distance_mm
    )

    # Get raw pixel data for pupil and CRs (same df, no sphere projection)
    _, pupil_centers_px, _, tear_duct_px, outer_eye_px, cr_points_px = extract_frame_data(df)

    # Compute pixel calibration (same as existing pipeline)
    mean_tear_duct_px = np.mean(tear_duct_px, axis=0)
    mean_outer_eye_px = np.mean(outer_eye_px, axis=0)
    eye_center_px = (mean_tear_duct_px + mean_outer_eye_px) / 2.0
    tear_duct_to_outer_px = np.linalg.norm(mean_outer_eye_px - mean_tear_duct_px)
    px_to_mm_scale = FERRET_EYE_OPENING_MM / tear_duct_to_outer_px

    # Compute R_camera_to_eye (mirrors process_ferret_eye_data)
    rest_gaze_cam = np.median(gaze_directions_cam, axis=0)
    rest_gaze_cam /= np.linalg.norm(rest_gaze_cam)

    mean_tear_duct_cam = np.mean(tear_duct_cam, axis=0)
    mean_outer_eye_cam = np.mean(outer_eye_cam, axis=0)
    eye_opening_dir = mean_outer_eye_cam - mean_tear_duct_cam
    if eye_side == "right":
        eye_opening_dir = -eye_opening_dir
    y_approx = np.cross(rest_gaze_cam, eye_opening_dir)
    y_norm = np.linalg.norm(y_approx)
    y_approx = y_approx / y_norm if y_norm > 1e-10 else np.array([0.0, -1.0, 0.0])

    R_camera_to_eye = compute_camera_to_eye_rotation(rest_gaze_cam, y_approx)
    R_eye_to_skull = get_eye_to_skull_rotation_matrix(eye_side)

    # Run Guestrin method on raw CR pixel data
    optical_axes_cam, corneal_centers_cam, converged = estimate_optical_axis_per_frame(
        pupil_pixels=pupil_centers_px,
        cr_pixels=cr_points_px,
        eye_center_px=eye_center_px,
        px_to_mm_scale=px_to_mm_scale,
        camera_distance_mm=eye_camera_distance_mm,
        led_positions=led_positions_mm,
        corneal_radius_mm=corneal_radius_mm,
    )

    # Camera → eye → skull
    optical_axes_eye = (R_camera_to_eye @ optical_axes_cam.T).T
    optical_axes_eye /= np.maximum(np.linalg.norm(optical_axes_eye, axis=1, keepdims=True), 1e-10)

    optical_axes_skull = (R_eye_to_skull @ optical_axes_eye.T).T
    optical_axes_skull /= np.maximum(np.linalg.norm(optical_axes_skull, axis=1, keepdims=True), 1e-10)

    # Per-frame azimuth / elevation and deltas from anatomical constants
    n_frames = len(timestamps)
    azimuths = np.zeros(n_frames)
    elevations = np.zeros(n_frames)
    for i in range(n_frames):
        azimuths[i], elevations[i] = compute_azimuth_elevation_deg(optical_axes_skull[i])
    delta_azimuths = azimuths - EYE_AZIMUTH_DEG
    delta_elevations = elevations - EYE_ELEVATION_DEG

    # Rest gaze from converged frames
    converged_mask = converged
    n_converged = int(np.sum(converged_mask))
    if n_converged > 0:
        rest_axis = np.median(optical_axes_skull[converged_mask], axis=0)
        rest_axis /= np.linalg.norm(rest_axis)
        rest_azimuth, rest_elevation = compute_azimuth_elevation_deg(rest_axis)
    else:
        rest_azimuth, rest_elevation = float("nan"), float("nan")

    logger.info(
        f"Converged: {n_converged}/{n_frames} frames ({100 * n_converged / n_frames:.1f}%)"
    )
    logger.info(
        f"Rest gaze — azimuth: {rest_azimuth:.2f}° (target {EYE_AZIMUTH_DEG}°), "
        f"elevation: {rest_elevation:.2f}° (target {EYE_ELEVATION_DEG}°)"
    )
    logger.info(
        f"Delta — azimuth: {rest_azimuth - EYE_AZIMUTH_DEG:+.2f}°, "
        f"elevation: {rest_elevation - EYE_ELEVATION_DEG:+.2f}°"
    )

    df_out = pl.DataFrame(
        {
            "frame": list(range(n_frames)),
            "timestamp": timestamps.tolist(),
            "optical_axis_cam_x": optical_axes_cam[:, 0].tolist(),
            "optical_axis_cam_y": optical_axes_cam[:, 1].tolist(),
            "optical_axis_cam_z": optical_axes_cam[:, 2].tolist(),
            "optical_axis_skull_x": optical_axes_skull[:, 0].tolist(),
            "optical_axis_skull_y": optical_axes_skull[:, 1].tolist(),
            "optical_axis_skull_z": optical_axes_skull[:, 2].tolist(),
            "azimuth_deg": azimuths.tolist(),
            "elevation_deg": elevations.tolist(),
            "delta_azimuth_deg": delta_azimuths.tolist(),
            "delta_elevation_deg": delta_elevations.tolist(),
            "corneal_center_cam_x": corneal_centers_cam[:, 0].tolist(),
            "corneal_center_cam_y": corneal_centers_cam[:, 1].tolist(),
            "corneal_center_cam_z": corneal_centers_cam[:, 2].tolist(),
            "converged": converged.tolist(),
        }
    )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_csv(output_path)
    logger.info(f"Saved to {output_path}")

    return df_out


def run_cr_optical_axis_analysis_all_eyes(
    recording_folder: RecordingFolder,
    led_positions_mm: NDArray[np.float64] = CR_LED_POSITIONS_MM,
    corneal_radius_mm: float = FERRET_CORNEAL_RADIUS_MM,
) -> dict[str, pl.DataFrame]:
    """Run CR-based optical axis estimation for both eyes.

    Args:
        recording_folder: RecordingFolder at SKULL_POST_PROCESSED stage or later
        led_positions_mm: (2, 3) LED positions in camera frame (index 0=cr_top, 1=cr_bottom)
        corneal_radius_mm: ferret corneal sphere radius in mm

    Returns:
        Dict mapping eye name to per-frame results DataFrame
    """
    results = {}
    for eye_name in ("left_eye", "right_eye"):
        logger.info(f"Processing {eye_name}...")
        results[eye_name] = run_cr_optical_axis_analysis(
            recording_folder=recording_folder,
            eye_name=eye_name,
            led_positions_mm=led_positions_mm,
            corneal_radius_mm=corneal_radius_mm,
        )
    return results


if __name__ == "__main__":
    RECORDING_PATH = Path("/home/scholl-lab/ferret_recordings/session_2025-07-09_ferret_757_EyeCameras_P41_E13/full_recording")

    recording_folder = RecordingFolder.from_folder_path(
        RECORDING_PATH, expected_processing_step=PipelineStep.SKULL_POST_PROCESSED
    )
    results = run_cr_optical_axis_analysis_all_eyes(recording_folder=recording_folder)
    for eye_name, df in results.items():
        print(f"\n{eye_name}:")
        print(df.head())
