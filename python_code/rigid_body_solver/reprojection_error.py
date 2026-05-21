"""Compute reprojection error of skull solver output onto 2D DLC-tracked points."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import polars as pl
import toml
from numpy.typing import NDArray

from python_code.kinematics_core.rigid_body_kinematics_model import RigidBodyKinematics
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

logger = logging.getLogger(__name__)

SKULL_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "base",
    "left_cam_tip",
    "right_cam_tip",
]
CAMERA_IDS = ["24676894", "24908831", "24908832", "25000609", "25006505"]
INTRINSICS_PATH = Path(__file__).parent.parent / "cameras" / "intrinsics" / "intrinsics.json"
DLC_CONFIDENCE_THRESHOLD = 0.8


def _extract_camera_id_from_filename(filename: str) -> str | None:
    for camera_id in CAMERA_IDS:
        if camera_id in filename:
            return camera_id
    return None


def load_optimized_skull_keypoints(
    solver_output_dir: Path,
) -> tuple[NDArray[np.float64], list[str], NDArray[np.float64]]:
    """
    Load optimized skull keypoints from solver output.

    Returns:
        trajectories: (n_frames, n_skull_keypoints, 3) array in mm
        keypoint_names: ordered list of skull keypoint names
        timestamps: (n_frames,) array in seconds
    """
    skull_kinematics_csv = solver_output_dir / "skull_kinematics.csv"
    skull_reference_geometry_json = solver_output_dir / "skull_reference_geometry.json"

    if not skull_kinematics_csv.exists():
        raise FileNotFoundError(f"Skull kinematics CSV not found: {skull_kinematics_csv}")
    if not skull_reference_geometry_json.exists():
        raise FileNotFoundError(f"Skull reference geometry not found: {skull_reference_geometry_json}")

    skull_kinematics = RigidBodyKinematics.load_from_disk(
        kinematics_csv_path=skull_kinematics_csv,
        reference_geometry_json_path=skull_reference_geometry_json,
    )

    keypoint_trajectories = skull_kinematics.keypoint_trajectories
    keypoint_names = list(keypoint_trajectories.keypoint_names)
    trajectories = keypoint_trajectories.trajectories_fr_id_xyz

    logger.info(f"Loaded skull kinematics: {skull_kinematics.n_frames} frames")
    logger.info(f"  Keypoints: {keypoint_names}")

    return trajectories, keypoint_names, skull_kinematics.timestamps


def load_dlc_2d_points(
    dlc_output_dir: Path,
) -> tuple[dict[str, NDArray[np.float64]], list[str]]:
    """
    Load 2D DLC-tracked points per camera.

    Returns:
        points_by_camera: dict[camera_id, (n_frames, n_keypoints, 3)] where dim-3 = [x, y, confidence]
        keypoint_names: ordered list of DLC keypoint names from the CSV headers
    """
    csv_files = sorted(dlc_output_dir.glob("*snapshot*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No DLC snapshot CSVs found in: {dlc_output_dir}")

    points_by_camera: dict[str, NDArray[np.float64]] = {}
    keypoint_names: list[str] | None = None

    for csv_file in csv_files:
        camera_id = _extract_camera_id_from_filename(csv_file.name)
        if camera_id is None:
            logger.warning(f"Could not extract camera ID from: {csv_file.name}, skipping")
            continue

        df = pd.read_csv(csv_file, header=[1, 2])
        df = df.iloc[:, 1:]  # drop the first column (row-index header)

        if df.shape[1] % 3 != 0:
            logger.warning(f"Unexpected column count in {csv_file.name}: {df.shape[1]}, skipping")
            continue

        if keypoint_names is None:
            # Multi-index column level 0 has keypoint names, repeating every 3 columns
            keypoint_names = [df.columns[i][0] for i in range(0, df.shape[1], 3)]

        data = df.values.reshape(df.shape[0], df.shape[1] // 3, 3).astype(np.float64)
        points_by_camera[camera_id] = data
        logger.info(f"  Loaded DLC data for camera {camera_id}: {data.shape[0]} frames, {data.shape[1]} keypoints")

    if keypoint_names is None:
        raise ValueError("No valid DLC CSV files found")

    return points_by_camera, keypoint_names


def load_camera_calibration(calibration_toml_path: Path) -> dict[str, dict]:
    """
    Load camera extrinsics from calibration TOML.

    Returns:
        dict[camera_id, {"world_position": ndarray (3,), "world_orientation": ndarray (3,3), "matrix": ndarray (3,3)}]
        where world_position is the camera center in world coordinates and
        world_orientation is the camera-to-world rotation matrix (3x3).
    """
    calibration = toml.load(calibration_toml_path)
    result: dict[str, dict] = {}

    for key, value in calibration.items():
        if not key.startswith("cam_"):
            continue
        camera_id = str(value["name"])
        result[camera_id] = {
            "world_position": np.array(value["world_position"], dtype=np.float64),
            "world_orientation": np.array(value["world_orientation"], dtype=np.float64),
            "matrix": np.array(value["matrix"], dtype=np.float64),
        }
        logger.info(f"  Loaded extrinsics for camera {camera_id}")

    return result


def load_camera_intrinsics() -> dict[str, dict]:
    """
    Load camera intrinsics from the bundled intrinsics.json.

    Returns:
        dict[camera_id, {"camera_matrix": ndarray (3,3), "distortion_coefficients": ndarray (5,)}]
    """
    with open(INTRINSICS_PATH) as f:
        data = json.load(f)

    return {
        cam_id: {
            "camera_matrix": np.array(cam_data["camera_matrix"], dtype=np.float64),
            "distortion_coefficients": np.array(cam_data["distortion_coefficients"], dtype=np.float64),
        }
        for cam_id, cam_data in data.items()
    }


def project_points_to_camera(
    points_3d_world: NDArray[np.float64],
    world_position: NDArray[np.float64],
    world_orientation: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
    distortion_coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Project 3D world-space points to 2D pixel coordinates for one camera.

    Args:
        points_3d_world: (n, 3) points in world space (mm)
        world_position: (3,) camera center in world space (mm)
        world_orientation: (3, 3) camera-to-world rotation matrix
        camera_matrix: (3, 3) intrinsic camera matrix
        distortion_coefficients: (5,) distortion coefficients

    Returns:
        (n, 2) projected pixel coordinates
    """
    # world_orientation is camera-to-world (R_cw); invert to get world-to-camera (R_wc)
    R_wc = world_orientation.T
    tvec = (-R_wc @ world_position).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R_wc)

    projected, _ = cv2.projectPoints(
        points_3d_world.astype(np.float64).reshape(-1, 1, 3),
        rvec,
        tvec,
        camera_matrix,
        distortion_coefficients,
    )
    return projected.reshape(-1, 2)


def compute_reprojection_errors(
    solver_output_dir: Path,
    dlc_output_dir: Path,
    calibration_toml_path: Path,
) -> tuple[NDArray[np.float64], list[str], list[str], NDArray[np.float64]]:
    """
    Compute per-frame, per-camera, per-skull-keypoint reprojection errors.

    NaN is used where the DLC confidence is below the threshold or the 3D point is NaN.

    Returns:
        errors: (n_frames, n_cameras, n_skull_keypoints) array of errors in pixels
        skull_keypoint_names: list of skull keypoint names in order
        camera_ids: list of camera IDs in the same order as errors axis-1
        timestamps: (n_frames,) array of timestamps in seconds
    """
    logger.info("Loading data for reprojection error computation...")

    trajectories, keypoint_names_from_solver, timestamps = load_optimized_skull_keypoints(solver_output_dir)
    points_by_camera, dlc_keypoint_names = load_dlc_2d_points(dlc_output_dir)
    calibration = load_camera_calibration(calibration_toml_path)
    intrinsics = load_camera_intrinsics()

    # Determine which solver keypoints are skull keypoints and their column indices
    skull_indices_in_solver = []
    skull_keypoint_names: list[str] = []
    for name in SKULL_KEYPOINTS:
        if name in keypoint_names_from_solver:
            skull_indices_in_solver.append(keypoint_names_from_solver.index(name))
            skull_keypoint_names.append(name)
        else:
            logger.warning(f"Skull keypoint '{name}' not found in solver output, skipping")

    # Extract skull-only trajectories: (n_frames, n_skull_keypoints, 3)
    skull_trajectories = trajectories[:, skull_indices_in_solver, :]
    n_frames = skull_trajectories.shape[0]
    n_skull_keypoints = len(skull_keypoint_names)

    # Build ordered list of cameras that have both calibration data and DLC data
    camera_ids = [
        cam_id for cam_id in CAMERA_IDS
        if cam_id in points_by_camera and cam_id in calibration
    ]
    n_cameras = len(camera_ids)

    logger.info(f"Computing reprojection errors: {n_frames} frames, {n_cameras} cameras, {n_skull_keypoints} keypoints")

    errors = np.full((n_frames, n_cameras, n_skull_keypoints), np.nan, dtype=np.float64)

    for cam_idx, camera_id in enumerate(camera_ids):
        dlc_data = points_by_camera[camera_id]  # (n_dlc_frames, n_dlc_keypoints, 3)
        cal = calibration[camera_id]
        intr = intrinsics.get(camera_id)
        if intr is None:
            logger.warning(f"No intrinsics found for camera {camera_id}, skipping")
            continue

        # Match DLC frames to solver frames (they should both be n_frames from the same cameras)
        n_dlc_frames = dlc_data.shape[0]
        if n_dlc_frames != n_frames:
            logger.warning(
                f"Camera {camera_id}: DLC has {n_dlc_frames} frames but solver has {n_frames}. "
                "Using min of the two."
            )
        usable_frames = min(n_frames, n_dlc_frames)

        # Find matching DLC keypoint indices for each skull keypoint
        dlc_keypoint_indices = []
        for name in skull_keypoint_names:
            if name in dlc_keypoint_names:
                dlc_keypoint_indices.append(dlc_keypoint_names.index(name))
            else:
                dlc_keypoint_indices.append(None)
                logger.warning(f"Skull keypoint '{name}' not found in DLC CSV for camera {camera_id}")

        # Project skull keypoints to this camera for all frames at once
        # skull_trajectories: (n_frames, n_skull_keypoints, 3)
        # Flatten to (n_frames * n_skull_keypoints, 3), project, then reshape
        skull_pts_flat = skull_trajectories[:usable_frames].reshape(-1, 3)
        projected_flat = project_points_to_camera(
            points_3d_world=skull_pts_flat,
            world_position=cal["world_position"],
            world_orientation=cal["world_orientation"],
            camera_matrix=intr["camera_matrix"],
            distortion_coefficients=intr["distortion_coefficients"],
        )
        projected = projected_flat.reshape(usable_frames, n_skull_keypoints, 2)

        # Compute per-keypoint errors
        for kp_idx, (skull_kp_name, dlc_kp_idx) in enumerate(zip(skull_keypoint_names, dlc_keypoint_indices)):
            if dlc_kp_idx is None:
                continue

            observed_xy = dlc_data[:usable_frames, dlc_kp_idx, :2]      # (usable_frames, 2)
            confidence = dlc_data[:usable_frames, dlc_kp_idx, 2]         # (usable_frames,)
            proj_xy = projected[:, kp_idx, :]                             # (usable_frames, 2)

            # Compute Euclidean distance in pixels
            diff = proj_xy - observed_xy
            pixel_errors = np.linalg.norm(diff, axis=1)                  # (usable_frames,)

            # Mask low-confidence DLC detections
            low_conf_mask = confidence < DLC_CONFIDENCE_THRESHOLD
            pixel_errors[low_conf_mask] = np.nan

            # Mask frames where 3D point was NaN (triangulation failure)
            solver_nan_mask = np.any(np.isnan(skull_trajectories[:usable_frames, kp_idx, :]), axis=1)
            pixel_errors[solver_nan_mask] = np.nan

            errors[:usable_frames, cam_idx, kp_idx] = pixel_errors

    return errors, skull_keypoint_names, camera_ids, timestamps


def build_reprojection_error_df(
    errors: NDArray[np.float64],
    skull_keypoint_names: list[str],
    camera_ids: list[str],
    timestamps: NDArray[np.float64],
    use_timestamp_s: bool = False,
) -> pl.DataFrame:
    """
    Build tidy DataFrame with per-camera/keypoint reprojection errors and aggregate means.

    Row types per frame:
    - (camera_id, keypoint_name): raw error for that camera/keypoint
    - ("mean", keypoint_name): nanmean across cameras for that keypoint
    - ("mean", "mean"): nanmean across all cameras and keypoints
    """
    ts_col = "timestamp_s" if use_timestamp_s else "timestamp"
    n_frames = len(timestamps)

    rows: list[dict] = []
    for frame in range(n_frames):
        ts = float(timestamps[frame])

        # Per-camera, per-keypoint
        for cam_idx, camera_id in enumerate(camera_ids):
            for kp_idx, kp_name in enumerate(skull_keypoint_names):
                rows.append({
                    "frame": frame,
                    ts_col: ts,
                    "trajectory": camera_id,
                    "component": kp_name,
                    "value": float(errors[frame, cam_idx, kp_idx]),
                    "units": "px",
                })

        # Per-keypoint mean across cameras
        for kp_idx, kp_name in enumerate(skull_keypoint_names):
            cam_errors = errors[frame, :, kp_idx]
            valid = cam_errors[~np.isnan(cam_errors)]
            mean_val = float(np.mean(valid)) if len(valid) > 0 else float("nan")
            rows.append({
                "frame": frame,
                ts_col: ts,
                "trajectory": "mean",
                "component": kp_name,
                "value": mean_val,
                "units": "px",
            })

        # Overall mean across all cameras and all keypoints
        all_errors = errors[frame]
        valid_all = all_errors[~np.isnan(all_errors)]
        overall_mean = float(np.mean(valid_all)) if len(valid_all) > 0 else float("nan")
        rows.append({
            "frame": frame,
            ts_col: ts,
            "trajectory": "mean",
            "component": "mean",
            "value": overall_mean,
            "units": "px",
        })

    return pl.DataFrame(rows)


def calculate_and_save_reprojection_error(recording_folder: RecordingFolder) -> Path:
    """
    Compute reprojection errors and save to solver_output in the original time base.

    Requires skull solver to have been run and calibration TOML to be present.
    """
    if recording_folder.mocap_solver_output is None:
        raise ValueError("Solver output directory does not exist. Run skull solver first.")
    if recording_folder.head_body_dlc_output is None:
        raise ValueError("DLC output directory does not exist.")
    if recording_folder.calibration_toml_path is None:
        raise ValueError("Calibration TOML not found.")

    output_path = recording_folder.mocap_solver_output / "reprojection_errors.csv"

    logger.info("=" * 60)
    logger.info("COMPUTING SKULL SOLVER REPROJECTION ERRORS")
    logger.info("=" * 60)

    errors, skull_keypoint_names, camera_ids, timestamps = compute_reprojection_errors(
        solver_output_dir=recording_folder.mocap_solver_output,
        dlc_output_dir=recording_folder.head_body_dlc_output,
        calibration_toml_path=recording_folder.calibration_toml_path,
    )

    df = build_reprojection_error_df(
        errors=errors,
        skull_keypoint_names=skull_keypoint_names,
        camera_ids=camera_ids,
        timestamps=timestamps,
        use_timestamp_s=False,
    )
    df.write_csv(output_path)

    valid_errors = errors[~np.isnan(errors)]
    if len(valid_errors) > 0:
        logger.info(f"Reprojection error: mean={np.mean(valid_errors):.2f} px, median={np.median(valid_errors):.2f} px, max={np.max(valid_errors):.2f} px")
    logger.info(f"Saved reprojection errors to: {output_path}")

    return output_path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m python_code.rigid_body_solver.reprojection_error <recording_folder>")
        sys.exit(1)

    from python_code.utilities.folder_utilities.recording_folder import PipelineStep
    recording_folder = RecordingFolder.from_folder_path(
        sys.argv[1],
        expected_processing_step=PipelineStep.SKULL_POST_PROCESSED,
    )
    calculate_and_save_reprojection_error(recording_folder)
