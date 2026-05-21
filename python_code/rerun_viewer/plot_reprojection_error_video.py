"""
Reprojection Error Video Viewer
================================

Plays all 5 synchronized Basler camera videos with two overlays per camera:
  • Tracked points  — original 2D DLC keypoint positions (per-keypoint colors, larger)
  • Reprojected points — 3D skull solver keypoints projected back to image plane (orange, smaller)

The displacement between tracked and reprojected for the same keypoint is the
reprojection error, making bad frames and bad cameras immediately visible.

Below the camera grid a time-series panel shows the per-frame overall mean
reprojection error so you can scrub to high-error moments.

Usage:
    python -m python_code.rerun_viewer.plot_reprojection_error_video <recording_folder>
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rigid_body_solver.reprojection_error import (
    CAMERA_IDS,
    SKULL_KEYPOINTS,
    DLC_CONFIDENCE_THRESHOLD,
    compute_projected_skull_points,
    load_dlc_2d_points,
)
from python_code.rerun_viewer.rerun_utils.process_videos import (
    process_video_frames,
    _send_encoded_frames_chunked,
)
from python_code.rerun_viewer.rerun_utils.video_data import VideoData
from python_code.utilities.folder_utilities.recording_folder import (
    BaslerCamera,
    RecordingFolder,
)

# ── Camera metadata ───────────────────────────────────────────────────────────
CAMERA_LABEL = {
    BaslerCamera.TOPDOWN: "topdown",
    BaslerCamera.SIDE_0:  "side_0",
    BaslerCamera.SIDE_1:  "side_1",
    BaslerCamera.SIDE_2:  "side_2",
    BaslerCamera.SIDE_3:  "side_3",
}

# Downscale for manageable memory / transfer; 0.5 halves each dimension
RESIZE_FACTOR = 0.5

# ── Keypoint colours ──────────────────────────────────────────────────────────
# Tracked (DLC): distinct per-keypoint colours
TRACKED_COLORS: dict[str, tuple[int, int, int]] = {
    "nose":          (0,   220, 220),
    "left_eye":      (255, 220,  50),
    "right_eye":     (100, 230, 100),
    "left_ear":      ( 80, 160, 255),
    "right_ear":     (200, 100, 255),
    "base":          (180, 180, 180),
    "left_cam_tip":  (255, 140,  60),
    "right_cam_tip": (255, 200, 120),
}
TRACKED_RADIUS   = 6.0

# Reprojected (solver): uniform orange so it's always distinct from tracked
REPROJECTED_COLOR = (255, 80, 0)
REPROJECTED_RADIUS = 4.0

OVERALL_ERROR_COLOR = (255, 255, 100)


# ── Data loading helpers ──────────────────────────────────────────────────────

def _load_video(
    recording_folder: RecordingFolder,
    camera: BaslerCamera,
) -> VideoData | None:
    """Load a Basler camera video and timestamps. Returns None if not found."""
    camera_id = camera.value
    video_path   = recording_folder.get_synchronized_video_by_name(camera_id)
    timestamps_path = recording_folder.get_timestamp_by_name(camera_id)

    if not video_path or not video_path.exists():
        print(f"  [skip] No video for camera {camera_id}")
        return None
    if not timestamps_path or not timestamps_path.exists():
        print(f"  [skip] No timestamps for camera {camera_id}")
        return None

    try:
        return VideoData.create(
            annotated_video_path=video_path,   # use raw as annotated placeholder
            raw_video_path=video_path,
            timestamps_npy_path=timestamps_path,
            data_name=f"Camera {CAMERA_LABEL[camera]}",
            resize_factor=RESIZE_FACTOR,
        )
    except Exception as e:
        print(f"  [skip] Failed to load camera {camera_id}: {e}")
        return None


def _extract_skull_dlc_points(
    points_by_camera: dict[str, np.ndarray],
    dlc_keypoint_names: list[str],
    skull_keypoint_names: list[str],
    camera_id: str,
    n_video_frames: int,
    resize_factor: float,
) -> np.ndarray:
    """
    Extract skull keypoint (x, y) coords from DLC data for one camera.

    Returns (n_frames, n_skull_kp, 2) float64 with NaN where confidence < threshold.
    Coordinates are scaled by resize_factor to match the displayed video dimensions.
    """
    dlc_data = points_by_camera.get(camera_id)
    if dlc_data is None:
        n_skull_kp = len(skull_keypoint_names)
        return np.full((n_video_frames, n_skull_kp, 2), np.nan)

    n_frames_use = min(n_video_frames, dlc_data.shape[0])
    n_skull_kp = len(skull_keypoint_names)
    result = np.full((n_video_frames, n_skull_kp, 2), np.nan)

    for kp_idx, kp_name in enumerate(skull_keypoint_names):
        if kp_name not in dlc_keypoint_names:
            continue
        dlc_idx = dlc_keypoint_names.index(kp_name)
        xy         = dlc_data[:n_frames_use, dlc_idx, :2]   # (n, 2)
        confidence = dlc_data[:n_frames_use, dlc_idx, 2]    # (n,)

        xy_scaled = xy * resize_factor
        xy_scaled[confidence < DLC_CONFIDENCE_THRESHOLD] = np.nan

        result[:n_frames_use, kp_idx, :] = xy_scaled

    return result


def _scale_projected_points(
    projected: np.ndarray,  # (n_frames, n_skull_kp, 2) in original px
    resize_factor: float,
) -> np.ndarray:
    return projected * resize_factor


# ── Rerun logging ─────────────────────────────────────────────────────────────

def _log_2d_points_bulk(
    entity_path: str,
    timestamps: np.ndarray,     # (n_frames,) zeroed seconds
    positions: np.ndarray,      # (n_frames, n_kp, 2) — NaN for missing
    colors: list[tuple[int, int, int]] | tuple[int, int, int],
    radius: float,
) -> None:
    """
    Log a time-varying set of 2D keypoints to a single entity.

    Each frame can have multiple points (one per keypoint). NaN positions
    are masked out — they won't appear in the viewer.
    """
    n_frames, n_kp, _ = positions.shape

    # Build per-keypoint colours: (n_kp, 3) or broadcast single colour
    if isinstance(colors, tuple):
        kp_colors = np.array([colors] * n_kp, dtype=np.uint8)   # (n_kp, 3)
    else:
        kp_colors = np.array(colors, dtype=np.uint8)            # (n_kp, 3)

    # Style once (static)
    rr.log(entity_path, rr.Points2D.from_fields(), static=True)

    # Send frame-by-frame (Points2D.columns doesn't support NaN masking easily,
    # so we iterate — still efficient because cv2 decoding dominates)
    for frame_idx in range(n_frames):
        frame_pts  = positions[frame_idx]           # (n_kp, 2)
        valid_mask = ~np.any(np.isnan(frame_pts), axis=1)

        valid_pts    = frame_pts[valid_mask]         # (n_valid, 2)
        valid_colors = kp_colors[valid_mask]         # (n_valid, 3)

        if len(valid_pts) == 0:
            continue

        rr.set_time("time", duration=float(timestamps[frame_idx]))
        rr.log(
            entity_path,
            rr.Points2D(
                positions=valid_pts,
                colors=valid_colors,
                radii=np.full(len(valid_pts), radius),
            ),
        )


def _log_error_timeseries(
    recording_folder: RecordingFolder,
    base: str = "/reprojection_error",
) -> None:
    """Log overall mean reprojection error if available (best-effort)."""
    resampled_csv = (
        recording_folder.analyzable_output / "reprojection_errors" / "reprojection_errors.csv"
        if recording_folder.analyzable_output
        else None
    )
    original_csv = (
        recording_folder.mocap_solver_output / "reprojection_errors.csv"
        if recording_folder.mocap_solver_output
        else None
    )
    csv_path = None
    ts_col   = "timestamp_s"
    if resampled_csv and resampled_csv.exists():
        csv_path = resampled_csv
    elif original_csv and original_csv.exists():
        csv_path = original_csv
        ts_col   = "timestamp"

    if csv_path is None:
        print("  [skip] No reprojection error CSV found — time-series panel will be empty")
        return

    import polars as pl
    df = pl.read_csv(csv_path)
    overall = (
        df.filter((pl.col("trajectory") == "mean") & (pl.col("component") == "mean"))
        .sort("frame")
    )
    timestamps = overall[ts_col].to_numpy().astype(np.float64)
    if ts_col == "timestamp":
        timestamps = timestamps - timestamps[0]
    values = overall["value"].to_numpy().astype(np.float64)

    entity = f"{base}/overall_mean"
    rr.log(entity, rr.SeriesLines(widths=2.0, colors=[OVERALL_ERROR_COLOR]), static=True)
    rr.log(entity, rr.SeriesPoints(marker_sizes=2.5, colors=[OVERALL_ERROR_COLOR]), static=True)
    rr.send_columns(
        entity_path=entity,
        indexes=[rr.TimeColumn("time", duration=timestamps)],
        columns=rr.Scalars.columns(scalars=values),
    )


# ── Blueprint ─────────────────────────────────────────────────────────────────

def _camera_view(
    label: str,
    video_data: VideoData,
) -> rrb.Spatial2DView:
    w = video_data.resized_width
    h = video_data.resized_height
    return rrb.Spatial2DView(
        name=label.replace("_", " ").title(),
        origin=f"/cameras/{label}",
        visual_bounds=VisualBounds2D.from_fields(
            range=Range2D(x_range=(0, w), y_range=(0, h))
        ),
    )


def _make_blueprint(
    camera_views: dict[str, rrb.Spatial2DView],
) -> rrb.Blueprint:
    topdown = camera_views.get("topdown")
    side_0  = camera_views.get("side_0")
    side_1  = camera_views.get("side_1")
    side_2  = camera_views.get("side_2")
    side_3  = camera_views.get("side_3")

    top_cameras = [v for v in [topdown, side_0, side_1] if v is not None]
    bot_cameras = [v for v in [side_2, side_3] if v is not None]

    error_view = rrb.TimeSeriesView(
        name="Mean Reprojection Error (px)",
        origin="/reprojection_error",
        plot_legend=rrb.PlotLegend(visible=False),
    )

    rows: list[rrb.Container | rrb.SpaceView] = []
    if top_cameras:
        rows.append(rrb.Horizontal(*top_cameras))
    if bot_cameras:
        rows.append(rrb.Horizontal(*bot_cameras, error_view))
    elif top_cameras:
        rows.append(error_view)

    return rrb.Blueprint(rrb.Vertical(*rows))


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_reprojection_error_video(recording_folder: RecordingFolder) -> None:
    print("Loading skull solver projected points...")
    if recording_folder.mocap_solver_output is None:
        raise ValueError("Skull solver output not found — run the skull solver first.")
    if recording_folder.calibration_toml_path is None:
        raise ValueError("No calibration TOML found — calibration is required for reprojection.")

    projected_by_camera, skull_keypoint_names = compute_projected_skull_points(
        solver_output_dir=recording_folder.mocap_solver_output,
        calibration_toml_path=recording_folder.calibration_toml_path,
    )

    print("Loading DLC 2D tracked points...")
    if recording_folder.head_body_dlc_output is None:
        raise ValueError("DLC output directory not found.")
    points_by_camera, dlc_keypoint_names = load_dlc_2d_points(recording_folder.head_body_dlc_output)

    # Build per-keypoint colour list for tracked points (length = n_skull_kp)
    tracked_kp_colors = [
        TRACKED_COLORS.get(name, (180, 180, 180)) for name in skull_keypoint_names
    ]

    print("Loading camera videos...")
    videos: dict[str, VideoData] = {}
    for camera in BaslerCamera:
        label = CAMERA_LABEL[camera]
        vd = _load_video(recording_folder, camera)
        if vd is not None:
            videos[label] = vd

    if not videos:
        raise RuntimeError("No camera videos found.")

    # Global time alignment: subtract the earliest first-timestamp across all cameras
    global_start = min(vd.timestamps[0] for vd in videos.values())
    print(f"Global start: {global_start:.3f}s — aligning all cameras to this origin.")

    # ── Initialise Rerun ─────────────────────────────────────────────────────
    recording_string = (
        f"reprojection_video_{recording_folder.recording_name}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    camera_views: dict[str, rrb.Spatial2DView] = {
        label: _camera_view(label, vd) for label, vd in videos.items()
    }
    rr.send_blueprint(_make_blueprint(camera_views))

    # ── Log reprojection error time series ───────────────────────────────────
    print("Logging reprojection error time series...")
    _log_error_timeseries(recording_folder)

    # ── Per-camera: encode video + log 2D overlays ───────────────────────────
    for camera in BaslerCamera:
        label     = CAMERA_LABEL[camera]
        camera_id = camera.value
        vd        = videos.get(label)
        if vd is None:
            continue

        entity_root      = f"/cameras/{label}"
        entity_video     = f"{entity_root}/raw"
        entity_tracked   = f"{entity_root}/tracked"
        entity_projected = f"{entity_root}/reprojected"

        # Globally normalised timestamps for this camera
        cam_timestamps = vd.timestamps - global_start  # (n_frames,) seconds from global_start
        n_video_frames  = len(cam_timestamps)

        print(f"[{label}] Encoding {n_video_frames} frames...")
        encoded = process_video_frames(
            video_cap=vd.raw_vid_cap,
            resize_factor=vd.resize_factor,
            resize_width=vd.resized_width,
            resize_height=vd.resized_height,
        )
        _send_encoded_frames_chunked(entity_video, cam_timestamps, encoded)
        print(f"[{label}] Video logged.")

        # ── DLC tracked points ────────────────────────────────────────────────
        tracked_xy = _extract_skull_dlc_points(
            points_by_camera=points_by_camera,
            dlc_keypoint_names=dlc_keypoint_names,
            skull_keypoint_names=skull_keypoint_names,
            camera_id=camera_id,
            n_video_frames=n_video_frames,
            resize_factor=RESIZE_FACTOR,
        )  # (n_video_frames, n_skull_kp, 2)

        print(f"[{label}] Logging tracked points...")
        _log_2d_points_bulk(
            entity_path=entity_tracked,
            timestamps=cam_timestamps,
            positions=tracked_xy,
            colors=tracked_kp_colors,
            radius=TRACKED_RADIUS,
        )

        # ── Reprojected points ────────────────────────────────────────────────
        projected = projected_by_camera.get(camera_id)
        if projected is not None:
            n_proj = min(n_video_frames, projected.shape[0])
            proj_xy = _scale_projected_points(projected[:n_proj], RESIZE_FACTOR)
            padded  = np.full((n_video_frames, proj_xy.shape[1], 2), np.nan)
            padded[:n_proj] = proj_xy

            print(f"[{label}] Logging reprojected points...")
            _log_2d_points_bulk(
                entity_path=entity_projected,
                timestamps=cam_timestamps,
                positions=padded,
                colors=REPROJECTED_COLOR,
                radius=REPROJECTED_RADIUS,
            )
        else:
            print(f"[{label}] No projected points available, skipping reprojected overlay.")

    print("Done. Rerun viewer should open.")


if __name__ == "__main__":
    from python_code.utilities.folder_utilities.recording_folder import PipelineStep

    if len(sys.argv) < 2:
        print(
            "Usage: python -m python_code.rerun_viewer.plot_reprojection_error_video <recording_folder>"
        )
        sys.exit(1)

    folder = RecordingFolder.from_folder_path(
        sys.argv[1],
        expected_processing_step=PipelineStep.SKULL_POST_PROCESSED,
    )
    plot_reprojection_error_video(folder)
