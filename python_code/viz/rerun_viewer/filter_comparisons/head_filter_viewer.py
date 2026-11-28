"""Rerun viewer: head data under each filter condition (2D reprojections, 3D, traces).

    uv run python -m python_code.viz.rerun_viewer.filter_comparisons.head_filter_viewer \
        <session>/full_recording [--show current sg_55ms_p3] [--start 120 --duration 60] [--cameras all]

Per head filter condition (variants with identical head data are merged, e.g. `current`
and `no_eye_filter` share the head filter and solver settings):
  * markers: the filtered 3D marker positions fed to the skull solver (small dots)
  * solved:  the rigid skull + spine keypoints the solver returns (dots + skeleton)
2D views: both reprojected into the mocap camera(s) over the video, with the raw DLC
detections (white, confidence >= 0.4) as the reference. 2D DLC tracking is the same for
every variant: the head filter acts on the triangulated 3D markers.
Traces: skull orientation (yaw/pitch/roll), world angular velocity and the nose marker.
"""

import logging
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import polars as pl
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from python_code.filter_comparison.build_variant import REFERENCE_MOCAP_VIDEO
from python_code.filter_comparison.triangulate_unfiltered import (
    DLC_CONFIDENCE_THRESHOLD,
    HEAD_DLC_MODEL_FOLDER,
    HEAD_KEYPOINT_NAMES,
    find_calibration_toml,
)
from python_code.filter_comparison.variants import HEAD_FS_HZ, FilterVariant
from python_code.viz.rerun_viewer.filter_comparisons.common import (
    Condition,
    TimeWindow,
    available_variants,
    build_arg_parser,
    choose_window,
    group_conditions,
    init_rerun,
    legend_markdown,
    log_annotation_context,
    log_video_window,
    send_points_2d,
    send_points_3d,
    send_series,
    shown_conditions,
)

logger = logging.getLogger(__name__)

CAMERA_LABELS: dict[str, str] = {
    "24676894": "topdown", "24908831": "side_0", "24908832": "side_1", "25000609": "side_2", "25006505": "side_3",
}
HEAD_CONNECTIONS: list[tuple[int, int]] = [
    (0, 4), (0, 5), (4, 6), (5, 7), (4, 5), (3, 1), (3, 2), (3, 8), (8, 9), (9, 10),
]
TRACE_GROUPS: dict[str, list[tuple[str, str]]] = {  # tab -> [(signal key, title)]
    "Orientation (deg)": [("yaw", "Skull yaw"), ("pitch", "Skull pitch"), ("roll", "Skull roll")],
    "Angular velocity (deg/s)": [("yaw_vel", "Yaw velocity (world)"), ("pitch_vel", "Pitch velocity (world)"),
                                 ("roll_vel", "Roll velocity (world)")],
    "Nose marker, solver input (mm)": [("nose_x", "Nose x"), ("nose_y", "Nose y"), ("nose_z", "Nose z")],
}


def head_condition_key(v: FilterVariant) -> tuple:
    return v.head_filter, v.solver_smoothing


def head_condition_label(v: FilterVariant) -> str:
    return f"head {v.head_filter.label(HEAD_FS_HZ)}, solver smoothing {'on' if v.solver_smoothing else 'off'}"


# ----------------------------------------------------------------------------- loading
def _keypoint_array(df: pl.DataFrame, name_col: str, names: list[str], frame_col: str = "frame") -> NDArray[np.float64]:
    """Tidy (frame, name, x/y/z) -> (n_frames, n_names, 3) in `names` order."""
    out = []
    for n in names:
        sub = df.filter(pl.col(name_col) == n).sort(frame_col)
        out.append(sub.select("x", "y", "z").to_numpy())
    return np.stack(out, axis=1).astype(np.float64)


def load_markers(variant_dir: Path) -> NDArray[np.float64]:
    df = pl.read_csv(variant_dir / "mocap_data" / "output_data" / "dlc" / "head_freemocap_data_by_frame.csv")
    return _keypoint_array(df, "keypoint", HEAD_KEYPOINT_NAMES)


def load_solved(variant_dir: Path) -> NDArray[np.float64]:
    df = (
        pl.read_csv(variant_dir / "mocap_data" / "output_data" / "solver_output" / "skull_and_spine_trajectories.csv")
        .pivot(on="component", index=["frame", "trajectory"], values="value")
    )
    return _keypoint_array(df, "trajectory", HEAD_KEYPOINT_NAMES)


def load_skull(variant_dir: Path) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    df = pl.read_csv(variant_dir / "mocap_data" / "output_data" / "solver_output" / "skull_kinematics.csv")

    def series(traj: str, comp: str) -> NDArray[np.float64]:
        return df.filter((pl.col("trajectory") == traj) & (pl.col("component") == comp)).sort("frame")["value"].to_numpy()

    t = df.filter((pl.col("trajectory") == "orientation") & (pl.col("component") == "w")).sort("frame")["timestamp_s"].to_numpy()
    q_wxyz = np.stack([series("orientation", c) for c in "wxyz"], axis=1)
    yaw, pitch, roll = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]]).as_euler("ZYX", degrees=True).T
    signals = {"yaw": np.rad2deg(np.unwrap(np.deg2rad(yaw))), "pitch": pitch, "roll": roll}
    for comp in ("yaw", "pitch", "roll"):
        signals[f"{comp}_vel"] = np.rad2deg(series("angular_velocity_global", comp))
    return t, signals


def load_dlc_2d(recording_path: Path, camera_id: str) -> NDArray[np.float64]:
    """Raw DLC detections for one camera, (n_frames, n_kp, 2), NaN below the triangulation threshold."""
    folder = recording_path / "mocap_data" / "dlc_output" / HEAD_DLC_MODEL_FOLDER
    csv = next(p for p in sorted(folder.glob(f"{camera_id}*snapshot*.csv")) if "_filtered" not in p.name)
    arr = pd.read_csv(csv, header=[1, 2], index_col=0).to_numpy(dtype=np.float64).reshape(-1, len(HEAD_KEYPOINT_NAMES), 3)
    xy = arr[..., :2].copy()
    xy[arr[..., 2] < DLC_CONFIDENCE_THRESHOLD] = np.nan
    return xy


def load_cameras(recording_path: Path) -> dict[str, dict]:
    cal = tomllib.loads(find_calibration_toml(recording_path).read_text())
    return {v["name"].split("_")[0]: v for k, v in cal.items() if k.startswith("cam_")}


def project(points: NDArray[np.float64], cam: dict) -> NDArray[np.float64]:
    n, k, _ = points.shape
    flat = points.reshape(-1, 3)
    ok = np.isfinite(flat).all(axis=1)
    out = np.full((n * k, 2), np.nan)
    if ok.any():
        uv, _ = cv2.projectPoints(flat[ok], np.array(cam["rotation"], dtype=np.float64),
                                  np.array(cam["translation"], dtype=np.float64),
                                  np.array(cam["matrix"], dtype=np.float64), np.array(cam["distortions"], dtype=np.float64))
        out[ok] = uv[:, 0, :]
    return out.reshape(n, k, 2)


# ----------------------------------------------------------------------------- blueprint
def _hide(paths: list[str]) -> dict:
    return {p: rrb.EntityBehavior(visible=False) for p in paths}


def build_blueprint(cameras: list[str], sizes: dict[str, tuple[int, int]], conditions: list[Condition], shown: set[str],
                    centre: NDArray[np.float64], legend_view: rrb.TextDocumentView) -> rrb.Blueprint:
    hidden = [c.name for c in conditions if c.name not in shown]
    cam_views = []
    for cam in cameras:
        label = CAMERA_LABELS.get(cam, cam)
        origin = f"/cameras/{label}"
        w, h = sizes[cam]
        cam_views.append(rrb.Spatial2DView(
            name=f"2D {label} (reprojected)", origin=origin,
            visual_bounds=rrb.VisualBounds2D(x_range=(0, w), y_range=(0, h)),
            overrides=_hide([f"{origin}/{c}/{s}" for c in hidden for s in ("solved", "markers")]),
        ))
    view_3d = rrb.Spatial3DView(
        name="3D head", origin="/head3d",
        overrides=_hide([f"/head3d/{c}/{s}" for c in hidden for s in ("solved", "markers")]),
        eye_controls=rrb.EyeControls3D(position=tuple(centre + np.array([0.0, -600.0, 400.0])),
                                       look_target=tuple(centre), eye_up=(0.0, 0.0, 1.0)),
        line_grid=rrb.LineGrid3D(visible=True, spacing=50.0, plane=rr.components.Plane3D.XY),
    )
    tabs = []
    for tab, signals in TRACE_GROUPS.items():
        tabs.append(rrb.Vertical(*[
            rrb.TimeSeriesView(name=title, origin=f"/traces/{key}",
                               overrides=_hide([f"/traces/{key}/{c}" for c in hidden]))
            for key, title in signals
        ], name=tab))
    left = rrb.Vertical(cam_views[0] if len(cam_views) == 1 else rrb.Tabs(*cam_views), view_3d, row_shares=[1, 1])
    right = rrb.Vertical(rrb.Tabs(*tabs), legend_view, row_shares=[4, 1])
    return rrb.Blueprint(rrb.Horizontal(left, right, column_shares=[1, 1]), collapse_panels=False)


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = build_arg_parser(__doc__)
    parser.add_argument("--cameras", nargs="+", default=[REFERENCE_MOCAP_VIDEO],
                        help=f"Camera serials for 2D views, or 'all' (default: {REFERENCE_MOCAP_VIDEO} = topdown)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    recording_path: Path = args.recording_path.resolve()
    comparison_dir = recording_path / "filter_comparison"
    names = available_variants(comparison_dir, args.variants)
    conditions = group_conditions(names, head_condition_key, head_condition_label)
    shown = shown_conditions(conditions, args.show)
    window: TimeWindow = choose_window(comparison_dir, names, args.start, args.duration)
    calib = load_cameras(recording_path)
    cameras = list(calib) if args.cameras == ["all"] else args.cameras
    scale = args.video_scale or (0.5 if len(cameras) == 1 else 0.35)

    init_rerun(f"head_filter_comparison_{recording_path.parent.name}", args.save)
    class_ids = log_annotation_context("/", HEAD_KEYPOINT_NAMES, HEAD_CONNECTIONS, conditions)

    # per-condition data (frames = mocap frames; all variants share the solver's timestamps)
    t_utc, _ = load_skull(comparison_dir / conditions[0].name)
    m = window.mask(t_utc)
    t = window.relative(t_utc[m])
    centre = np.zeros(3)
    for c in conditions:
        vdir = comparison_dir / c.name
        logger.info(f"Loading {c.description}")
        markers, solved = load_markers(vdir)[m], load_solved(vdir)[m]
        _, skull = load_skull(vdir)
        connected, unconnected = class_ids[c.name]
        send_points_3d(f"/head3d/{c.name}/solved", t, solved, class_id=connected, radius=2.5)
        send_points_3d(f"/head3d/{c.name}/markers", t, markers, class_id=unconnected, radius=1.5)
        for cam in cameras:
            label = CAMERA_LABELS.get(cam, cam)
            send_points_2d(f"/cameras/{label}/{c.name}/solved", t, project(solved, calib[cam]) * scale,
                           class_id=connected, radius=3.0, draw_order=20)
            send_points_2d(f"/cameras/{label}/{c.name}/markers", t, project(markers, calib[cam]) * scale,
                           class_id=unconnected, radius=2.0, draw_order=21)
        for key in ("yaw", "pitch", "roll", "yaw_vel", "pitch_vel", "roll_vel"):
            send_series(f"/traces/{key}/{c.name}", t, skull[key][m], name=c.name, color=c.color)
        for i, axis in enumerate("xyz"):
            send_series(f"/traces/nose_{axis}/{c.name}", t, markers[:, 0, i], name=c.name, color=c.color)
        if c is conditions[0]:
            centre = np.nanmedian(solved[:, :8].reshape(-1, 3), axis=0)

    sizes: dict[str, tuple[int, int]] = {}
    for cam in cameras:
        label = CAMERA_LABELS.get(cam, cam)
        frame_times = np.load(next((recording_path / "mocap_data" / "synchronized_corrected_videos").glob(f"{cam}*_utc*.npy"))) / 1e9
        dlc = load_dlc_2d(recording_path, cam)[: len(frame_times)]
        dm = window.mask(frame_times[: len(dlc)])
        send_points_2d(f"/cameras/{label}/dlc_raw", window.relative(frame_times[: len(dlc)][dm]), dlc[dm] * scale,
                       class_id=1, radius=2.0, draw_order=15)
        w, h = calib[cam]["size"]
        sizes[cam] = (int(w * scale), int(h * scale))
        if not args.no_video:
            video = next((recording_path / "mocap_data" / "synchronized_corrected_videos").glob(f"{cam}*.mp4"))
            sizes[cam] = log_video_window(video_path=video, frame_times_utc_s=frame_times, window=window,
                                          entity_path=f"/cameras/{label}/video", scale=scale)

    notes = (f"Window {window.start_s:.1f}–{window.end_s:.1f} s on the common clock. White = raw DLC detections "
             f"(confidence ≥ {DLC_CONFIDENCE_THRESHOLD}), identical for every variant. Small dots = filtered 3D markers "
             "(solver input); dots + skeleton = solved rigid skull. Colours match the HTML report.")
    legend = rrb.TextDocumentView(origin="/legend", name="Conditions")
    rr.log("/legend", rr.TextDocument(legend_markdown("Head filter conditions", {"Head conditions": conditions}, shown, notes),
                                      media_type=rr.MediaType.MARKDOWN), static=True)
    rr.send_blueprint(build_blueprint(cameras, sizes, conditions, shown, centre, legend))
    logger.info("Head filter viewer ready")


if __name__ == "__main__":
    main()
