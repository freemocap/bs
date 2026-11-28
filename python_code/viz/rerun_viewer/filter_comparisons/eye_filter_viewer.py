"""Rerun viewer: eye data under each filter condition (2D on the eye video, 3D eye, traces).

    uv run python -m python_code.viz.rerun_viewer.filter_comparisons.eye_filter_viewer \
        <session>/full_recording [--show current sg_55ms_p3] [--eyes left] [--start 120 --duration 60]

Per eye filter condition (variants with identical eye data are merged; the head filter
does not touch eye-in-head data, e.g. `current` == `no_head_filter` here):
  * 2D: the filtered DLC pixel keypoints (pupil ring p1-p8, tear duct, outer eye, corneal
    reflections) over the raw eye video, with the raw DLC detections (white) as reference.
    These are the exact pixels the eye pipeline aligns and fits (same loader + filter).
  * 3D: the fitted eye-in-head pupil ring and gaze direction from eye kinematics.
  * Traces: eye-in-head adduction / elevation, horizontal / vertical velocity, and pupil
    centre pixel position (raw DLC in white).
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray

from python_code.eye_analysis.data_models.csv_io import load_trajectory_dataset
from python_code.eye_analysis.data_processing.align_data.eye_anatomical_alignment import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SAVGOL_WINDOW_LENGTH,
)
from python_code.ferret_gaze.eye_kinematics.eye_kinematics_rerun_viewer import (
    COLOR_SPHERE_WIRE,
    generate_sphere_line_strips_with_pole_at_z,
)
from python_code.eye_analysis.process_eye_session import find_eye_dlc_csv
from python_code.filter_comparison.variants import EYE_FS_HZ, VARIANTS_BY_NAME, FilterVariant
from python_code.utilities.temporal_filters import FilterSpec
from python_code.viz.rerun_viewer.filter_comparisons.common import (
    RAW_COLOR,
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
    time_column,
)

logger = logging.getLogger(__name__)

PUPIL_POINTS = [f"p{i}" for i in range(1, 9)]
PIXEL_KEYPOINTS = [*PUPIL_POINTS, "tear_duct", "outer_eye", "cr_top", "cr_bottom"]
PIXEL_CONNECTIONS = [(i, (i + 1) % 8) for i in range(8)] + [(8, 9)]
KINEMATIC_KEYPOINTS = [*PUPIL_POINTS, "pupil_center", "tear_duct", "outer_eye"]
KINEMATIC_CONNECTIONS = [(i, (i + 1) % 8) for i in range(8)]
EYE_SIGN = {"left": -1.0, "right": 1.0}  # adduction -> yaw-sense horizontal velocity (same as the report)
TRACE_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Eye in head (deg)": [("adduction", "Adduction"), ("elevation", "Elevation")],
    "Velocity (deg/s)": [("horizontal_vel", "Horizontal velocity (yaw sense)"), ("vertical_vel", "Vertical velocity")],
    "Pupil centre pixels": [("pupil_px_x", "Pupil centre x (px)"), ("pupil_px_y", "Pupil centre y (px)")],
}


def eye_condition_key(v: FilterVariant) -> FilterSpec:
    return v.eye_filter


def eye_condition_label(v: FilterVariant) -> str:
    return f"eye {v.eye_filter.label(EYE_FS_HZ)}"


# ----------------------------------------------------------------------------- loading
def load_pixels(recording_path: Path, eye: str, frame_times_s: NDArray, spec: FilterSpec) -> tuple[NDArray, NDArray]:
    """(raw, cleaned) DLC pixel keypoints, (n_frames, n_kp, 2), exactly as eye_alignment_main loads them."""
    ds = load_trajectory_dataset(
        filepath=find_eye_dlc_csv(recording_path, f"{eye}_eye"), timestamps=frame_times_s,
        min_confidence=DEFAULT_MIN_CONFIDENCE, savgol_window_length=DEFAULT_SAVGOL_WINDOW_LENGTH,
        savgol_polyorder=DEFAULT_SAVGOL_POLYORDER, filter_spec=spec,
    )
    raw = np.stack([ds.trajectories[k].raw.data for k in PIXEL_KEYPOINTS], axis=1)
    cleaned = np.stack([ds.trajectories[k].cleaned.data for k in PIXEL_KEYPOINTS], axis=1)
    return raw, cleaned


def load_kinematics(variant_dir: Path, eye: str) -> tuple[NDArray, NDArray, NDArray, dict[str, NDArray]]:
    """Eye kinematics at eye-camera frames: t (UTC s), keypoints (n, k, 3) mm, gaze target (n, 3), angles (deg)."""
    df = pl.read_csv(variant_dir / "eye_data" / "output_data" / "eye_kinematics" / f"{eye}_eye_kinematics.csv")

    def series(traj: str, comp: str) -> NDArray:
        return df.filter((pl.col("trajectory") == traj) & (pl.col("component") == comp)).sort("frame")["value"].to_numpy()

    t = df.filter((pl.col("trajectory") == "eye_in_head") & (pl.col("component") == "adduction")).sort("frame")["timestamp_s"].to_numpy()
    kp = np.stack([np.stack([series(f"keypoint__{k}", c) for c in "xyz"], axis=1) for k in KINEMATIC_KEYPOINTS], axis=1)
    gaze = np.stack([series("keypoint__gaze_target", c) for c in "xyz"], axis=1)
    angles = {"adduction": np.rad2deg(series("eye_in_head", "adduction")),
              "elevation": np.rad2deg(series("eye_in_head", "elevation"))}
    return t, kp, gaze, angles


# ----------------------------------------------------------------------------- per eye
def log_eye(
    *, eye: str, recording_path: Path, comparison_dir: Path, conditions: list[Condition], window: TimeWindow,
    no_video: bool, scale: float,
) -> tuple[int, int]:
    root = f"/{eye}"
    frame_times = np.load(recording_path / "eye_data" / "eye_videos" / f"{eye}_eye_timestamps_utc.npy") / 1e9
    m = window.mask(frame_times)
    t_px = window.relative(frame_times[m])
    ids_2d = log_annotation_context(f"{root}/video", PIXEL_KEYPOINTS, PIXEL_CONNECTIONS, conditions)
    ids_3d = log_annotation_context(f"{root}/3d", KINEMATIC_KEYPOINTS, KINEMATIC_CONNECTIONS, conditions)

    radius = 3.0
    raw = None
    for c in conditions:
        logger.info(f"{eye} eye: loading {c.description}")
        raw, cleaned = load_pixels(recording_path, eye, frame_times, VARIANTS_BY_NAME[c.name].eye_filter)
        send_points_2d(f"{root}/video/{c.name}", t_px, cleaned[m] * scale, class_id=ids_2d[c.name][0], radius=2.0, draw_order=20)
        pupil_px = np.nanmean(cleaned[:, :8], axis=1)[m]
        send_series(f"{root}/traces/pupil_px_x/{c.name}", t_px, pupil_px[:, 0], name=c.name, color=c.color)
        send_series(f"{root}/traces/pupil_px_y/{c.name}", t_px, pupil_px[:, 1], name=c.name, color=c.color)

        t_k_utc, kp, gaze, angles = load_kinematics(comparison_dir / c.name, eye)
        km = window.mask(t_k_utc)
        t_k = window.relative(t_k_utc[km])
        radius = float(np.nanmedian(np.linalg.norm(kp[:, 8], axis=1)))
        send_points_3d(f"{root}/3d/{c.name}/pupil", t_k, kp[km], class_id=ids_3d[c.name][0], radius=0.08)
        direction = gaze[km] / np.linalg.norm(gaze[km], axis=1, keepdims=True) * radius * 2.0
        rr.log(f"{root}/3d/{c.name}/gaze", rr.Arrows3D.from_fields(origins=[[0.0, 0.0, 0.0]], colors=[c.color], radii=0.05), static=True)
        rr.send_columns(f"{root}/3d/{c.name}/gaze", indexes=[time_column(t_k)], columns=rr.Arrows3D.columns(vectors=direction))
        for key in ("adduction", "elevation"):
            send_series(f"{root}/traces/{key}/{c.name}", t_k, angles[key][km], name=c.name, color=c.color)
        t_all = window.relative(t_k_utc)
        h_vel = EYE_SIGN[eye] * np.gradient(angles["adduction"], t_all)
        v_vel = np.gradient(angles["elevation"], t_all)
        send_series(f"{root}/traces/horizontal_vel/{c.name}", t_k, h_vel[km], name=c.name, color=c.color)
        send_series(f"{root}/traces/vertical_vel/{c.name}", t_k, v_vel[km], name=c.name, color=c.color)

    # raw DLC reference (identical for every condition) and static eye sphere
    send_points_2d(f"{root}/video/dlc_raw", t_px, raw[m] * scale, class_id=1, radius=1.5, draw_order=15)
    raw_pupil = np.nanmean(raw[:, :8], axis=1)[m]
    send_series(f"{root}/traces/pupil_px_x/dlc_raw", t_px, raw_pupil[:, 0], name="raw DLC", color=RAW_COLOR, width=1.0)
    send_series(f"{root}/traces/pupil_px_y/dlc_raw", t_px, raw_pupil[:, 1], name="raw DLC", color=RAW_COLOR, width=1.0)
    strips = generate_sphere_line_strips_with_pole_at_z(radius, 8, 16)
    rr.log(f"{root}/3d/sphere", rr.LineStrips3D(strips=strips, colors=[COLOR_SPHERE_WIRE] * len(strips), radii=[0.01]), static=True)

    size = (int(400 * scale), int(400 * scale))
    if not no_video:
        size = log_video_window(video_path=recording_path / "eye_data" / "eye_videos" / f"{eye}_eye.mp4",
                                frame_times_utc_s=frame_times, window=window, entity_path=f"{root}/video/image", scale=scale)
    return size


def _hide(paths: list[str]) -> dict:
    return {p: rrb.EntityBehavior(visible=False) for p in paths}


def eye_tab(eye: str, size: tuple[int, int], conditions: list[Condition], shown: set[str]) -> rrb.Container:
    root = f"/{eye}"
    hidden = [c.name for c in conditions if c.name not in shown]
    video = rrb.Spatial2DView(
        name=f"{eye} eye video + DLC keypoints", origin=f"{root}/video",
        visual_bounds=rrb.VisualBounds2D(x_range=(0, size[0]), y_range=(0, size[1])),
        overrides=_hide([f"{root}/video/{c}" for c in hidden]),
    )
    view_3d = rrb.Spatial3DView(
        name=f"{eye} eye 3D (eye in head)", origin=f"{root}/3d",
        overrides=_hide([f"{root}/3d/{c}/{s}" for c in hidden for s in ("pupil", "gaze")]),
        eye_controls=rrb.EyeControls3D(position=(0.0, 0.0, 15.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 1.0, 0.0),
                                       kind=rrb.Eye3DKind.Orbital),
        line_grid=rrb.LineGrid3D(visible=False),
    )
    tabs = [
        rrb.Vertical(*[rrb.TimeSeriesView(name=title, origin=f"{root}/traces/{key}",
                                          overrides=_hide([f"{root}/traces/{key}/{c}" for c in hidden]))
                       for key, title in signals], name=tab)
        for tab, signals in TRACE_GROUPS.items()
    ]
    return rrb.Horizontal(rrb.Vertical(video, view_3d), rrb.Tabs(*tabs), name=f"{eye} eye", column_shares=[1, 1])


def main() -> None:
    parser = build_arg_parser(__doc__)
    parser.add_argument("--eyes", nargs="+", choices=["left", "right"], default=["left", "right"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    recording_path: Path = args.recording_path.resolve()
    comparison_dir = recording_path / "filter_comparison"
    names = available_variants(comparison_dir, args.variants)
    conditions = group_conditions(names, eye_condition_key, eye_condition_label)
    shown = shown_conditions(conditions, args.show)
    window = choose_window(comparison_dir, names, args.start, args.duration)
    scale = args.video_scale or 1.0

    init_rerun(f"eye_filter_comparison_{recording_path.parent.name}", args.save)
    sizes = {eye: log_eye(eye=eye, recording_path=recording_path, comparison_dir=comparison_dir, conditions=conditions,
                          window=window, no_video=args.no_video, scale=scale) for eye in args.eyes}

    notes = (f"Window {window.start_s:.1f}–{window.end_s:.1f} s on the common clock. White = raw DLC detections, "
             "identical for every variant. 3D: pupil ring + gaze arrow in the eye-in-head frame. Colours match the HTML report.")
    rr.log("/legend", rr.TextDocument(legend_markdown("Eye filter conditions", {"Eye conditions": conditions}, shown, notes),
                                      media_type=rr.MediaType.MARKDOWN), static=True)
    tabs = [eye_tab(eye, sizes[eye], conditions, shown) for eye in args.eyes]
    rr.send_blueprint(rrb.Blueprint(
        rrb.Vertical(tabs[0] if len(tabs) == 1 else rrb.Tabs(*tabs), rrb.TextDocumentView(origin="/legend", name="Conditions"),
                     row_shares=[5, 1]),
        collapse_panels=False,
    ))
    logger.info("Eye filter viewer ready")


if __name__ == "__main__":
    main()
