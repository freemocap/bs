"""Shared helpers for the filter-comparison Rerun viewers (head and eye).

Both viewers read the variant folders written by `python_code.filter_comparison`
(`<clip>/filter_comparison/<variant>/`) and put each filter condition under its own
entity, so any condition can be shown/hidden with the eye icon in Rerun's blueprint
panel and any two can be overlaid. Variants that produce identical data for a stream
(e.g. `current` and `no_eye_filter` have the same head) are merged into one condition.

Time is seconds on the gaze pipeline's common clock (0 = first common timestamp), the
same axis as the HTML filter comparison report.
"""

import argparse
import logging
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray

from python_code.filter_comparison.compare_filter_variants import (
    VARIANT_COLORS,
    _default_window_start,
    load_variant_signals,
)
from python_code.filter_comparison.build_variant import variant_is_complete
from python_code.filter_comparison.variants import FILTER_VARIANTS, VARIANTS_BY_NAME, FilterVariant
from python_code.viz.rerun_viewer.rerun_utils.process_videos import process_video_frame

logger = logging.getLogger(__name__)

RAW_COLOR: tuple[int, int, int] = (235, 235, 235)  # unfiltered tracker output reference
DEFAULT_SHOW: tuple[str, ...] = ("current", "unfiltered")


def variant_color(name: str) -> tuple[int, int, int]:
    """Same colour per variant as the HTML report (colour follows the variant name)."""
    h = VARIANT_COLORS[name].lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@dataclass(frozen=True)
class Condition:
    """One distinct filter condition for a stream: the first variant with it names it."""

    name: str  # entity name (first variant of the group)
    variants: tuple[str, ...]  # all variants giving identical data for this stream
    label: str  # filter description

    @property
    def color(self) -> tuple[int, int, int]:
        return variant_color(self.name)

    @property
    def description(self) -> str:
        same = f" (same as {', '.join(self.variants[1:])})" if len(self.variants) > 1 else ""
        return f"{self.name}: {self.label}{same}"


def group_conditions(
    variant_names: list[str], key: Callable[[FilterVariant], Hashable], label: Callable[[FilterVariant], str]
) -> list[Condition]:
    groups: dict[Hashable, list[str]] = {}
    for name in variant_names:
        groups.setdefault(key(VARIANTS_BY_NAME[name]), []).append(name)
    return [Condition(name=g[0], variants=tuple(g), label=label(VARIANTS_BY_NAME[g[0]])) for g in groups.values()]


def shown_conditions(conditions: list[Condition], show: list[str]) -> set[str]:
    """Conditions initially visible: those containing any variant named in --show."""
    if "all" in show:
        return {c.name for c in conditions}
    return {c.name for c in conditions if any(v in show for v in c.variants)}


def hidden_overrides(view_origin: str, conditions: list[Condition], shown: set[str], sub: str = "") -> dict:
    """Blueprint overrides hiding the non-shown conditions under `view_origin` (toggle with the eye icon)."""
    return {
        f"{view_origin}/{c.name}{sub}": rrb.EntityBehavior(visible=False) for c in conditions if c.name not in shown
    }


# ----------------------------------------------------------------------------- CLI / windowing
def build_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording_path", type=Path, help="Clip folder that contains filter_comparison/, e.g. <session>/full_recording")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS_BY_NAME), default=None,
                   help="Variants to load (default: every built variant)")
    p.add_argument("--show", nargs="+", choices=[*VARIANTS_BY_NAME, "all"], default=list(DEFAULT_SHOW),
                   help="Variants visible when the viewer opens, or 'all'; the rest are loaded but hidden "
                        "(default: current unfiltered)")
    p.add_argument("--start", type=float, default=None,
                   help="Window start, s on the common clock (default: around the 20 s of most head movement)")
    p.add_argument("--duration", type=float, default=60.0, help="Window length in s; 0 = whole recording (heavy: video)")
    p.add_argument("--no-video", action="store_true", help="Skip the video underlay (much faster to load)")
    p.add_argument("--video-scale", type=float, default=None, help="Resize factor for video frames")
    p.add_argument("--save", type=Path, default=None, help="Write an .rrd file instead of spawning the viewer")
    return p


def available_variants(comparison_dir: Path, requested: list[str] | None) -> list[str]:
    names = requested or [v.name for v in FILTER_VARIANTS]
    built = [n for n in names if variant_is_complete(comparison_dir / n)]
    if missing := sorted(set(names) - set(built)):
        logger.warning(f"Skipping variants that are not built: {missing}")
    if not built:
        raise FileNotFoundError(f"No built filter variants in {comparison_dir}")
    return built


@dataclass(frozen=True)
class TimeWindow:
    t0_utc_s: float  # common-clock zero, UTC seconds
    start_s: float
    end_s: float

    def mask(self, timestamps_utc_s: NDArray[np.float64]) -> NDArray[np.bool_]:
        t = timestamps_utc_s - self.t0_utc_s
        return (t >= self.start_s) & (t <= self.end_s)

    def relative(self, timestamps_utc_s: NDArray[np.float64]) -> NDArray[np.float64]:
        return timestamps_utc_s - self.t0_utc_s


def choose_window(comparison_dir: Path, variant_names: list[str], start: float | None, duration: float) -> TimeWindow:
    ref = "current" if "current" in variant_names else variant_names[0]
    t0 = float(np.load(comparison_dir / ref / "analyzable_output" / "common_timestamps_original.npy")[0])
    if duration <= 0:
        return TimeWindow(t0_utc_s=t0, start_s=-np.inf, end_s=np.inf)
    if start is None:
        vs = load_variant_signals(comparison_dir / ref)
        busiest = _default_window_start(vs, min(20.0, duration))  # 20 s with most head movement
        start = max(0.0, busiest - (duration - min(20.0, duration)) / 2)
    logger.info(f"Window {start:.1f}-{start + duration:.1f} s (common clock)")
    return TimeWindow(t0_utc_s=t0, start_s=start, end_s=start + duration)


def init_rerun(name: str, save: Path | None) -> None:
    rr.init(f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", spawn=save is None)
    if save is not None:
        rr.save(save)


# ----------------------------------------------------------------------------- logging helpers
def time_column(t: NDArray[np.float64]) -> rr.TimeColumn:
    return rr.TimeColumn("time", duration=t)


def log_video_window(
    *, video_path: Path, frame_times_utc_s: NDArray[np.float64], window: TimeWindow, entity_path: str, scale: float,
    jpeg_quality: int = 75,
) -> tuple[int, int]:
    """JPEG-encode the frames inside the window and send them. Returns the (scaled) image size.

    Frames are read sequentially (grab() up to the first one) rather than seeked, because
    seeking in these mpeg4 files is not frame-accurate. Video frame i has timestamp i.
    """
    idx = np.flatnonzero(window.mask(frame_times_utc_s))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = int(width * scale), int(height * scale)
    if idx.size == 0:
        cap.release()
        return out_w, out_h
    logger.info(f"Encoding {idx.size} frames of {video_path.name}")
    for _ in range(int(idx[0])):
        cap.grab()
    blobs: list[bytes] = []
    for _ in idx:
        ok, frame = cap.read()
        if not ok:
            break
        blobs.append(process_video_frame(frame, scale, out_w, out_h, jpeg_quality=jpeg_quality))
    cap.release()
    t = window.relative(frame_times_utc_s[idx[: len(blobs)]])
    rr.send_columns(
        entity_path, indexes=[time_column(t)],
        columns=rr.EncodedImage.columns(blob=blobs, media_type=["image/jpeg"] * len(blobs)),
    )
    return out_w, out_h


def send_points_2d(entity_path: str, t: NDArray, xy: NDArray, *, class_id: int, radius: float, draw_order: float = 10.0) -> None:
    """(n_frames, n_points, 2) keypoints; keypoint i gets keypoint_id i of the class (labels, skeleton lines)."""
    n, k, _ = xy.shape
    rr.log(entity_path, rr.Points2D.from_fields(radii=radius, class_ids=class_id, keypoint_ids=np.arange(k), show_labels=False,
                                                 draw_order=draw_order), static=True)
    rr.send_columns(entity_path, indexes=[time_column(t)],
                    columns=rr.Points2D.columns(positions=xy.reshape(-1, 2)).partition(lengths=[k] * n))


def send_points_3d(entity_path: str, t: NDArray, xyz: NDArray, *, class_id: int, radius: float) -> None:
    n, k, _ = xyz.shape
    rr.log(entity_path, rr.Points3D.from_fields(radii=radius, class_ids=class_id, keypoint_ids=np.arange(k), show_labels=False), static=True)
    rr.send_columns(entity_path, indexes=[time_column(t)],
                    columns=rr.Points3D.columns(positions=xyz.reshape(-1, 3)).partition(lengths=[k] * n))


def send_series(entity_path: str, t: NDArray, values: NDArray, *, name: str, color: tuple[int, int, int], width: float = 1.5) -> None:
    rr.log(entity_path, rr.SeriesLines(colors=[color], names=[name], widths=[width]), static=True)
    rr.send_columns(entity_path, indexes=[time_column(t)], columns=rr.Scalars.columns(scalars=values))


def log_annotation_context(
    entity_path: str, keypoint_names: list[str], connections: list[tuple[int, int]], conditions: list[Condition],
) -> dict[str, tuple[int, int]]:
    """One class per condition (with skeleton connections) + an unconnected 'marker' class.

    Returns {condition name: (connected class id, unconnected class id)}; class 1 = raw reference.
    """
    kp = [rr.AnnotationInfo(id=i, label=n) for i, n in enumerate(keypoint_names)]
    classes = [rr.ClassDescription(info=rr.AnnotationInfo(id=1, label="raw", color=RAW_COLOR),
                                   keypoint_annotations=kp, keypoint_connections=connections)]
    ids: dict[str, tuple[int, int]] = {}
    for i, c in enumerate(conditions):
        a, b = 10 + 2 * i, 11 + 2 * i
        classes.append(rr.ClassDescription(info=rr.AnnotationInfo(id=a, label=c.name, color=c.color),
                                           keypoint_annotations=kp, keypoint_connections=connections))
        classes.append(rr.ClassDescription(info=rr.AnnotationInfo(id=b, label=f"{c.name} (markers)", color=c.color),
                                           keypoint_annotations=kp))
        ids[c.name] = (a, b)
    rr.log(entity_path, rr.AnnotationContext(classes), static=True)
    return ids


def legend_markdown(title: str, sections: dict[str, list[Condition]], shown: set[str], notes: str) -> str:
    lines = [f"# {title}", "", "Each filter condition is its own entity: toggle it with the eye icon in the blueprint "
             "panel (left) in every view, or click its name in a plot legend. Overlay any two to compare.", ""]
    for heading, conds in sections.items():
        lines += [f"**{heading}**", ""]
        lines += [f"- {'👁 ' if c.name in shown else ''}`{c.name}`: {c.description.split(': ', 1)[1]}" for c in conds]
        lines.append("")
    lines += [notes]
    return "\n".join(lines)
