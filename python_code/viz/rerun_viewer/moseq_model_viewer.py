"""
Rerun viewer for a fitted Keypoint-MoSeq model
==============================================

Given just a fitted model's directory (`<kpms project>/<model_name>`) and the
index of one of its training sessions, reload exactly the keypoints that
model was trained on -- using the loader recorded in the project's
`training_recordings.yaml` (see `python_code/moseq/training_config.py`) --
and show them in Rerun alongside the syllable active at every frame. The
left/right eye and top-down videos from the recording's `display_videos/`
folder are added when present (the viewer works without them).

What's logged
-------------
- `/keypoints/<session>`: the model's `use_bodyparts` + config.yml skeleton,
  colored by the active syllable (grey for syllables below `min_frequency`;
  near-transparent for zero-confidence points, e.g. eye-quality-flagged
  blinks on BOTH_EYES_SKULL_FRAME).
- `/syllable/<session>/label`: text of the active syllable, its bout
  position and its usage.
- `/syllable/<session>/id`: syllable id as a time series (an ethogram).
- `/syllable/<session>/onsets`: one log line per bout onset.
- `/latents/<session>/pc<i>`: the model's latent states (optional).
- `/video/<name>`: display videos, on the same timeline.

Everything is on two timelines: `frame` (index into the session) and `time`
(seconds, from `common_timestamps.npy` when present, else frame / fps).

Runs in the repo's uv environment (rerun) -- not the kpms conda env --
which is why this only imports `kpms_loader` (no keypoint_moseq dependency)
and never `run_moseq_pipeline`.

Usage
-----
    python -m python_code.viz.rerun_viewer.moseq_model_viewer \\
        /home/scholab/moseq/both_eyes_skull_frame_405_407_test/kappa_sweep_1e+04 --list

    python -m python_code.viz.rerun_viewer.moseq_model_viewer \\
        /home/scholab/moseq/both_eyes_skull_frame_405_407_test/kappa_sweep_1e+04 \\
        -s 2 --start 0 --end 6000

Logging a full session's videos is memory-heavy (~2 GB of JPEGs for three
videos at 0.5 scale on a 45k-frame session) -- use `--start/--end`,
`--resize` or `--no-video` to keep it smaller.
"""

import argparse
import colorsys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
import yaml
from numpy.typing import NDArray

from python_code.moseq import kpms_loader
from python_code.moseq.training_config import load_training_config
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.viz.rerun_viewer.rerun_utils.process_videos import log_video_file

# Loader value strings (`KPMS_Loader.<X>.value` in run_moseq_pipeline.py) ->
# kpms_loader function. Only the custom loaders are supported: the built-in
# kpms formats (DLC, SLEAP, ...) need keypoint_moseq itself to load.
_LOADERS = {
    "3d_data": kpms_loader.load_3d_data_kpms,
    "solver_output": kpms_loader.load_solver_output_kpms,
    "eye_3d": kpms_loader.load_3d_eye_kpms,
    "skull_and_gaze": kpms_loader.load_skull_and_gaze_kpms,
    "both_eyes": kpms_loader.load_both_eyes_kpms,
    "head_with_pupil_points": kpms_loader.load_head_with_pupil_points_kpms,
    "both_eyes_skull_frame": kpms_loader.load_both_eyes_skull_frame_kpms,
}

# kpms's own default when config.yml doesn't set one (matches
# run_moseq_pipeline._KPMS_MIN_FREQUENCY).
_KPMS_MIN_FREQUENCY = 0.005

_RARE_SYLLABLE_COLOR = (150, 150, 150, 255)
_LOW_CONFIDENCE_ALPHA = 40

_VIDEO_NAMES = {
    "left_eye": "left_eye_display_video",
    "right_eye": "right_eye_display_video",
    "top_down_mocap": "topdown_mocap_display_video",
}


@dataclass
class ModelInfo:
    model_dir: Path
    project_dir: Path
    config: dict
    loader: str
    recording_paths: list[str]

    @property
    def results_dir(self) -> Path:
        return self.model_dir / "results"


def _load_model_info(model_dir: str | Path) -> ModelInfo:
    model_dir = Path(model_dir)
    project_dir = model_dir.parent
    if not (model_dir / "results").is_dir():
        raise FileNotFoundError(
            f"No results/ folder in {model_dir} -- pass a fitted model directory "
            "(`<project>/<model_name>`) whose results have been extracted."
        )
    with open(project_dir / "config.yml") as f:
        config = yaml.safe_load(f)
    training_config = load_training_config(project_dir)
    return ModelInfo(
        model_dir=model_dir,
        project_dir=project_dir,
        config=config,
        loader=training_config["loader"],
        recording_paths=training_config["recording_paths"],
    )


def _session_listing(info: ModelInfo) -> str:
    return "\n".join(f"  [{i}] {path}" for i, path in enumerate(info.recording_paths))


def list_training_sessions(model_dir: str | Path) -> None:
    """Print each training session's index (for `session_index`) and path."""
    info = _load_model_info(model_dir)
    print(f"Model: {info.model_dir}\nLoader: {info.loader}\nTraining sessions:")
    print(_session_listing(info))


def _load_session_keypoints(
    info: ModelInfo, recording_folder: RecordingFolder
) -> tuple[dict[str, NDArray], dict[str, NDArray], list[str]]:
    """Load one recording with the model's loader, subset/reordered to the
    model's `use_bodyparts` (as kpms does before fitting)."""
    if info.loader not in _LOADERS:
        raise NotImplementedError(
            f"Loader '{info.loader}' isn't supported by this viewer (needs keypoint_moseq "
            f"to load). Supported: {sorted(_LOADERS)}."
        )
    coordinates, confidences, bodyparts = _LOADERS[info.loader](recording_folder)
    use_bodyparts = info.config["use_bodyparts"]
    missing = [bp for bp in use_bodyparts if bp not in bodyparts]
    if missing:
        raise ValueError(f"Loader '{info.loader}' did not return bodyparts {missing}.")
    idx = [bodyparts.index(bp) for bp in use_bodyparts]
    coordinates = {key: coords[:, idx] for key, coords in coordinates.items()}
    confidences = {key: conf[:, idx] for key, conf in confidences.items()}
    return coordinates, confidences, use_bodyparts


def _syllable_bout_frequencies(results_dir: Path) -> dict[int, float]:
    """Fraction of all bouts (across every session the model was fit on) that
    are each syllable -- the usage measure kpms's `min_frequency` filters on."""
    counts: dict[int, int] = {}
    for csv_path in results_dir.glob("*.csv"):
        syllables = pd.read_csv(csv_path, usecols=["syllable"])["syllable"].to_numpy()
        onsets = syllables[np.r_[True, syllables[1:] != syllables[:-1]]]
        for syllable, n in zip(*np.unique(onsets, return_counts=True)):
            counts[int(syllable)] = counts.get(int(syllable), 0) + int(n)
    total = sum(counts.values())
    return {syllable: n / total for syllable, n in counts.items()}


def _syllable_color(syllable: int) -> tuple[int, int, int, int]:
    """Stable, well-separated color per syllable id (golden-ratio hue steps)."""
    hue = (syllable * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255), 255


def _bouts(syllables: NDArray) -> tuple[NDArray, NDArray]:
    """(onset frame, length) of each run of identical syllables."""
    onsets = np.flatnonzero(np.r_[True, syllables[1:] != syllables[:-1]])
    lengths = np.diff(np.r_[onsets, len(syllables)])
    return onsets, lengths


def _point_radius(coords: NDArray, edges: list[tuple[int, int]]) -> float:
    """A point size that suits the data's units (mm, or ×100 skull-frame
    units): a fraction of the median skeleton edge length."""
    if edges:
        a, b = np.array(edges).T
        lengths = np.linalg.norm(coords[:, a] - coords[:, b], axis=-1)
        scale = np.nanmedian(lengths)
    else:
        scale = np.nanmedian(np.nanstd(coords, axis=1))
    return float(0.08 * scale) if np.isfinite(scale) and scale > 0 else 1.0


def _session_timestamps(recording_folder: RecordingFolder, n_frames: int, fps: float) -> NDArray:
    path = recording_folder.common_timestamps
    if path is not None:
        timestamps = np.load(path)
        if len(timestamps) == n_frames:
            return timestamps - timestamps[0]
        print(
            f"Note: common_timestamps.npy has {len(timestamps)} frames but the model has "
            f"{n_frames}; using frame / fps instead."
        )
    return np.arange(n_frames) / fps


def _log_keypoints(
    entity_path: str,
    coords: NDArray,
    conf: NDArray,
    syllables: NDArray,
    rare: set[int],
    edges: list[tuple[int, int]],
    bodyparts: list[str],
    indexes: list,
) -> None:
    n_frames, n_keypoints, _ = coords.shape
    frame_colors = np.array(
        [_RARE_SYLLABLE_COLOR if s in rare else _syllable_color(s) for s in syllables], dtype=np.uint8
    )
    colors = np.repeat(frame_colors[:, None, :], n_keypoints, axis=1)
    colors[conf <= 0, 3] = _LOW_CONFIDENCE_ALPHA

    rr.log(entity_path, rr.Points3D.from_fields(
        radii=_point_radius(coords, edges), labels=bodyparts, show_labels=False
    ), static=True)
    rr.send_columns(
        entity_path,
        indexes=indexes,
        columns=rr.Points3D.columns(
            positions=coords.reshape(-1, 3),
            colors=colors.reshape(-1, 4),
        ).partition([n_keypoints] * n_frames),
    )

    if edges:
        a, b = np.array(edges).T
        strips = np.stack([coords[:, a], coords[:, b]], axis=2)  # (T, E, 2, 3)
        rr.log(f"{entity_path}/skeleton", rr.LineStrips3D.from_fields(
            radii=0.4 * _point_radius(coords, edges)
        ), static=True)
        rr.send_columns(
            f"{entity_path}/skeleton",
            indexes=indexes,
            columns=rr.LineStrips3D.columns(
                strips=list(strips.reshape(-1, 2, 3)),
                colors=frame_colors.repeat(len(edges), axis=0),
            ).partition([len(edges)] * n_frames),
        )


def _log_syllables(
    entity_path: str,
    session_syllables: NDArray,
    frames: NDArray,
    session_timestamps: NDArray,
    fps: float,
    frequencies: dict[int, float],
    rare: set[int],
    indexes: list,
) -> None:
    """Log syllable id, per-frame label and bout onsets for the frames in
    `frames`. Bouts are computed on the whole session, so a bout cut by the
    frame window still reports its true length."""
    syllables = session_syllables[frames]
    rr.send_columns(
        f"{entity_path}/id",
        indexes=indexes,
        columns=rr.Scalars.columns(scalars=syllables.astype(np.float64)),
    )

    onsets, lengths = _bouts(session_syllables)
    bout_index = np.repeat(np.arange(len(onsets)), lengths)[frames]
    position_in_bout = frames - onsets[bout_index]
    labels = [
        f"## Syllable {s}{' (rare)' if s in rare else ''}\n\n"
        f"frame {pos + 1} / {lengths[b]} of bout  \n"
        f"usage: {100 * frequencies.get(int(s), 0.0):.1f}% of bouts  \n"
        f"session frame {f}"
        for s, pos, b, f in zip(syllables, position_in_bout, bout_index, frames)
    ]
    rr.send_columns(
        f"{entity_path}/label",
        indexes=indexes,
        columns=rr.TextDocument.columns(
            text=labels, media_type=[rr.MediaType.MARKDOWN] * len(labels)
        ),
    )

    in_window = (onsets >= frames[0]) & (onsets <= frames[-1])
    window_onsets, window_lengths = onsets[in_window], lengths[in_window]
    texts = [
        f"syllable {session_syllables[o]}{' (rare)' if session_syllables[o] in rare else ''}: "
        f"{n} frames ({1000 * n / fps:.0f} ms)"
        for o, n in zip(window_onsets, window_lengths)
    ]
    rr.send_columns(
        f"{entity_path}/onsets",
        indexes=[
            rr.TimeColumn("frame", sequence=window_onsets),
            rr.TimeColumn("time", duration=session_timestamps[window_onsets]),
        ],
        columns=rr.TextLog.columns(text=texts),
    )


def view_moseq_model(
    model_dir: str | Path,
    session_index: int = 0,
    start_frame: int | None = None,
    end_frame: int | None = None,
    include_videos: bool = True,
    video_resize_factor: float = 0.5,
    show_latents: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Open a Rerun viewer for one training session of a fitted kpms model.

    Parameters
    ----------
    model_dir:
        The fitted model's directory, `<kpms project>/<model_name>` (must
        contain `results/`; the project dir must contain `config.yml` and
        `training_recordings.yaml`).
    session_index:
        Index into the project's training recordings (`--list` /
        `list_training_sessions` prints them).
    start_frame, end_frame:
        Optional `[start, end)` frame window to log (applies to keypoints,
        syllables and videos).
    include_videos:
        Log `display_videos/` (left eye, right eye, top-down) when present.
    video_resize_factor:
        Scale applied to video frames before JPEG encoding.
    show_latents:
        Also plot the model's latent states.
    save_path:
        Write an .rrd file (open later with `rerun <file>.rrd`) instead of
        spawning a viewer.
    """
    info = _load_model_info(model_dir)
    if not 0 <= session_index < len(info.recording_paths):
        raise IndexError(
            f"session_index {session_index} out of range; training sessions are:\n"
            f"{_session_listing(info)}"
        )
    recording_folder = RecordingFolder.from_folder_path(info.recording_paths[session_index])
    print(f"Loading '{recording_folder.recording_name}' with loader '{info.loader}'...")

    coordinates, confidences, bodyparts = _load_session_keypoints(info, recording_folder)
    fps = float(info.config.get("fps", 30))
    skeleton = info.config.get("skeleton") or []
    edges = [(bodyparts.index(a), bodyparts.index(b)) for a, b in skeleton
             if a in bodyparts and b in bodyparts]

    frequencies = _syllable_bout_frequencies(info.results_dir)
    min_frequency = info.config.get("min_frequency", _KPMS_MIN_FREQUENCY)
    rare = {s for s, freq in frequencies.items() if freq < min_frequency}

    session_results = {}
    for key in coordinates:
        results_path = info.results_dir / f"{key}.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"No results for session '{key}' at {results_path}.")
        results = pd.read_csv(results_path)
        if len(results) != len(coordinates[key]):
            raise ValueError(
                f"{results_path.name} has {len(results)} frames but the loader returned "
                f"{len(coordinates[key])} -- the recording's data has changed since fitting."
            )
        session_results[key] = results

    n_frames = len(next(iter(coordinates.values())))
    window = slice(start_frame or 0, min(end_frame or n_frames, n_frames))
    frames = np.arange(n_frames)[window]
    if len(frames) == 0:
        raise ValueError(f"Empty frame window {window} for a {n_frames}-frame session.")
    session_timestamps = _session_timestamps(recording_folder, n_frames, fps)
    timestamps = session_timestamps[window]
    indexes = [rr.TimeColumn("frame", sequence=frames), rr.TimeColumn("time", duration=timestamps)]

    rr.init(
        f"moseq_{info.project_dir.name}_{info.model_dir.name}_{recording_folder.recording_name}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        spawn=save_path is None,
    )
    if save_path is not None:
        rr.save(save_path)
    rr.log("/keypoints", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for key, coords in coordinates.items():
        results = session_results[key]
        session_syllables = results["syllable"].to_numpy()
        _log_keypoints(
            f"/keypoints/{key}", coords[window], confidences[key][window],
            session_syllables[window], rare, edges, bodyparts, indexes,
        )
        _log_syllables(
            f"/syllable/{key}", session_syllables, frames, session_timestamps, fps,
            frequencies, rare, indexes,
        )
        if show_latents:
            for column in [c for c in results.columns if c.startswith("latent_state")]:
                rr.send_columns(
                    f"/latents/{key}/pc{column.split()[-1]}",
                    indexes=indexes,
                    columns=rr.Scalars.columns(scalars=results[column].to_numpy()[window]),
                )

    video_views = []
    if include_videos:
        for name, attribute in _VIDEO_NAMES.items():
            video_path = getattr(recording_folder, attribute)
            if video_path is None:
                print(f"Note: no {name} display video for this recording; skipping.")
                continue
            print(f"Logging {video_path.name}...")
            log_video_file(
                video_path, f"/video/{name}", timestamps,
                resize_factor=video_resize_factor, start_frame=window.start, frame_indices=frames,
            )
            video_views.append(rrb.Spatial2DView(name=name, origin=f"/video/{name}"))

    keys = list(coordinates)
    label_views = [
        rrb.TextDocumentView(name=f"{key} syllable", origin=f"/syllable/{key}/label") for key in keys
    ]
    plot_tabs = [
        rrb.TimeSeriesView(name="syllable id", origin="/syllable", contents="/syllable/**/id"),
        rrb.TextLogView(name="bout onsets", origin="/syllable"),
    ]
    if show_latents:
        plot_tabs.append(rrb.TimeSeriesView(name="latent states", origin="/latents"))

    left = rrb.Vertical(
        rrb.Spatial3DView(name=f"{info.loader} keypoints", origin="/keypoints"),
        rrb.Horizontal(*label_views),
        row_shares=[4, 1],
    )
    right_parts = []
    if video_views:
        right_parts.append(rrb.Grid(*video_views))
    right_parts.append(rrb.Tabs(*plot_tabs))
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(left, rrb.Vertical(*right_parts)),
        rrb.TimePanel(timeline="frame"),
    ))
    print(f"Done.{f' Saved to {save_path}' if save_path is not None else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a fitted Keypoint-MoSeq model's syllables in Rerun.")
    parser.add_argument("model_dir", type=Path, help="<kpms project>/<model_name> directory")
    parser.add_argument("-s", "--session", type=int, default=0, help="training session index (see --list)")
    parser.add_argument("--list", action="store_true", help="list training sessions and exit")
    parser.add_argument("--start", type=int, default=None, help="first frame to log")
    parser.add_argument("--end", type=int, default=None, help="frame to stop logging at (exclusive)")
    parser.add_argument("--no-video", action="store_true", help="skip display videos")
    parser.add_argument("--resize", type=float, default=0.5, help="video resize factor")
    parser.add_argument("--no-latents", action="store_true", help="skip latent state plots")
    parser.add_argument("--save", type=Path, default=None, help="write an .rrd file instead of spawning a viewer")
    args = parser.parse_args()

    if args.list:
        list_training_sessions(args.model_dir)
    else:
        view_moseq_model(
            args.model_dir,
            session_index=args.session,
            start_frame=args.start,
            end_frame=args.end,
            include_videos=not args.no_video,
            video_resize_factor=args.resize,
            show_latents=not args.no_latents,
            save_path=args.save,
        )
