"""
Animated per-syllable head and eye movement
===========================================

GIF version of the top row of `eye_syllable_viz.plot_eye_movement_by_syllable`:
the same density-sampled instances, animated frame by frame over the
syllable window, with four panels:

- skull, top view (animal facing up the page, its left on the page left)
- skull, side view (egocentric x forward, z up)
- left eye gaze path (adduction vs. elevation, deg)
- right eye gaze path (mirrored, as in the PNGs)

The skull panels draw the pointwise median of the sampled instances'
egocentrically aligned head keypoints, with a faint skeleton at the window's
first frame and at syllable onset for reference. The eye panels grow every
instance's path up to the current frame, with a dot at its current position
and the pointwise median in dashed red.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.eye_syllable_gif \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil kappa_1e+03_nlags8_seed0 [--syllables 6 12]
"""

import argparse
from pathlib import Path

import keypoint_moseq as kpms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from numpy.typing import NDArray

from python_code.moseq.kpms_loader import load_eye_in_head_kpms
from python_code.moseq.run_moseq_pipeline import KPMS_Loader, load_keypoints
from python_code.moseq.training_config import (
    load_training_config,
    load_training_loader_options,
    load_training_recording_folders,
)
from python_code.moseq.utils.bodyparts import HEAD_BODYPARTS
from python_code.moseq.visualization.eye_syllable_viz import _HEAD_EDGES, _NOSE_INDEX, _valid_instances

FPS = 120  # recording frame rate, for the time readout
_HOLD_FRAMES = 6  # extra copies of the last frame, so the loop pauses at the end


def _skull_panel(ax, traj: NDArray, pre: int, title: str, xlabel: str, ylabel: str):
    """
    Static parts of a skull panel for a (T, K, 2) median trajectory already
    projected into page coordinates, plus the artists `_update` moves each
    frame.
    """
    ax.plot(*traj[:, _NOSE_INDEX].T, color="0.8", linewidth=1, zorder=0)
    for t, color in [(0, "0.85"), (pre, "0.7")]:
        for ii, jj in _HEAD_EDGES:
            ax.plot(*traj[t, (ii, jj)].T, color=color, linewidth=1.5, linestyle=":", zorder=1)

    lines = [ax.plot([], [], color="k", linewidth=2.5, zorder=3)[0] for _ in _HEAD_EDGES]
    points = ax.scatter(*traj[0].T, color="tab:blue", s=30, zorder=4, edgecolor="none")
    (nose_trail,) = ax.plot([], [], color="tab:red", linewidth=2, zorder=2)

    flat = traj.reshape(-1, 2)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    pad = 0.1 * (hi - lo).max() + 2
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def update(t: int):
        for line, (ii, jj) in zip(lines, _HEAD_EDGES):
            line.set_data(*traj[t, (ii, jj)].T)
        points.set_offsets(traj[t])
        nose_trail.set_data(*traj[: t + 1, _NOSE_INDEX].T)
        return [*lines, points, nose_trail]

    return update


def _eye_panel(ax, eye_trajs_deg: NDArray, lim_deg: float, title: str, mirror_x: bool = False):
    """
    Static parts of an eye panel for (n, T, 2) gaze paths, plus the artists
    `_update` grows each frame. Instances with any NaN frame are dropped, as
    in `eye_syllable_viz._plot_eye_panel`.
    """
    n_sampled, T, _ = eye_trajs_deg.shape
    trajs = _valid_instances(eye_trajs_deg)
    norm = plt.Normalize(0, T - 1)
    collections = []
    for _ in trajs:
        lc = LineCollection([], cmap="viridis", norm=norm, linewidth=1.2, alpha=0.6)
        ax.add_collection(lc)
        collections.append(lc)
    heads = ax.scatter(np.zeros(len(trajs)), np.zeros(len(trajs)), color="k", s=12, zorder=3)
    (median_line,) = ax.plot([], [], color="crimson", linewidth=2, linestyle="--", zorder=4)
    median_dot = ax.scatter([np.nan], [np.nan], color="crimson", s=40, zorder=5)
    median = np.median(trajs, axis=0) if len(trajs) else None
    if not len(trajs):
        ax.text(0.5, 0.5, "no instance with good eye data\nfor the whole window",
                transform=ax.transAxes, ha="center", va="center", color="0.4")

    ax.axhline(0, color="0.85", linewidth=1, zorder=0)
    ax.axvline(0, color="0.85", linewidth=1, zorder=0)
    ax.set_xlim((lim_deg, -lim_deg) if mirror_x else (-lim_deg, lim_deg))
    ax.set_ylim(-lim_deg, lim_deg)
    ax.set_aspect("equal")
    n = len(trajs)
    ax.set_title(f"{title} ({n} instances)" if n == n_sampled else f"{title} ({n}/{n_sampled} with good eye data)")
    ax.set_xlabel("medial (nose) ← adduction (deg)" if mirror_x else "adduction (deg) → medial (nose)")
    ax.set_ylabel("elevation (deg)")

    def update(t: int):
        for lc, traj in zip(collections, trajs):
            points = traj[: t + 1].reshape(-1, 1, 2)
            lc.set_segments(np.concatenate([points[:-1], points[1:]], axis=1))
            lc.set_array(np.arange(t))
        if len(trajs):
            heads.set_offsets(trajs[:, t])
            median_line.set_data(*median[: t + 1].T)
            median_dot.set_offsets(median[t : t + 1])
        return [*collections, heads, median_line, median_dot]

    mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    return update, mappable


def save_syllable_gif(
    output_path: Path,
    syllable: int,
    head_trajs: NDArray,
    left_trajs: NDArray,
    right_trajs: NDArray,
    lim_deg: float,
    pre: int,
    gif_fps: float = 8,
) -> None:
    """One GIF: (n, T, K, 3) egocentric head and (n, T, 2) eye trajectories."""
    T = head_trajs.shape[1]
    head_median = np.median(head_trajs, axis=0)  # (T, K, 3)
    # z is relative to the head's centroid height at onset, so "up" reads as
    # change in height rather than absolute height above the floor.
    head_median[..., 2] -= head_median[pre, :, 2].mean()

    # Skull views on the top row, eyes underneath, so each eye sits on the
    # same side of the page as in the top view.
    fig = plt.figure(figsize=(12, 11))
    boxes = [[0.07, 0.53, 0.38, 0.38], [0.53, 0.53, 0.38, 0.38], [0.07, 0.1, 0.38, 0.36], [0.53, 0.1, 0.38, 0.36]]
    ax_top, ax_side, ax_left, ax_right = (fig.add_axes(b) for b in boxes)
    cax = fig.add_axes([0.93, 0.1, 0.012, 0.36])

    updates = [
        # Top view with the animal facing up the page: egocentric x (forward)
        # is page-up and +y (animal's left) is page-left, matching the eyes.
        _skull_panel(
            ax_top, np.stack([-head_median[..., 1], head_median[..., 0]], axis=-1), pre,
            "skull, top view (facing up)", "\u2190 animal's left | right \u2192 (mm)", "forward (mm)",
        ),
        _skull_panel(
            ax_side, head_median[..., [0, 2]], pre,
            "skull, side view (facing right)", "forward (mm)", "up (mm, rel. onset)",
        ),
    ]
    left_update, mappable = _eye_panel(ax_left, left_trajs, lim_deg, "left eye")
    right_update, _ = _eye_panel(ax_right, right_trajs, lim_deg, "right eye", mirror_x=True)
    updates += [left_update, right_update]
    fig.colorbar(mappable, cax=cax).set_label("frame within window")

    title = fig.suptitle("", fontsize=14)
    fig.text(
        0.5, 0.01,
        "Skull: median of sampled instances (dotted grey = window start / onset, red = nose path).\n"
        "Eyes: every instance (black dot = current gaze), dashed red = pointwise median.",
        ha="center", fontsize=9, color="0.3",
    )

    def draw(t: int):
        ms = (t - pre) / FPS * 1000
        phase = "before onset" if t < pre else "syllable"
        title.set_text(f"syllable {syllable} ({len(head_trajs)} instances)   t = {ms:+.0f} ms ({phase})")
        title.set_color("0.45" if t < pre else "k")
        return [artist for update in updates for artist in update(t)] + [title]

    frames = list(range(T)) + [T - 1] * _HOLD_FRAMES
    FuncAnimation(fig, draw, frames=frames, blit=False).save(output_path, writer=PillowWriter(fps=gif_fps), dpi=80)
    plt.close(fig)


def make_eye_syllable_gifs(
    project_dir: str | Path,
    model_name: str,
    syllables: list[int] | None = None,
    pre: int = 5,
    post: int = 15,
    min_duration: float = 3,
    min_frequency: float = 0.005,
    n_neighbors: int = 24,
    gif_fps: float = 8,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """
    Load the model's training data and save one GIF per syllable (or only
    `syllables`), sampling instances exactly as
    `eye_syllable_viz.plot_eye_movement_by_syllable` does.
    """
    project_dir = Path(project_dir)
    output_dir = Path(output_dir or project_dir / model_name / "eye_syllable_plots" / "gifs")
    output_dir.mkdir(parents=True, exist_ok=True)

    recording_folders = load_training_recording_folders(project_dir)
    coordinates, _, bodyparts = load_keypoints(
        KPMS_Loader(load_training_config(project_dir)["loader"]),
        recording_folders,
        **load_training_loader_options(project_dir),
    )
    if not all(b in bodyparts for b in HEAD_BODYPARTS):
        raise ValueError(f"model keypoints {bodyparts} don't include all of {HEAD_BODYPARTS}")
    eye_left, eye_right = load_eye_in_head_kpms(recording_folders, quality_level="medium")
    results = kpms.load_results(str(project_dir), model_name)

    head_coordinates = kpms.reindex_by_bodyparts(coordinates, bodyparts, HEAD_BODYPARTS)
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}
    instances = kpms.get_syllable_instances(
        {k: v["syllable"] for k, v in results.items()},
        pre=pre, post=post, min_duration=min_duration, min_frequency=min_frequency, min_instances=n_neighbors,
    )
    sampled = kpms.sample_instances(
        instances, n_neighbors, mode="density", coordinates=head_coordinates,
        centroids=centroids, headings=headings, pre=pre, post=post, n_neighbors=n_neighbors,
    )

    all_angles = np.concatenate([a for eye in (eye_left, eye_right) for a in eye.values()])
    lim_deg = float(np.nanpercentile(np.abs(all_angles), 99)) * 1.1

    paths = []
    for syllable, picks in sorted(sampled.items()):
        if syllables is not None and syllable not in syllables:
            continue
        head = kpms.get_instance_trajectories(
            picks, head_coordinates, pre=pre, post=post, centroids=centroids, headings=headings
        )
        left = kpms.get_instance_trajectories(picks, eye_left, pre=pre, post=post)
        right = kpms.get_instance_trajectories(picks, eye_right, pre=pre, post=post)
        path = output_dir / f"syllable_{syllable:02d}.gif"
        save_syllable_gif(path, syllable, head, left, right, lim_deg, pre, gif_fps)
        print(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("--syllables", type=int, nargs="+", help="default: every plotted syllable")
    parser.add_argument("--fps", type=float, default=8, help="GIF playback rate (frames/s)")
    args = parser.parse_args()
    make_eye_syllable_gifs(args.project_dir, args.model_name, args.syllables, gif_fps=args.fps)
