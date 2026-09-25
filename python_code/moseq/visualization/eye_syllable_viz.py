"""
Per-syllable head vs. eye movement visualization
=================================================

kpms's built-in `trajectory_plots`/`generate_grid_movies`
(`python_code/moseq/run_moseq_pipeline.py`) plot every keypoint of a
HEAD_WITH_PUPIL_POINTS model on one spatial scale. Head displacement is tens
of mm; pupil-point deflection is a couple mm -- so on that shared scale, eye
movement is invisible even when it's real.

This module answers the underlying question directly instead: for each
syllable, is there a coherent eye-movement signal, and is it large or small
relative to the syllable's head movement? It does this using a much better
eye-movement signal than the raw pupil-point keypoints: `eye_in_head`
adduction/elevation (see `kpms_loader.load_eye_in_head_kpms`), anatomical
gaze angles already expressed in the skull's own body frame, so they are
unaffected by head rotation and live on their own natural (degree) scale
regardless of how far the head moves.

Entry points:

`plot_eye_movement_by_syllable`
    Per syllable: the representative (density-sampled, egocentrically
    aligned) head trajectory alongside a colored path of where each eye was
    pointing over the same window. Saves one PNG per syllable and returns
    `movement_stats` (magnitude) and `direction_stats` (per-instance net
    displacement vectors) for the two summary plots below.

`plot_path_length_summary`
    Per-syllable head and eye movement *magnitude* (mean +/- SD path length
    across sampled instances) -- the direct answer to "are eye movements
    dominated by head movements, or expressed independently?".

`plot_movement_direction_summary`
    Per-syllable head and eye movement *direction*: polar plots of each
    instance's net displacement vector plus the vector-averaged direction,
    with a resultant-length consistency measure -- path length alone (as in
    `plot_path_length_summary`) says nothing about whether a syllable moves
    the same way each time or in random directions that happen to average to
    a similar distance.

Usage
-----
    from python_code.moseq.kpms_loader import load_eye_in_head_kpms
    from python_code.moseq.run_moseq_pipeline import KPMS_Loader, load_keypoints
    from python_code.moseq.visualization.eye_syllable_viz import (
        plot_eye_movement_by_syllable,
        plot_path_length_summary,
        plot_movement_direction_summary,
    )

    coordinates, _, bodyparts = load_keypoints(KPMS_Loader.HEAD_WITH_PUPIL_POINTS, recording_folder)
    eye_in_head_left, eye_in_head_right = load_eye_in_head_kpms(recording_folder)
    results = kpms.load_results(project_dir, model_name)

    movement_stats, direction_stats = plot_eye_movement_by_syllable(
        project_dir, model_name, coordinates, eye_in_head_left, eye_in_head_right,
        results, bodyparts,
    )
    plot_path_length_summary(
        movement_stats, output_path=f"{project_dir}/{model_name}/eye_syllable_plots/path_length_summary.png"
    )
    plot_movement_direction_summary(
        direction_stats, output_path=f"{project_dir}/{model_name}/eye_syllable_plots/direction_summary.png"
    )
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import keypoint_moseq as kpms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import NDArray

from python_code.moseq.utils.bodyparts import HEAD_BODYPARTS
from python_code.moseq.utils.skeletons import HEAD_SKELETON

_HEAD_EDGES = [(HEAD_BODYPARTS.index(a), HEAD_BODYPARTS.index(b)) for a, b in HEAD_SKELETON]
_NOSE_INDEX = HEAD_BODYPARTS.index("nose")


def _per_instance_path_lengths(trajectories: NDArray) -> NDArray:
    """
    Total frame-to-frame Euclidean distance along each of a batch of
    trajectories, shape (n_instances, T, D) -> (n_instances,).

    Deliberately per-instance rather than computed on a pointwise-averaged
    trajectory: for a channel with no canonical reference direction (e.g. raw
    gaze angle, unlike egocentrically-aligned head pose), averaging
    instances *before* measuring path length can cancel out real excursions
    that point different ways from one instance to the next, understating
    the true movement magnitude.
    """
    return np.linalg.norm(np.diff(trajectories, axis=1), axis=-1).sum(axis=1)


def _instance_displacement_vectors(trajectories: NDArray) -> NDArray:
    """
    Net displacement (last frame minus first frame) for each of a batch of
    2D trajectories, shape (n_instances, T, 2) -> (n_instances, 2).

    Unlike path length, a net displacement vector is meaningful to
    vector-average across instances -- averaging cancels out instances that
    moved in different directions, which is exactly the "consistency"
    question `plot_movement_direction_summary` is trying to answer (as
    opposed to `_per_instance_path_lengths`, where that same cancellation
    would just be a bug).
    """
    return trajectories[:, -1] - trajectories[:, 0]


def _plot_head_panel(ax, head_traj_xy: NDArray, n_fade_steps: int = 6) -> None:
    """
    Draw a top-down fading-skeleton plot of a single (T, n_head_keypoints, 2)
    trajectory on `ax`, plus a continuous line tracing the nose across the
    whole window so the overall path shape is visible at a glance.
    """
    T = head_traj_xy.shape[0]
    ax.plot(*head_traj_xy[:, _NOSE_INDEX].T, color="0.7", linewidth=1, zorder=0)

    steps = np.linspace(0, T - 1, n_fade_steps).round().astype(int)
    for step_i, t in enumerate(steps):
        alpha = (step_i + 1) / n_fade_steps
        for ii, jj in _HEAD_EDGES:
            ax.plot(
                *head_traj_xy[t, (ii, jj)].T,
                color="k",
                alpha=alpha,
                linewidth=2,
                zorder=step_i * 2,
            )
        ax.scatter(
            *head_traj_xy[t].T,
            color="tab:blue",
            alpha=alpha,
            s=20,
            zorder=step_i * 2 + 1,
            edgecolor="none",
        )

    ax.set_aspect("equal")
    ax.set_title("head (egocentric, top-down)")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")


def _plot_eye_panel(ax, eye_trajs_deg: NDArray, lim_deg: float, title: str):
    """
    Draw every sampled instance's (T, 2) [adduction_deg, elevation_deg] path
    on `ax` as a thin line, colored along its length by time-within-window
    (viridis, shared scale across all lines -- use the returned mappable for
    a colorbar). A dashed line shows the pointwise median across instances
    for reference.

    Gaze direction at syllable onset is essentially arbitrary from one
    instance to the next (unlike head pose, which is rotated/centered into a
    common egocentric frame before averaging), so a single "representative"
    eye trajectory tends to cancel out real excursions that point different
    ways across instances. Plotting every instance avoids hiding that: a
    syllable with a real, consistent eye-movement component shows lines
    that agree in direction/magnitude; one without shows lines pointing every
    which way, which the dashed median alone would make look like "no
    movement."

    Returns the `LineCollection` mappable (for a shared colorbar) and proxy
    artists for the onset/end markers (for a shared legend).
    """
    n_instances, T, _ = eye_trajs_deg.shape
    norm = plt.Normalize(0, T - 1)
    mappable = None
    for traj in eye_trajs_deg:
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments, cmap="viridis", norm=norm, array=np.arange(T - 1), linewidth=1, alpha=0.5
        )
        ax.add_collection(lc)
        mappable = lc
        ax.scatter(*traj[0], facecolor="none", edgecolor="k", s=25, alpha=0.5, zorder=3)
        ax.scatter(*traj[-1], marker="*", color="k", s=35, alpha=0.5, zorder=3)

    median_traj = np.median(eye_trajs_deg, axis=0)
    (median_line,) = ax.plot(
        *median_traj.T, color="crimson", linewidth=2, linestyle="--", zorder=4
    )

    ax.axhline(0, color="0.85", linewidth=1, zorder=0)
    ax.axvline(0, color="0.85", linewidth=1, zorder=0)
    ax.set_xlim(-lim_deg, lim_deg)
    ax.set_ylim(-lim_deg, lim_deg)
    ax.set_aspect("equal")
    ax.set_title(f"{title} ({n_instances} instances)")
    ax.set_xlabel("adduction (deg)")
    ax.set_ylabel("elevation (deg)")

    return mappable, median_line


def plot_eye_movement_by_syllable(
    project_dir: str | Path,
    model_name: str,
    coordinates: dict,
    eye_in_head_left: dict,
    eye_in_head_right: dict,
    results: dict,
    bodyparts: list[str],
    pre: int = 5,
    post: int = 15,
    min_duration: float = 3,
    min_frequency: float = 0.005,
    n_neighbors: int = 24,
    output_dir: str | Path | None = None,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, NDArray]]]:
    """
    Generate one figure per syllable showing its representative head
    trajectory alongside each eye's head-relative gaze-angle path over the
    same window.

    Parameters
    ----------
    coordinates:
        From `load_keypoints(KPMS_Loader.HEAD_WITH_PUPIL_POINTS, source)` --
        must include at least the 8 `HEAD_BODYPARTS` keypoints.
    eye_in_head_left, eye_in_head_right:
        From `kpms_loader.load_eye_in_head_kpms(source)`.
    results:
        From `kpms.load_results`/`kpms.extract_results` for the same
        recordings, model-fit on `coordinates` (so `results[...]["syllable"]`
        etc. share frame indexing with `coordinates`).
    bodyparts:
        Full bodypart list matching `coordinates`'s keypoint axis (i.e.
        `HEAD_WITH_PUPIL_POINTS_BODYPARTS`).
    pre, post:
        Window (in frames) around syllable onset, as in
        `kpms.get_syllable_instances`/`generate_trajectory_plots`.
    min_duration, min_frequency, n_neighbors:
        Same meaning as in `run_moseq_pipeline.trajectory_plots`. `n_neighbors`
        is also the minimum instance count a syllable needs to be plotted.
    output_dir:
        Defaults to `{project_dir}/{model_name}/eye_syllable_plots`.

    Returns
    -------
    movement_stats: dict
        `{syllable: {"head_path_length_mm", "left_eye_path_length_deg",
        "right_eye_path_length_deg", "left_eye_path_length_deg_std",
        "right_eye_path_length_deg_std"}}`, for `plot_path_length_summary`.
        Path lengths are the mean (and, for the eyes, std) of *per-instance*
        path length -- not the path length of the pointwise-averaged
        trajectory shown in the plot, which understates true movement
        whenever instances don't share a common excursion direction (see
        `_plot_eye_panel`).
    direction_stats: dict
        `{syllable: {"head_xy", "left_eye", "right_eye"}}`, each an
        `(n_instances, 2)` array of that channel's net displacement vector
        (window end minus window onset) per sampled instance, for
        `plot_movement_direction_summary`.
    """
    for name, coords in coordinates.items():
        for side, eye_in_head in [("left", eye_in_head_left), ("right", eye_in_head_right)]:
            if eye_in_head[name].shape[0] != coords.shape[0]:
                raise ValueError(
                    f"Frame count mismatch for '{name}': coordinates has "
                    f"{coords.shape[0]} frames but {side} eye_in_head has "
                    f"{eye_in_head[name].shape[0]}. These are expected to be on "
                    "the same common-timestamp grid -- see "
                    "kpms_loader.load_eye_in_head_kpms."
                )

    if output_dir is None:
        output_dir = Path(project_dir) / model_name / "eye_syllable_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    head_coordinates = kpms.reindex_by_bodyparts(coordinates, bodyparts, HEAD_BODYPARTS)

    syllables = {k: v["syllable"] for k, v in results.items()}
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}

    syllable_instances = kpms.get_syllable_instances(
        syllables,
        pre=pre,
        post=post,
        min_duration=min_duration,
        min_frequency=min_frequency,
        min_instances=n_neighbors,
    )
    if len(syllable_instances) == 0:
        raise ValueError(
            "No syllables with sufficient instances to plot. Try lowering "
            "n_neighbors/min_duration/min_frequency."
        )

    sampled_instances = kpms.sample_instances(
        syllable_instances,
        n_neighbors,
        mode="density",
        coordinates=head_coordinates,
        centroids=centroids,
        headings=headings,
        pre=pre,
        post=post,
        n_neighbors=n_neighbors,
    )

    # Shared axis scale across all syllables, so panel size itself reflects
    # how much the eye actually moved (rather than each syllable being
    # auto-scaled to fill its own panel).
    all_angles = np.concatenate(
        [a for eye_in_head in (eye_in_head_left, eye_in_head_right) for a in eye_in_head.values()]
    )
    lim_deg = float(np.nanpercentile(np.abs(all_angles), 99)) * 1.1

    movement_stats: dict[int, dict[str, float]] = {}
    direction_stats: dict[int, dict[str, NDArray]] = {}

    for syllable, instances in sampled_instances.items():
        head_trajs = kpms.get_instance_trajectories(
            instances, head_coordinates, pre=pre, post=post, centroids=centroids, headings=headings
        )  # (n_instances, pre+post, n_head_keypoints, 3)
        left_trajs = kpms.get_instance_trajectories(
            instances, eye_in_head_left, pre=pre, post=post
        )  # (n_instances, pre+post, 2)
        right_trajs = kpms.get_instance_trajectories(instances, eye_in_head_right, pre=pre, post=post)

        # Fixed-position axes (rather than tight_layout, which doesn't know
        # how to reserve space for a colorbar spanning two of the three
        # subplots and ends up overlapping it onto the right eye panel).
        fig = plt.figure(figsize=(16, 5))
        ax_head = fig.add_axes([0.04, 0.16, 0.27, 0.72])
        ax_left = fig.add_axes([0.37, 0.16, 0.24, 0.72])
        ax_right = fig.add_axes([0.63, 0.16, 0.24, 0.72])
        cax = fig.add_axes([0.90, 0.16, 0.015, 0.72])

        fig.suptitle(f"syllable {syllable} ({len(instances)} instances)")
        _plot_head_panel(ax_head, np.median(head_trajs, axis=0)[:, :, :2])
        left_mappable, median_line = _plot_eye_panel(ax_left, left_trajs, lim_deg, "left eye")
        _plot_eye_panel(ax_right, right_trajs, lim_deg, "right eye")

        onset_marker = ax_left.scatter([], [], facecolor="none", edgecolor="k", s=25)
        end_marker = ax_left.scatter([], [], marker="*", color="k", s=35)
        fig.legend(
            [onset_marker, end_marker, median_line],
            [
                f"onset (t=-{pre} frames)",
                f"end (t=+{post} frames)",
                "pointwise median across instances (biased low -- see spread of thin lines)",
            ],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.45, 0.0),
        )
        cbar = fig.colorbar(left_mappable, cax=cax)
        cbar.set_label("frame within window (time)")

        fig.savefig(output_dir / f"syllable_{syllable:02d}.png", dpi=150)
        plt.close(fig)

        head_nose_trajs = head_trajs[:, :, _NOSE_INDEX, :2]  # (n_instances, T, 2), top-down
        head_path_lengths = _per_instance_path_lengths(head_nose_trajs)
        left_path_lengths = _per_instance_path_lengths(left_trajs)
        right_path_lengths = _per_instance_path_lengths(right_trajs)

        movement_stats[syllable] = {
            "head_path_length_mm": float(head_path_lengths.mean()),
            "head_path_length_mm_std": float(head_path_lengths.std()),
            "left_eye_path_length_deg": float(left_path_lengths.mean()),
            "right_eye_path_length_deg": float(right_path_lengths.mean()),
            "left_eye_path_length_deg_std": float(left_path_lengths.std()),
            "right_eye_path_length_deg_std": float(right_path_lengths.std()),
        }
        direction_stats[syllable] = {
            "head_xy": _instance_displacement_vectors(head_nose_trajs),
            "left_eye": _instance_displacement_vectors(left_trajs),
            "right_eye": _instance_displacement_vectors(right_trajs),
        }

    return movement_stats, direction_stats


def plot_path_length_summary(
    movement_stats: dict[int, dict[str, float]],
    output_path: str | Path,
) -> None:
    """
    Per-syllable comparison of head movement (top) and eye movement (bottom),
    one bar/point per syllable, in syllable order.

    A scatter of head-magnitude vs. eye-magnitude (one dot per syllable) asks
    the reader to eyeball a trend across only as many points as there are
    syllables -- with 10 syllables that's not really legible, and it throws
    away the instance-to-instance variability (`*_std`) entirely. Lining
    both quantities up against a shared syllable axis, with error bars showing
    the spread across each syllable's sampled instances, is more direct: a
    syllable with a small eye bar and small error bar has genuinely little/
    consistent eye movement; a large error bar (relative to the mean) means
    the "mean" is being pulled around by a few outlier instances (e.g. a rare
    large saccade) rather than reflecting something stereotyped about the
    syllable itself -- see `_plot_eye_panel`'s spaghetti plots for those
    syllables to see why.
    """
    syllables = sorted(movement_stats)
    x = np.arange(len(syllables))

    head = np.array([movement_stats[s]["head_path_length_mm"] for s in syllables])
    head_std = np.array([movement_stats[s]["head_path_length_mm_std"] for s in syllables])
    left = np.array([movement_stats[s]["left_eye_path_length_deg"] for s in syllables])
    left_std = np.array([movement_stats[s]["left_eye_path_length_deg_std"] for s in syllables])
    right = np.array([movement_stats[s]["right_eye_path_length_deg"] for s in syllables])
    right_std = np.array([movement_stats[s]["right_eye_path_length_deg_std"] for s in syllables])

    def _clipped_yerr(mean: NDArray, std: NDArray) -> NDArray:
        """Asymmetric error bar that never dips below 0 -- path length can't
        be negative, so a plain symmetric +/-SD bar is misleading whenever
        SD exceeds the mean (common here for the eye channel)."""
        return np.vstack([np.minimum(std, mean), std])

    fig, (ax_head, ax_eye) = plt.subplots(2, 1, figsize=(max(7, len(syllables) * 0.8), 7), sharex=True)
    fig.suptitle("Head vs. eye movement magnitude, per syllable\n(mean ± SD across sampled instances)")

    ax_head.bar(x, head, yerr=_clipped_yerr(head, head_std), capsize=3, color="0.6")
    ax_head.set_ylabel("head path length (mm)")

    offset = 0.15
    ax_eye.errorbar(
        x - offset, left, yerr=_clipped_yerr(left, left_std),
        fmt="o", capsize=3, color="tab:blue", label="left eye",
    )
    ax_eye.errorbar(
        x + offset, right, yerr=_clipped_yerr(right, right_std),
        fmt="o", capsize=3, color="tab:orange", label="right eye",
    )
    ax_eye.set_ylabel("eye path length (deg)")
    ax_eye.set_xlabel("syllable")
    ax_eye.set_xticks(x)
    ax_eye.set_xticklabels(syllables)
    ax_eye.legend()

    for ax in (ax_head, ax_eye):
        ax.grid(axis="y", color="0.9")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Anatomical labels for the four cardinal directions of each polar panel in
# `plot_movement_direction_summary`, confirmed empirically against this
# project's fitted model (left/right ear position in the egocentrically
# aligned frame kpms uses for centroid/heading -- see kinematics_core's
# `[[head_bodyparts]]`-style reference geometry for how the skull frame
# itself is defined). Eye axes need no left/right variant: `adduction` is
# already sign-flipped per eye in `kpms_loader.load_eye_in_head_kpms` so
# "medial" means "toward that eye's own nose side" for both.
_DIRECTION_PANELS = [
    ("head_xy", "head displacement (mm)", ["forward", "left", "back", "right"]),
    ("left_eye", "left eye displacement (deg)", ["medial\n(nose)", "up", "lateral", "down"]),
    ("right_eye", "right eye displacement (deg)", ["medial\n(nose)", "up", "lateral", "down"]),
]


def plot_movement_direction_summary(
    direction_stats: dict[int, dict[str, NDArray]],
    output_path: str | Path,
) -> None:
    """
    Per-syllable movement *direction*, as three polar plots (head, left eye,
    right eye): which way, and how consistently, does each syllable move?

    Path length (`plot_path_length_summary`) can't distinguish "this syllable
    always moves the same way" from "this syllable moves a similar distance
    but in a different, uncorrelated direction each time" -- both give the
    same mean path length, but only the first is a real behavioral
    regularity. This plot answers that directly: each sampled instance's net
    displacement vector (window end minus window onset -- see
    `_instance_displacement_vectors`) is drawn as a small dot at its
    (direction, magnitude); the vector average across instances is drawn as
    a bold radial line, whose thickness/opacity encode the resultant length
    ratio R = |mean vector| / mean(|instance vectors|) -- R=1 means every
    instance moved the same way (bold, opaque line), R=0 means directions
    are effectively random and the "mean" is close to meaningless (thin,
    faint line), matching how path length's mean/SD can already hint at
    inconsistency (see `plot_path_length_summary`) but without saying what
    direction, if any, the instances agree on.

    Axis convention: 0 degrees points along the positive-x axis of the
    channel's own frame, increasing counterclockwise -- see
    `_DIRECTION_PANELS` for what that means anatomically per channel.
    """
    syllables = sorted(direction_stats)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    color_for = {syllable: colors[i % 10] for i, syllable in enumerate(syllables)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), subplot_kw={"projection": "polar"})

    for ax, (key, title, theta_labels) in zip(axes, _DIRECTION_PANELS):
        all_vectors = np.concatenate([direction_stats[s][key] for s in syllables])
        r_max = float(np.linalg.norm(all_vectors, axis=-1).max()) * 1.1

        for syllable in syllables:
            vectors = direction_stats[syllable][key]
            magnitudes = np.linalg.norm(vectors, axis=-1)
            theta = np.arctan2(vectors[:, 1], vectors[:, 0])
            ax.scatter(theta, magnitudes, color=color_for[syllable], alpha=0.3, s=12, zorder=2)

            mean_vector = vectors.mean(axis=0)
            mean_r = float(np.linalg.norm(mean_vector))
            mean_theta = float(np.arctan2(mean_vector[1], mean_vector[0]))
            resultant_ratio = mean_r / magnitudes.mean() if magnitudes.mean() > 0 else 0.0
            ax.plot(
                [mean_theta, mean_theta],
                [0, mean_r],
                color=color_for[syllable],
                linewidth=1 + 3 * resultant_ratio,
                alpha=0.4 + 0.6 * resultant_ratio,
                solid_capstyle="round",
                zorder=3,
            )

        ax.set_title(title)
        ax.set_ylim(0, r_max)
        ax.set_thetagrids([0, 90, 180, 270], theta_labels)

    handles = [
        plt.Line2D([0], [0], color=color_for[s], lw=3, label=f"syllable {s}") for s in syllables
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(syllables), 10), fontsize=8, frameon=False)
    fig.suptitle(
        "Per-syllable movement direction (net displacement, onset→end)\n"
        "dots = individual instances · bold line = vector mean "
        "(thicker/more opaque = more directionally consistent)"
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.90))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from python_code.moseq.kpms_loader import load_eye_in_head_kpms
    from python_code.moseq.run_moseq_pipeline import KPMS_Loader, load_keypoints
    from python_code.moseq.training_config import (
        load_training_config,
        load_training_recording_folders,
    )

    # Must match the recordings the model at `model_name` was trained on --
    # read back from training_recordings.yaml (written by
    # run_moseq_pipeline.main()) rather than duplicated by hand here. Every
    # downstream call is dict-keyed by recording_name, so a list of
    # RecordingFolders works exactly like a single one, just merged.
    project_dir = "/home/scholab/moseq/head_with_pupil_points_405_407_block_pca_test/"
    model_name = "2026_09_23-22_34_13"  # set to the trained model's name

    loader = KPMS_Loader(load_training_config(project_dir)["loader"])
    recording_folders = load_training_recording_folders(project_dir)

    coordinates, _, bodyparts = load_keypoints(loader, recording_folders)
    eye_in_head_left, eye_in_head_right = load_eye_in_head_kpms(recording_folders)
    results = kpms.load_results(project_dir, model_name)

    movement_stats, direction_stats = plot_eye_movement_by_syllable(
        project_dir, model_name, coordinates, eye_in_head_left, eye_in_head_right, results, bodyparts
    )
    plot_dir = Path(project_dir) / model_name / "eye_syllable_plots"
    plot_path_length_summary(movement_stats, output_path=plot_dir / "path_length_summary.png")
    plot_movement_direction_summary(direction_stats, output_path=plot_dir / "direction_summary.png")
