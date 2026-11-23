"""
Left vs. right examples of a symmetric syllable
================================================

`eye_syllable_viz.plot_eye_movement_by_syllable` density-samples one exemplar
and its nearest neighbors per syllable, so a syllable that contains mirror
image versions (head moves left with the eyes going right, and head moves right
with the eyes going left) is shown in one direction only.

For each requested syllable, this splits its instances by the nose's net
lateral displacement in the onset-aligned egocentric frame (the frame of the
head panel), density-samples each side separately, and draws the usual head /
left eye / right eye panels plus their polar direction plots, head-moves-left
on top and head-moves-right underneath.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.bidirectional_syllable_viz \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil kappa_1e+03_nlags8_seed0 6 12 20
"""

import argparse
from pathlib import Path

import keypoint_moseq as kpms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from python_code.moseq.kpms_loader import load_eye_in_head_kpms
from python_code.moseq.run_moseq_pipeline import KPMS_Loader, load_keypoints
from python_code.moseq.training_config import (
    load_training_config,
    load_training_loader_options,
    load_training_recording_folders,
)
from python_code.moseq.utils.bodyparts import HEAD_BODYPARTS
from python_code.moseq.visualization.eye_syllable_viz import (
    _DIRECTION_PANELS,
    _NOSE_INDEX,
    _draw_syllable_directions,
    _instance_displacement_vectors,
    _plot_eye_panel,
    _plot_head_panel,
    _style_direction_axis,
    _valid_instances,
)

# Instances whose nose moves less than this sideways (mm) go in neither group.
MIN_LATERAL_MM = 5.0
_SIDES = [("left", 1.0), ("right", -1.0)]  # egocentric +y is the animal's left


def plot_bidirectional_syllables(
    project_dir: str | Path,
    model_name: str,
    syllables: list[int],
    pre: int = 5,
    post: int = 15,
    min_duration: float = 3,
    n_neighbors: int = 24,
    output_dir: str | Path | None = None,
) -> None:
    project_dir = Path(project_dir)
    output_dir = Path(output_dir or project_dir / model_name / "eye_syllable_plots" / "bidirectional")
    output_dir.mkdir(parents=True, exist_ok=True)

    recording_folders = load_training_recording_folders(project_dir)
    coordinates, _, bodyparts = load_keypoints(
        KPMS_Loader(load_training_config(project_dir)["loader"]),
        recording_folders,
        **load_training_loader_options(project_dir),
    )
    eye_left, eye_right = load_eye_in_head_kpms(recording_folders, quality_level="medium")
    results = kpms.load_results(str(project_dir), model_name)

    head_coordinates = kpms.reindex_by_bodyparts(coordinates, bodyparts, HEAD_BODYPARTS)
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}
    instances = kpms.get_syllable_instances(
        {k: v["syllable"] for k, v in results.items()},
        pre=pre, post=post, min_duration=min_duration, min_frequency=0.0, min_instances=n_neighbors,
    )

    # Replace each requested syllable with its two sides, keeping every other
    # syllable so density sampling's global density is the usual one.
    groups, totals = {}, {}
    for syllable in syllables:
        trajs = kpms.get_instance_trajectories(
            instances[syllable], head_coordinates, pre=pre, post=post, centroids=centroids, headings=headings
        )
        lateral = _instance_displacement_vectors(trajs[:, :, _NOSE_INDEX, :2])[:, 1]
        totals[syllable] = len(lateral)
        for side, sign in _SIDES:
            members = [inst for inst, y in zip(instances[syllable], lateral) if sign * y > MIN_LATERAL_MM]
            if len(members) < n_neighbors:
                raise ValueError(f"syllable {syllable}: only {len(members)} head-moves-{side} instances")
            groups[(syllable, side)] = members
        del instances[syllable]
    instances.update(groups)

    sampled = kpms.sample_instances(
        instances, n_neighbors, mode="density", coordinates=head_coordinates,
        centroids=centroids, headings=headings, pre=pre, post=post, n_neighbors=n_neighbors,
    )

    all_angles = np.concatenate([a for eye in (eye_left, eye_right) for a in eye.values()])
    lim_deg = float(np.nanpercentile(np.abs(all_angles), 99)) * 1.1

    for syllable in syllables:
        fig = plt.figure(figsize=(16, 19))
        fig.suptitle(f"syllable {syllable}: head moves left (top) vs. right (bottom)", fontsize=14)
        for row, (side, _) in enumerate(_SIDES):
            group = instances[(syllable, side)]
            picks = sampled[(syllable, side)]
            top = 0.5 - 0.5 * row  # each side gets half the figure
            fig.text(
                0.03, top + 0.445,
                f"Head moves {side}: {len(group)} of {totals[syllable]} instances "
                f"({len(group) / totals[syllable]:.0%}); {len(picks)} density-sampled shown",
                fontsize=12, fontweight="bold",
            )
            ax_head = fig.add_axes([0.04, top + 0.255, 0.27, 0.165])
            ax_left = fig.add_axes([0.37, top + 0.255, 0.24, 0.165])
            ax_right = fig.add_axes([0.63, top + 0.255, 0.24, 0.165])
            cax = fig.add_axes([0.90, top + 0.255, 0.015, 0.165])
            boxes = {
                "head_xy": [0.04, top + 0.03, 0.27, 0.165],
                "left_eye": [0.37, top + 0.03, 0.24, 0.165],
                "right_eye": [0.63, top + 0.03, 0.24, 0.165],
            }

            head = kpms.get_instance_trajectories(
                picks, head_coordinates, pre=pre, post=post, centroids=centroids, headings=headings
            )
            left = kpms.get_instance_trajectories(picks, eye_left, pre=pre, post=post)
            right = kpms.get_instance_trajectories(picks, eye_right, pre=pre, post=post)
            vectors = {
                "head_xy": _instance_displacement_vectors(head[:, :, _NOSE_INDEX, :2]),
                "left_eye": _instance_displacement_vectors(_valid_instances(left)),
                "right_eye": _instance_displacement_vectors(_valid_instances(right)),
            }

            _plot_head_panel(ax_head, np.median(head, axis=0)[:, :, :2])
            mappable, _ = _plot_eye_panel(ax_left, left, lim_deg, "left eye")
            _plot_eye_panel(ax_right, right, lim_deg, "right eye", mirror_x=True)
            fig.colorbar(mappable, cax=cax).set_label("frame within window (time)")
            for key, title, theta_labels, r_max in _DIRECTION_PANELS:
                ax = fig.add_axes(boxes[key], projection="polar")
                ratio = _draw_syllable_directions(ax, vectors[key], "#1a2a6c", str(syllable))
                if ratio is not None:
                    title += f"\nnet displacement, {len(vectors[key])} instances, R = {ratio:.2f}"
                _style_direction_axis(ax, key, title, theta_labels, r_max)

        fig.text(
            0.5, 0.005,
            f"Window: {pre} frames before to {post} frames after onset. Circles = window start, "
            f"stars = end, dashed red = pointwise median. Instances moving < {MIN_LATERAL_MM:.0f} mm "
            "sideways are in neither group.",
            ha="center", fontsize=9, color="0.3",
        )
        path = output_dir / f"syllable_{syllable:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("syllables", type=int, nargs="+")
    args = parser.parse_args()
    plot_bidirectional_syllables(args.project_dir, args.model_name, args.syllables)
