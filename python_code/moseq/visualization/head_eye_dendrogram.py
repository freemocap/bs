"""
Syllable similarity dendrogram for a fitted head + eye AR-HMM
=============================================================

`kpms.plot_similarity_dendrogram` clusters syllables on egocentric keypoint
trajectories. For models from `run_head_eye_arhmm.py` those keypoints are only
the head, so eye movement (what these models are about) can't affect the tree,
and its `min_frequency` cut drops rare syllables.

This clusters every syllable with at least `--min-instances` clean instances on
its mean onset-aligned trajectory in the channels the model was fit on (read
from the project's `feature_scales.csv`, so `--no-pupil`, `--head-rates` and
`--block` fits all work), in the same normalized units the model saw. Next to
the tree, each channel's mean trajectory is drawn as a heatmap row per
syllable, in leaf order, so each cluster can be read off directly.

Saves `{project_dir}/{model_name}/head_eye_plots/similarity_dendrogram.png`
and `syllable_distances.csv`.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.head_eye_dendrogram \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil kappa_1e+03_nlags8_seed0
"""

import argparse
from pathlib import Path

import keypoint_moseq as kpms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

from python_code.moseq.head_eye_features import (
    HeadEyeFeatures,
    keep_channels,
    load_head_eye_features,
    use_head_rates,
)
from python_code.moseq.run_multi_session_kappa_sweeps import KEEP_FRAMES
from python_code.moseq.training_config import load_training_recording_folders
from python_code.moseq.visualization.eye_syllable_viz import _valid_instances

FPS = 120
PRE, POST = 10, 30
MIN_INSTANCES = 10
LINKAGE = "average"


def load_model_features(project_dir: Path) -> HeadEyeFeatures:
    """The channels the project's models were fit on, normalized as they were
    (without the training noise floor)."""
    recording_folders = load_training_recording_folders(project_dir)
    fitted = pd.read_csv(project_dir / "feature_scales.csv")
    channels = list(fitted["channel"])

    features = load_head_eye_features(recording_folders, keep_frames=KEEP_FRAMES)
    if any(c.endswith(("_pitch_rate", "_roll_rate")) for c in channels):
        features = use_head_rates(features, recording_folders)
    features = keep_channels(features, channels)
    if not np.allclose(features.scales, fitted["scale"], rtol=1e-3):
        raise ValueError(
            f"Recomputed scales {features.scales} differ from {project_dir / 'feature_scales.csv'}; "
            "the training recordings or keep frames have changed since the fit."
        )
    return features


def syllable_mean_trajectories(
    features: HeadEyeFeatures, results: dict, pre: int, post: int, min_instances: int
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """`{syllable: (pre + post, n_channels)}` mean onset-aligned trajectory and
    `{syllable: n_instances}`, over instances with no invalid frame."""
    masked = {
        name: np.where(features.valid[name].all(axis=1, keepdims=True), x, np.nan)
        for name, x in features.features.items()
    }
    instances = kpms.get_syllable_instances(
        {name: r["syllable"] for name, r in results.items()},
        pre=pre, post=post, min_duration=1, min_frequency=0, min_instances=1,
    )
    means, counts = {}, {}
    for syllable, syllable_instances in instances.items():
        valid = _valid_instances(
            kpms.get_instance_trajectories(syllable_instances, masked, pre=pre, post=post)
        )
        if len(valid) >= min_instances:
            means[syllable] = valid.mean(axis=0)
            counts[syllable] = len(valid)
    return means, counts


def _usage(results: dict) -> pd.Series:
    labels = np.concatenate([r["syllable"] for r in results.values()])
    return pd.Series(labels).value_counts(normalize=True)


def plot_dendrogram(
    means: dict[int, np.ndarray],
    usage: pd.Series,
    channels: list[str],
    pre: int,
    output_path: Path,
    title: str,
) -> pd.DataFrame:
    syllables = sorted(means)
    X = np.stack([means[s] for s in syllables])  # (S, T, C)
    distances = pdist(X.reshape(len(syllables), -1), "euclidean")
    Z = linkage(distances, LINKAGE)
    order = leaves_list(Z)

    n_rows = len(syllables)
    n_ch = len(channels)
    fig = plt.figure(figsize=(3 + 1.6 * n_ch + 1.2, max(5, 0.22 * n_rows + 1.5)))
    gs = GridSpec(
        2, n_ch + 3, figure=fig, width_ratios=[2.2, 0.2, 0.8] + [1.6] * n_ch,
        height_ratios=[1, 0.02], wspace=0.08, hspace=0.25 * 20 / max(n_rows, 20),
    )

    ax_tree = fig.add_subplot(gs[0, 0])
    # Leaves bottom-to-top in `order`, at y = 5, 15, ...; heatmaps use the same.
    dendrogram(Z, orientation="left", no_labels=True, ax=ax_tree, color_threshold=0, above_threshold_color="0.35")
    ax_tree.set_ylim(0, 10 * n_rows)
    ax_tree.axis("off")

    ordered = [syllables[i] for i in order]
    extent = (None, None, 0, 10 * n_rows)

    ax_use = fig.add_subplot(gs[0, 2])
    ax_use.barh(np.arange(n_rows) * 10 + 5, [usage.get(s, 0) * 100 for s in ordered], height=7, color="0.55")
    ax_use.set_ylim(0, 10 * n_rows)
    ax_use.set_yticks(np.arange(n_rows) * 10 + 5, ordered, fontsize=7)
    ax_use.set_xlabel("usage %", fontsize=7)
    ax_use.tick_params(axis="x", labelsize=6)
    for spine in ("top", "right"):
        ax_use.spines[spine].set_visible(False)

    t_ms = (np.arange(X.shape[1]) - pre) / FPS * 1000
    for c, channel in enumerate(channels):
        ax = fig.add_subplot(gs[0, c + 3])
        values = X[order, :, c]
        limit = np.nanpercentile(np.abs(values), 98) or 1.0
        image = ax.imshow(
            values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower",
            extent=(t_ms[0], t_ms[-1], *extent[2:]), interpolation="nearest",
        )
        ax.axvline(0, color="0.2", lw=0.6, ls=":")
        ax.set_yticks([])
        ax.set_title(channel, fontsize=8)
        ax.set_xlabel("ms from onset", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
        cbar = fig.colorbar(image, cax=fig.add_subplot(gs[1, c + 3]), orientation="horizontal")
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle(f"{title}\nmean onset-aligned trajectory (normalized units), {LINKAGE}-linkage Euclidean", fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(squareform(distances), index=syllables, columns=syllables)


def run(project_dir: str | Path, model_name: str, pre: int, post: int, min_instances: int) -> pd.DataFrame:
    project_dir = Path(project_dir)
    output_dir = project_dir / model_name / "head_eye_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_model_features(project_dir)
    results = kpms.load_results(str(project_dir), model_name)
    means, counts = syllable_mean_trajectories(features, results, pre, post, min_instances)
    usage = _usage(results)
    dropped = sorted(set(usage.index) - set(means))
    if dropped:
        print(f"Too few clean instances (<{min_instances}) to plot syllables {dropped}")

    output_path = output_dir / "similarity_dendrogram.png"
    distances = plot_dendrogram(means, usage, features.channels, pre, output_path, f"{project_dir.name} / {model_name}")
    distances.to_csv(output_dir / "syllable_distances.csv")
    print(f"Saved {output_path} ({len(means)} syllables)")
    return distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir")
    parser.add_argument("model_name")
    parser.add_argument("--pre", type=int, default=PRE, help="frames before onset")
    parser.add_argument("--post", type=int, default=POST, help="frames after onset")
    parser.add_argument("--min-instances", type=int, default=MIN_INSTANCES)
    args = parser.parse_args()
    run(args.project_dir, args.model_name, args.pre, args.post, args.min_instances)
