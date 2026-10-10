"""
PC-level eye-movement diagnostics for HEAD_WITH_PUPIL_POINTS models
====================================================================

`kpms.plot_pcs` (see `run_moseq_pipeline.fit_pca`) draws each PC as a
perturbation of the mean pose, all keypoints on one spatial scale. As with
the raw grid movies/trajectory plots (see `eye_syllable_viz.py`), that scale
problem hides eye movement: pupil-point deflection is a couple mm against
tens of mm of head displacement, so it's hard to tell by eye whether a given
PC is carrying any eye signal at all.

This module answers that question two ways, both cheaper and more direct
than reading loading plots:

`compute_pc_keypoint_group_loadings` / `plot_pc_keypoint_group_loadings`
    For each PC, what fraction of its keypoint-space loading energy sits on
    head keypoints vs. left-eye vs. right-eye pupil points? This is a
    property of the PCA decomposition alone (`fit_pca`'s output) -- it
    doesn't need a fitted AR-HMM. A PC with near-zero eye-group loading is
    not carrying eye motion, full stop, regardless of what the model does
    with it downstream.

`compute_pc_eye_angle_correlation` / `plot_pc_eye_angle_correlation`
    For each PC, how well does its fitted latent trajectory (`x`, from
    `kpms.extract_results`) correlate with the *independently* measured
    eye-in-head angles (`kpms_loader.load_eye_in_head_kpms`, computed by the
    eye kinematics pipeline, not from these pupil-point keypoints at all)?
    This is the check that actually matters: a PC can have eye keypoints in
    its loading and still fail to track real eye movement (e.g. if eye
    motion is small enough that it's dominated by noise in that PC), and
    conversely a PC nominally "about" a head keypoint can pick up correlated
    eye motion through egocentric alignment. Correlating against a
    ground-truth signal computed outside the PCA sidesteps both failure
    modes.

Entry point
-----------
    from python_code.moseq.visualization.eye_pc_viz import run_eye_pc_diagnostics

    stats = run_eye_pc_diagnostics(
        project_dir, model_name, recording_folder,
        use_bodyparts=HEAD_WITH_PUPIL_POINTS_BODYPARTS,
        keypoint_groups={
            "head": HEAD_BODYPARTS,
            "left_eye": HEAD_WITH_PUPIL_POINTS_LEFT_EYE_BODYPARTS,
            "right_eye": HEAD_WITH_PUPIL_POINTS_RIGHT_EYE_BODYPARTS,
        },
    )

Called automatically from `run_moseq_pipeline.run_pipeline` whenever
`eye_diagnostics_recording_folder` is passed (wired up in `main()` for
`KPMS_Loader.HEAD_WITH_PUPIL_POINTS`).
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def compute_pc_keypoint_group_loadings(
    pca,
    use_bodyparts: list[str],
    keypoint_groups: dict[str, list[str]],
    n_pcs: int | None = None,
) -> dict:
    """
    Decompose each PC's loading vector by keypoint group.

    kpms fits PCA on an egocentric "centered embedding" of the keypoints
    (`k` keypoints -> `k-1` embedded dims, via `center_embedding`; see
    `keypoint_moseq.viz.plot_pcs`, which uses the same `Gamma` matrix to draw
    loadings back in real keypoint space). This mirrors that transform to
    get each PC's per-keypoint displacement, then sums squared displacement
    within each named group of keypoints.

    Parameters
    ----------
    pca:
        Fitted PCA model, as returned by `kpms.fit_pca`/`kpms.load_pca`. Note
        this holds the *full* PCA decomposition (all `k-1` embedded
        dimensions worth of components) -- only its top `latent_dim` are
        actually used by the AR-HMM, so pass `n_pcs=latent_dim` to restrict
        this to the PCs that matter (`run_eye_pc_diagnostics` does this
        automatically).
    use_bodyparts:
        The bodypart list the PCA was fit on (`config()["use_bodyparts"]`),
        in order.
    keypoint_groups:
        `{group_name: [bodypart names]}`. Must partition `use_bodyparts`
        exactly (every bodypart in exactly one group) -- this raises
        otherwise, since a silently-dropped or double-counted keypoint would
        make the fractions meaningless.
    n_pcs:
        Restrict to the first `n_pcs` components (`pca.components_` is
        ordered by explained variance). Defaults to all of them.

    Returns
    -------
    dict with:
        `fraction_by_group`: `{group_name: (n_pcs,) array}`, summing to 1
            across groups for each PC.
        `n_pcs`: number of PCs actually used (after the `n_pcs` truncation).
    """
    from jax_moseq.models.keypoint_slds import center_embedding

    covered = {bp for bps in keypoint_groups.values() for bp in bps}
    if covered != set(use_bodyparts):
        missing = set(use_bodyparts) - covered
        extra = covered - set(use_bodyparts)
        raise ValueError(
            "keypoint_groups must partition use_bodyparts exactly. "
            f"Missing: {sorted(missing)}. Not in use_bodyparts: {sorted(extra)}."
        )

    k = len(use_bodyparts)
    d = len(pca.mean_) // (k - 1)
    if n_pcs is None:
        n_pcs = pca.components_.shape[0]

    Gamma = np.array(center_embedding(k))  # (k, k-1)
    components = pca.components_[:n_pcs].reshape(n_pcs, k - 1, d)
    per_keypoint = np.einsum("ij,pjd->pid", Gamma, components)  # (n_pcs, k, d)
    energy = (per_keypoint**2).sum(axis=-1)  # (n_pcs, k)
    energy_fraction = energy / energy.sum(axis=-1, keepdims=True)

    name_to_index = {name: i for i, name in enumerate(use_bodyparts)}
    fraction_by_group = {
        group_name: energy_fraction[:, [name_to_index[bp] for bp in group_bodyparts]].sum(axis=1)
        for group_name, group_bodyparts in keypoint_groups.items()
    }

    return {"fraction_by_group": fraction_by_group, "n_pcs": n_pcs}


def plot_pc_keypoint_group_loadings(
    loadings: dict,
    output_path: str | Path,
) -> None:
    """Stacked bar chart of `compute_pc_keypoint_group_loadings`'s output, one bar per PC."""
    fraction_by_group = loadings["fraction_by_group"]
    n_pcs = loadings["n_pcs"]
    group_names = list(fraction_by_group.keys())

    fig, ax = plt.subplots(figsize=(max(6, n_pcs * 0.7), 4))
    x = np.arange(n_pcs)
    bottom = np.zeros(n_pcs)
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))
    for color, group_name in zip(colors, group_names):
        fractions = fraction_by_group[group_name]
        ax.bar(x, fractions, bottom=bottom, label=group_name, color=color)
        bottom += fractions

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_pcs)])
    ax.set_xlabel("PC")
    ax.set_ylabel("fraction of loading energy")
    ax.set_ylim(0, 1.02)
    ax.set_title("PC loading energy by keypoint group")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


_EYE_ANGLE_CHANNELS = ["left_adduction", "left_elevation", "right_adduction", "right_elevation"]


def compute_pc_eye_angle_correlation(
    latent_states: dict[str, NDArray],
    eye_in_head_left: dict[str, NDArray],
    eye_in_head_right: dict[str, NDArray],
) -> dict:
    """
    Per-PC R^2 against each of the four eye-in-head angle channels, pooled
    across all recordings.

    Parameters
    ----------
    latent_states:
        `{recording_name: (n_frames, latent_dim) array}`, e.g.
        `{k: v["latent_state"] for k, v in results.items()}` from
        `kpms.extract_results`/`kpms.load_results`.
    eye_in_head_left, eye_in_head_right:
        From `kpms_loader.load_eye_in_head_kpms`, `{recording_name:
        (n_frames, 2) array}` of `[adduction_deg, elevation_deg]`, on the
        same per-recording frame grid as `latent_states` (see
        `load_eye_in_head_kpms`'s docstring) -- must have matching keys and
        frame counts.

    Returns
    -------
    dict with:
        `r2`: `(latent_dim, 4)` array of R^2 (squared Pearson correlation),
            columns ordered as `channels`. NaN wherever fewer than 10 frames
            have both values finite.
        `channels`: the 4 column names.
        `best_channel_per_dim`: for each PC, the channel it correlates with
            most strongly.
        `best_r2_per_dim`: that best R^2, per PC.
    """
    latent_parts = []
    angle_parts = []
    for name, x in latent_states.items():
        for side, eye_in_head in (("left", eye_in_head_left), ("right", eye_in_head_right)):
            if eye_in_head[name].shape[0] != x.shape[0]:
                raise ValueError(
                    f"Frame count mismatch for '{name}': latent_state has {x.shape[0]} "
                    f"frames but {side} eye_in_head has {eye_in_head[name].shape[0]}. "
                    "These are expected to be on the same common-timestamp grid -- see "
                    "kpms_loader.load_eye_in_head_kpms."
                )
        latent_parts.append(np.asarray(x))
        angle_parts.append(np.concatenate([eye_in_head_left[name], eye_in_head_right[name]], axis=1))

    x_all = np.concatenate(latent_parts, axis=0)  # (n_frames, latent_dim)
    angles_all = np.concatenate(angle_parts, axis=0)  # (n_frames, 4)

    n_pcs = x_all.shape[1]
    r2 = np.full((n_pcs, len(_EYE_ANGLE_CHANNELS)), np.nan)
    for dim in range(n_pcs):
        for channel in range(len(_EYE_ANGLE_CHANNELS)):
            valid = np.isfinite(x_all[:, dim]) & np.isfinite(angles_all[:, channel])
            if valid.sum() < 10:
                continue
            r = np.corrcoef(x_all[valid, dim], angles_all[valid, channel])[0, 1]
            r2[dim, channel] = r**2

    best_channel_idx = np.nanargmax(r2, axis=1)
    return {
        "r2": r2,
        "channels": _EYE_ANGLE_CHANNELS,
        "best_channel_per_dim": [_EYE_ANGLE_CHANNELS[i] for i in best_channel_idx],
        "best_r2_per_dim": r2[np.arange(n_pcs), best_channel_idx],
    }


def plot_pc_eye_angle_correlation(
    correlation: dict,
    output_path: str | Path,
) -> None:
    """Heatmap of `compute_pc_eye_angle_correlation`'s `r2`, PC x channel, annotated with values."""
    r2 = correlation["r2"]
    channels = correlation["channels"]
    n_pcs = r2.shape[0]

    fig, ax = plt.subplots(figsize=(5.5, max(3, n_pcs * 0.45)))
    im = ax.imshow(r2, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=30, ha="right")
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels([str(i) for i in range(n_pcs)])
    ax.set_ylabel("PC")
    for dim in range(n_pcs):
        for channel in range(len(channels)):
            value = r2[dim, channel]
            if np.isfinite(value):
                ax.text(
                    channel, dim, f"{value:.2f}", ha="center", va="center",
                    color="white" if value < 0.6 else "black", fontsize=8,
                )
    ax.set_title("latent PC vs. eye-in-head angle ($R^2$)")
    fig.colorbar(im, ax=ax, label="$R^2$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_eye_pc_diagnostics(
    project_dir: str | Path,
    model_name: str,
    recording_folder,
    use_bodyparts: list[str],
    keypoint_groups: dict[str, list[str]],
    pca=None,
    results: dict | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Run both PC/eye-movement diagnostics and save their plots.

    Parameters
    ----------
    project_dir, model_name:
        Used to load `pca`/`results` if not passed, and as the save
        location for the plots.
    recording_folder:
        The `RecordingFolder`(s) the model was trained on (same value passed
        to `load_keypoints`) -- used to load ground-truth eye-in-head angles
        via `kpms_loader.load_eye_in_head_kpms`.
    use_bodyparts, keypoint_groups:
        See `compute_pc_keypoint_group_loadings`.
    pca, results:
        Optionally pass already-loaded objects (`kpms.load_pca`,
        `kpms.load_results`) to avoid loading them again.
    output_dir:
        Defaults to `{project_dir}/{model_name}/eye_pc_plots`.

    Returns
    -------
    dict with `loading_by_keypoint_group` (from `compute_pc_keypoint_group_loadings`)
    and `eye_angle_correlation` (from `compute_pc_eye_angle_correlation`).
    """
    import keypoint_moseq as kpms  # deferred: heavy, JAX-backed import

    from python_code.moseq.kpms_loader import load_eye_in_head_kpms

    if pca is None:
        pca = kpms.load_pca(str(project_dir))
    if results is None:
        results = kpms.load_results(str(project_dir), model_name)

    if output_dir is None:
        output_dir = Path(project_dir) / model_name / "eye_pc_plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_states = {name: r["latent_state"] for name, r in results.items()}
    latent_dim = next(iter(latent_states.values())).shape[1]

    # `pca` holds the full decomposition (all k-1 embedded dims); restrict to
    # the top `latent_dim` PCs actually used by the AR-HMM, so this lines up
    # one-to-one with the correlation check below (and with
    # variance_explained_viz's per-dim AR-R^2/eta^2).
    loadings = compute_pc_keypoint_group_loadings(
        pca, use_bodyparts, keypoint_groups, n_pcs=latent_dim
    )
    plot_pc_keypoint_group_loadings(loadings, output_dir / "pc_loading_by_keypoint_group.png")

    eye_in_head_left, eye_in_head_right = load_eye_in_head_kpms(recording_folder)
    correlation = compute_pc_eye_angle_correlation(latent_states, eye_in_head_left, eye_in_head_right)
    plot_pc_eye_angle_correlation(correlation, output_dir / "pc_eye_angle_correlation.png")

    return {"loading_by_keypoint_group": loadings, "eye_angle_correlation": correlation}


if __name__ == "__main__":
    # Ad hoc test against an already-fitted HEAD_WITH_PUPIL_POINTS project --
    # rebuilds the exact recording folders it was trained on from
    # `training_recordings.yaml` (see `training_config.py`) instead of
    # duplicating that list by hand, so this can't silently drift from what
    # the model actually saw. Point `project_dir`/`model_name` at any other
    # fitted HEAD_WITH_PUPIL_POINTS project to test that one instead.
    from python_code.moseq.training_config import load_training_recording_folders
    from python_code.moseq.utils.bodyparts import (
        HEAD_BODYPARTS,
        HEAD_WITH_PUPIL_POINTS_BODYPARTS,
        HEAD_WITH_PUPIL_POINTS_LEFT_EYE_BODYPARTS,
        HEAD_WITH_PUPIL_POINTS_RIGHT_EYE_BODYPARTS,
    )

    project_dir = "/home/scholab/moseq/head_with_pupil_points_405_407_test"
    model_name = "2026_09_23-01_08_09"

    recording_folders = load_training_recording_folders(project_dir)
    print(f"Loaded {len(recording_folders)} recording folder(s) from training_recordings.yaml")

    stats = run_eye_pc_diagnostics(
        project_dir,
        model_name,
        recording_folders,
        use_bodyparts=HEAD_WITH_PUPIL_POINTS_BODYPARTS,
        keypoint_groups={
            "head": HEAD_BODYPARTS,
            "left_eye": HEAD_WITH_PUPIL_POINTS_LEFT_EYE_BODYPARTS,
            "right_eye": HEAD_WITH_PUPIL_POINTS_RIGHT_EYE_BODYPARTS,
        },
    )

    print("fraction_by_group:")
    for group, fractions in stats["loading_by_keypoint_group"]["fraction_by_group"].items():
        print(f"  {group}: {fractions.round(3)}")
    print("best_channel_per_dim:", stats["eye_angle_correlation"]["best_channel_per_dim"])
    print("best_r2_per_dim:", stats["eye_angle_correlation"]["best_r2_per_dim"].round(3))
    print(f"Plots saved to {project_dir}/{model_name}/eye_pc_plots/")
