"""
Report for a joint head-eye AR-HMM sweep and its shifted control
================================================================

For real fits from `run_head_eye_arhmm.py` and their `--shifted-control`
fits (same seeds, a separate project dir), generates every per-model plot and
then a summary comparing the two.

Per model
    `head_eye_syllable_viz` plots (real and shifted). For real fits only:
    `eye_syllable_viz` per-syllable plots, `development_stats`, and kpms
    trajectory plots and grid movies via `replot_session.replot`.

Summary, in `{real_dir}/sweep_report`
    `heldout_gap_vs_age.png`: per-session held-out log likelihood, real minus
    shifted, against days since eye opening, next to each session's
    frame-level head-eye coupling. The gap is what the joint model gains from
    head-eye coupling, so this asks whether that grows with age.
    `heldout_traces.png`: held-out likelihood over iterations, for convergence.
    `channel_eta_squared.png`: onset-aligned eta^2 per channel, real vs.
    shifted, across seeds.
    Each plot's data is also saved as CSV.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.head_eye_sweep_report \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil_trainshift \\
        --kappa 1e3 --nlags 8 --seeds 0 1 2
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import keypoint_moseq as kpms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax_moseq.models import arhmm

from python_code.moseq.development_stats import (
    COHORT_COLORS,
    COHORT_LABELS,
    COHORT_OF_ANIMAL,
    MARKER_OF_ANIMAL,
    analyze,
    parse_session,
)
from python_code.moseq.head_eye_features import HeadEyeFeatures, load_head_eye_features
from python_code.moseq.kpms_loader import load_eye_in_head_kpms
from python_code.moseq.replot_session import replot
from python_code.moseq.run_head_eye_arhmm import _batched, _train_and_heldout_masks, stale_split_sessions
from python_code.moseq.run_moseq_pipeline import KPMS_Loader, load_keypoints
from python_code.moseq.run_multi_session_kappa_sweeps import KEEP_FRAMES
from python_code.moseq.training_config import (
    load_training_config,
    load_training_loader_options,
    load_training_recording_folders,
)
from python_code.moseq.visualization.eye_syllable_viz import (
    plot_eye_movement_by_syllable,
    plot_movement_direction_summary,
    plot_path_length_summary,
)
from python_code.moseq.visualization.head_eye_syllable_viz import (
    run_head_eye_diagnostics,
    session_coupling_table,
)

_INK = "#0b0b0b"
_MUTED_INK = "#52514e"
_GRID = "#e4e3df"
_REAL_COLOR, _SHIFTED_COLOR = COHORT_COLORS[0], COHORT_COLORS[1]
_SEED_COLORS = COHORT_COLORS


def _model_name(kappa: float, nlags: int, seed: int, shifted: bool) -> str:
    return f"{'shifted_' if shifted else ''}kappa_{kappa:.0e}_nlags{nlags}_seed{seed}"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_MUTED_INK)
    ax.tick_params(colors=_MUTED_INK, labelcolor=_INK)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def heldout_ll_by_session(
    project_dir: Path, model_name: str, features: HeadEyeFeatures, nlags: int,
    exclude: list[str] = (),
) -> pd.DataFrame:
    """
    Held-out log likelihood per frame and held-out frame count for each
    recording not in `exclude`, scored the same way as
    `run_head_eye_arhmm._heldout_ll_per_frame`. The shifted control only moves
    eye frames inside the training window, so both fits share these held-out
    masks.
    """
    model, _, _, _ = kpms.load_checkpoint(str(project_dir), model_name)
    params = model["params"]
    x = model["states"]["x"]
    _, heldout, _ = _train_and_heldout_masks(features, nlags)
    _, padding, (keys, _) = _batched(features.features)
    heldout_mask = _batched(heldout)[0] * padding

    rows, total_ll, total_frames = {}, 0.0, 0.0
    for name in features.features:
        if name in exclude:
            continue
        mask = jnp.asarray(heldout_mask * (keys == name)[:, None], dtype=float)
        ll = float(arhmm.marginal_log_likelihood(mask, x, params["Ab"], params["Q"], params["pi"]))
        frames = float(mask[:, nlags:].sum())
        rows[name] = {"ll": ll / frames, "frames": int(frames)}
        total_ll, total_frames = total_ll + ll, total_frames + frames
    print(f"{model_name}: pooled held-out LL {total_ll / total_frames:.4f} nats/frame")
    return pd.DataFrame.from_dict(rows, orient="index")


def heldout_gap_table(
    real_dir: Path, shifted_dir: Path, features: HeadEyeFeatures, kappa: float, nlags: int, seeds: list[int]
) -> pd.DataFrame:
    model_dirs = [
        directory / _model_name(kappa, nlags, seed, shifted)
        for seed in seeds
        for directory, shifted in [(real_dir, False), (shifted_dir, True)]
    ]
    stale = stale_split_sessions(model_dirs, features)
    if stale:
        print(
            f"[warn] Leaving out {stale}: these fits were trained with an older "
            "train/held-out split, so the current held-out frames overlap their "
            "training frames. Refit to score them."
        )

    rows = []
    for seed in seeds:
        real = heldout_ll_by_session(real_dir, _model_name(kappa, nlags, seed, False), features, nlags, stale)
        shifted = heldout_ll_by_session(shifted_dir, _model_name(kappa, nlags, seed, True), features, nlags, stale)
        for name in real.index:
            animal, eo_day = parse_session(name)
            rows.append({
                "recording": name, "animal": animal, "eo_day": eo_day, "seed": seed,
                "heldout_frames": real.loc[name, "frames"],
                "real_ll": real.loc[name, "ll"], "shifted_ll": shifted.loc[name, "ll"],
                "gap": real.loc[name, "ll"] - shifted.loc[name, "ll"],
            })
    return pd.DataFrame(rows)


def _plot_by_animal(ax, sessions: pd.DataFrame, x: str, y: str, low: str | None = None, high: str | None = None):
    """One marker per session, colored by cohort, shaped by animal, with each
    animal's sessions joined in age order."""
    for animal, group in sessions.groupby("animal"):
        group = group.sort_values(x)
        color = COHORT_COLORS[COHORT_OF_ANIMAL[animal]]
        if low is not None:
            ax.vlines(group[x], group[low], group[high], color=color, linewidth=1.5, alpha=0.6)
        ax.plot(group[x], group[y], color=color, linewidth=1, alpha=0.5)
        ax.scatter(
            group[x], group[y], s=48, color=color, marker=MARKER_OF_ANIMAL[animal],
            edgecolor="#fcfcfb", linewidth=1.5, zorder=3, label=f"ferret {animal}",
        )


def plot_heldout_gap_vs_age(gaps: pd.DataFrame, coupling: pd.DataFrame, path: Path) -> pd.DataFrame:
    sessions = (
        gaps.groupby(["recording", "animal", "eo_day"])["gap"]
        .agg(gap="mean", gap_min="min", gap_max="max")
        .reset_index()
        .merge(coupling, on="recording")
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    panels = [
        ("eo_day", "gap", "Days since eye opening",
         "Held-out LL, real − shifted (nats/frame)", "Joint-model coupling gain"),
        ("eo_day", "yaw_r", "Days since eye opening",
         "Peak lagged r, yaw rate vs. horizontal eye velocity", "Yaw coupling (negative = compensatory)"),
        ("eo_day", "pitch_r", "Days since eye opening",
         "Peak lagged r, pitch vs. vertical eye velocity", "Pitch coupling"),
        ("yaw_r", "gap", "Peak lagged r, yaw (frame level)",
         "Held-out LL, real − shifted (nats/frame)", "Coupling gain vs. yaw coupling"),
    ]
    for ax, (x, y, xlabel, ylabel, title) in zip(axes, panels):
        has_range = y == "gap"
        if x == "eo_day":
            _plot_by_animal(ax, sessions, x, y, "gap_min" if has_range else None, "gap_max" if has_range else None)
        else:
            for animal, group in sessions.groupby("animal"):
                ax.scatter(
                    group[x], group[y], s=48, color=COHORT_COLORS[COHORT_OF_ANIMAL[animal]],
                    marker=MARKER_OF_ANIMAL[animal], edgecolor="#fcfcfb", linewidth=1.5, zorder=3,
                )
        ax.axhline(0, color=_MUTED_INK, linewidth=0.8, zorder=1)
        ax.set_xlabel(xlabel, color=_INK)
        ax.set_ylabel(ylabel, color=_INK)
        ax.set_title(title, color=_INK, loc="left", fontsize=11)
        _style(ax)
    axes[0].annotate("bars: range across seeds", xy=(0.02, 0.97), xycoords="axes fraction",
                     va="top", fontsize=8, color=_MUTED_INK)
    handles, labels = axes[0].get_legend_handles_labels()
    order = np.argsort([COHORT_OF_ANIMAL[label.split()[-1]] for label in labels], kind="stable")
    fig.legend(
        [handles[i] for i in order], [f"{labels[i]} ({COHORT_LABELS[COHORT_OF_ANIMAL[labels[i].split()[-1]]]})"
                                      for i in order],
        loc="lower center", ncol=6, frameon=False, labelcolor=_INK,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return sessions


def plot_heldout_traces(
    real_dir: Path, shifted_dir: Path, kappa: float, nlags: int, seeds: list[int], path: Path
) -> pd.DataFrame:
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, seed in enumerate(seeds):
        for shifted, project_dir, style in [(False, real_dir, "-"), (True, shifted_dir, "--")]:
            trace = pd.read_json(
                project_dir / _model_name(kappa, nlags, seed, shifted) / "heldout_trace.json", typ="series"
            ).sort_index()
            label = f"seed {seed}, {'shifted' if shifted else 'real'}"
            ax.plot(trace.index, trace.values, style, color=_SEED_COLORS[i], linewidth=2, marker="o",
                    markersize=5, label=label)
            rows += [{"seed": seed, "shifted": shifted, "iteration": it, "heldout_ll": ll} for it, ll in trace.items()]
    ax.set_xlabel("Iteration", color=_INK)
    ax.set_ylabel("Held-out LL (nats/frame)", color=_INK)
    ax.set_title("Held-out likelihood during fitting (solid real, dashed shifted)", color=_INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, labelcolor=_INK)
    _style(ax)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_channel_eta_squared(
    real_dir: Path, shifted_dir: Path, kappa: float, nlags: int, seeds: list[int], path: Path
) -> pd.DataFrame:
    """Onset-aligned eta^2 per channel from each model's `head_eye_plots`."""
    rows = []
    for seed in seeds:
        for shifted, project_dir in [(False, real_dir), (True, shifted_dir)]:
            eta = pd.read_csv(
                project_dir / _model_name(kappa, nlags, seed, shifted) / "head_eye_plots" / "channel_eta_squared.csv",
                index_col=0,
            )["eta_squared"]
            rows += [{"seed": seed, "shifted": shifted, "channel": c, "eta_squared": v} for c, v in eta.items()]
    table = pd.DataFrame(rows)

    channels = list(dict.fromkeys(table["channel"]))
    modeled = set(pd.read_csv(real_dir / "feature_scales.csv")["channel"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, channel in enumerate(channels):
        if channel not in modeled:
            ax.axvspan(i - 0.45, i + 0.45, color=_GRID, alpha=0.6, zorder=0, linewidth=0)
    for shifted, color, offset, label in [(False, _REAL_COLOR, -0.15, "real"), (True, _SHIFTED_COLOR, 0.15, "shifted")]:
        subset = table[table["shifted"] == shifted]
        positions = np.array([channels.index(c) for c in subset["channel"]]) + offset
        ax.scatter(positions, subset["eta_squared"], s=16, color=color, alpha=0.5, edgecolor="none")
        means = subset.groupby("channel")["eta_squared"].mean().reindex(channels)
        ax.scatter(np.arange(len(channels)) + offset, means, s=80, marker="_", color=color, linewidth=2.5,
                   label=f"{label} (bar = mean of {len(seeds)} seeds)")
    ax.set_xticks(
        np.arange(len(channels)),
        [c if c in modeled else f"{c}\n(not in model)" for c in channels],
        rotation=30, ha="right",
    )
    ax.set_ylabel("Onset-aligned eta² by syllable", color=_INK)
    ax.set_title("How much of each channel's movement syllable identity explains", color=_INK, loc="left",
                 fontsize=11)
    ax.legend(frameon=False, labelcolor=_INK)
    _style(ax)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return table


def per_model_plots(project_dir: Path, model_name: str, real: bool, movies: bool) -> None:
    run_head_eye_diagnostics(project_dir, model_name)
    if not real:
        return

    recording_folders = load_training_recording_folders(project_dir)
    coordinates, _, bodyparts = load_keypoints(
        KPMS_Loader(load_training_config(project_dir)["loader"]),
        recording_folders,
        **load_training_loader_options(project_dir),
    )
    left, right = load_eye_in_head_kpms(recording_folders, quality_level="medium")
    results = kpms.load_results(str(project_dir), model_name)
    movement_stats, direction_stats = plot_eye_movement_by_syllable(
        project_dir, model_name, coordinates, left, right, results, bodyparts
    )
    plot_dir = project_dir / model_name / "eye_syllable_plots"
    plot_path_length_summary(movement_stats, output_path=plot_dir / "path_length_summary.png")
    plot_movement_direction_summary(
        direction_stats, output_path=plot_dir / "direction_summary.png"
    )

    analyze(project_dir.name, model_name, project_dir=project_dir)
    replot(project_dir, model_name, n_neighbors=24, skip_grid_movies=not movies)


def run_report(
    real_dir: Path, shifted_dir: Path, kappa: float, nlags: int, seeds: list[int],
    movies: bool = True, skip_per_model: bool = False,
) -> None:
    if not skip_per_model:
        for seed in seeds:
            per_model_plots(real_dir, _model_name(kappa, nlags, seed, False), real=True, movies=movies)
            per_model_plots(shifted_dir, _model_name(kappa, nlags, seed, True), real=False, movies=False)

    out_dir = real_dir / "sweep_report"
    out_dir.mkdir(exist_ok=True)
    features = load_head_eye_features(load_training_recording_folders(real_dir), keep_frames=KEEP_FRAMES)

    gaps = heldout_gap_table(real_dir, shifted_dir, features, kappa, nlags, seeds)
    gaps.to_csv(out_dir / "heldout_gap_by_session.csv", index=False)
    coupling = session_coupling_table(features)
    sessions = plot_heldout_gap_vs_age(gaps, coupling, out_dir / "heldout_gap_vs_age.png")
    sessions.to_csv(out_dir / "heldout_gap_vs_age.csv", index=False)
    plot_heldout_traces(real_dir, shifted_dir, kappa, nlags, seeds, out_dir / "heldout_traces.png").to_csv(
        out_dir / "heldout_traces.csv", index=False
    )
    plot_channel_eta_squared(real_dir, shifted_dir, kappa, nlags, seeds, out_dir / "channel_eta_squared.png").to_csv(
        out_dir / "channel_eta_squared.csv", index=False
    )
    print(sessions.sort_values(["animal", "eo_day"]).round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("real_dir", type=Path)
    parser.add_argument("shifted_dir", type=Path)
    parser.add_argument("--kappa", type=float, default=1e3)
    parser.add_argument("--nlags", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no-movies", action="store_true", help="skip kpms grid movies")
    parser.add_argument("--summary-only", action="store_true", help="skip the per-model plots")
    args = parser.parse_args()

    run_report(args.real_dir, args.shifted_dir, args.kappa, args.nlags, args.seeds,
               movies=not args.no_movies, skip_per_model=args.summary_only)
