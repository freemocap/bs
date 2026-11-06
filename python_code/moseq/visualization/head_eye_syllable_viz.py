"""
Head-eye coupling diagnostics for a fitted joint AR-HMM
=======================================================

For a model from `run_head_eye_arhmm.py`, what does each syllable do across
all 9 channels, and do syllables couple head and eye movement?

`plot_syllable_fingerprints`
    Heatmap of each syllable's mean onset-aligned change per channel.

`plot_coupling_scatter`
    One dot per syllable: mean head change vs. mean eye change, per axis.
    Compensatory syllables fall on the negative diagonal for yaw.

`session_coupling_table`
    Lagged head-eye correlation per session and axis, from the features alone,
    so session-to-session differences aren't hidden in pooled numbers.

Channels are compared in normalized units (see `head_eye_features.py`), and
`run_head_eye_diagnostics` also reports eta^2 per channel using
`eye_syllable_viz._syllable_variance_explained`.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.head_eye_syllable_viz \\
        /mnt/data/moseq/head_eye_arhmm_405_407_pilot kappa_1e+05_nlags8_seed0
"""

import argparse
from pathlib import Path

import keypoint_moseq as kpms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from python_code.moseq.head_eye_features import CHANNELS, HeadEyeFeatures, load_head_eye_features
from python_code.moseq.run_multi_session_kappa_sweeps import KEEP_FRAMES
from python_code.moseq.training_config import load_training_recording_folders
from python_code.moseq.visualization.eye_syllable_viz import (
    _overall_variance_explained,
    _syllable_variance_explained,
    _valid_instances,
)

FPS = 120
_MIN_DURATION = 3
_MIN_FREQUENCY = 0.005
_MIN_INSTANCES = 24
_PRE, _POST = 10, 30

_SAVGOL_WINDOW = 11
_MAX_LAG_FRAMES = 12

_ch = CHANNELS.index
_HORIZONTAL_GAZE = "horizontal_gaze"
_VERTICAL_GAZE = "vertical_gaze"
_VERTICAL_DIFFERENCE = "vertical_difference"


def _derived_gaze(features: NDArray) -> dict[str, NDArray]:
    """
    Whole-eye signals per axis, from (..., 9) features. Adduction is flipped
    per eye (`analyzable_output_csvs.md`), so right minus left adduction is
    conjugate gaze toward the animal's left, the same sign as head yaw. Lateral
    eyes make roll a vertical movement of opposite sign in each eye, so it's
    paired with the left-right elevation difference.
    """
    return {
        _HORIZONTAL_GAZE: (features[..., _ch("right_adduction")] - features[..., _ch("left_adduction")]) / 2,
        _VERTICAL_GAZE: (features[..., _ch("left_elevation")] + features[..., _ch("right_elevation")]) / 2,
        _VERTICAL_DIFFERENCE: (features[..., _ch("left_elevation")] - features[..., _ch("right_elevation")]) / 2,
    }


def _masked_trajectories(features: HeadEyeFeatures) -> dict[str, NDArray]:
    """Normalized features with invalid frames set to NaN, so instances that
    touch them drop out in `_valid_instances`."""
    return {name: np.where(features.valid[name], x, np.nan) for name, x in features.features.items()}


def _syllable_trajectories(
    trajectories: dict[str, NDArray], results: dict
) -> dict[int, NDArray]:
    """`{syllable: (n_instances, _PRE + _POST, 9)}` onset-aligned, valid only."""
    instances = kpms.get_syllable_instances(
        {name: r["syllable"] for name, r in results.items()},
        pre=_PRE,
        post=_POST,
        min_duration=_MIN_DURATION,
        min_frequency=_MIN_FREQUENCY,
        min_instances=_MIN_INSTANCES,
    )
    by_syllable = {}
    for syllable, syllable_instances in instances.items():
        valid = _valid_instances(
            kpms.get_instance_trajectories(syllable_instances, trajectories, pre=_PRE, post=_POST)
        )
        if len(valid):
            by_syllable[syllable] = valid - valid[:, _PRE : _PRE + 1]
    return by_syllable


def plot_syllable_fingerprints(by_syllable: dict[int, NDArray], output_path: Path) -> None:
    """Mean change from onset, per syllable (rows) and channel (columns)."""
    syllables = sorted(by_syllable)
    fingerprints = np.stack([by_syllable[s].mean(axis=0)[-1] for s in syllables])
    limit = np.nanpercentile(np.abs(fingerprints), 98)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(syllables))))
    image = ax.imshow(fingerprints, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(CHANNELS)), CHANNELS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(syllables)), syllables)
    ax.set_ylabel("syllable")
    ax.set_title(f"Mean change over the {(_POST - 1) / FPS * 1000:.0f} ms after onset")
    fig.colorbar(image, ax=ax, label="normalized units")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_coupling_scatter(by_syllable: dict[int, NDArray], output_path: Path) -> pd.DataFrame:
    """
    One dot per syllable: mean head change vs. mean gaze change after onset.
    Yaw rate is integrated first so both axes are displacements. Returns the
    plotted values.
    """
    rows = []
    for syllable, trajs in by_syllable.items():
        gaze = _derived_gaze(trajs.mean(axis=0)[-1])
        rows.append(
            {
                "syllable": syllable,
                "n_instances": len(trajs),
                "head_yaw": trajs[:, _PRE:, _ch("head_yaw_rate")].mean(axis=0).sum(),
                "head_pitch": trajs[:, -1, _ch("head_pitch")].mean(),
                "head_roll": trajs[:, -1, _ch("head_roll")].mean(),
                **gaze,
            }
        )
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (head, eye) in zip(
        axes,
        [
            ("head_yaw", _HORIZONTAL_GAZE),
            ("head_pitch", _VERTICAL_GAZE),
            ("head_roll", _VERTICAL_DIFFERENCE),
        ],
    ):
        ax.scatter(table[head], table[eye], s=8 * np.sqrt(table["n_instances"]), alpha=0.6, color="tab:blue")
        for _, row in table.iterrows():
            ax.annotate(int(row["syllable"]), (row[head], row[eye]), fontsize=7)
        ax.axhline(0, color="0.85", zorder=0)
        ax.axvline(0, color="0.85", zorder=0)
        r = np.corrcoef(table[head], table[eye])[0, 1] if len(table) > 2 else np.nan
        ax.set_title(f"r across syllables = {r:.2f}")
        ax.set_xlabel(f"{head} change (normalized)")
        ax.set_ylabel(f"{eye} change (normalized)")
    fig.suptitle("Per-syllable head vs. eye movement (dot area ~ instances)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return table


def _peak_lagged_correlation(eye: NDArray, head: NDArray, keep: NDArray) -> tuple[float, float]:
    """(lag ms, r) of the largest-magnitude correlation; positive lag means
    the eye follows the head."""
    best_lag, best_r = 0, 0.0
    for lag in range(-_MAX_LAG_FRAMES, _MAX_LAG_FRAMES + 1):
        shifted_eye = np.roll(eye, -lag)
        shifted_keep = keep & np.roll(keep, -lag)
        r = np.corrcoef(shifted_eye[shifted_keep], head[shifted_keep])[0, 1]
        if abs(r) > abs(best_r):
            best_lag, best_r = lag, float(r)
    return best_lag / FPS * 1000, best_r


def session_coupling_table(features: HeadEyeFeatures) -> pd.DataFrame:
    """Peak lagged correlation between head and gaze velocity, per session."""
    rows = []
    for name, x in features.features.items():
        velocity = savgol_filter(x, _SAVGOL_WINDOW, 3, deriv=1, delta=1 / FPS, axis=0)
        gaze = _derived_gaze(velocity)
        keep = (
            np.convolve(~features.valid[name].all(axis=1), np.ones(_SAVGOL_WINDOW), mode="same") == 0
        )
        row: dict = {"recording": name}
        for label, head, eye in [
            ("yaw", savgol_filter(x[:, _ch("head_yaw_rate")], _SAVGOL_WINDOW, 3), gaze[_HORIZONTAL_GAZE]),
            ("pitch", velocity[:, _ch("head_pitch")], gaze[_VERTICAL_GAZE]),
            ("roll", velocity[:, _ch("head_roll")], gaze[_VERTICAL_DIFFERENCE]),
        ]:
            row[f"{label}_lag_ms"], row[f"{label}_r"] = _peak_lagged_correlation(eye, head, keep)
        rows.append(row)
    return pd.DataFrame(rows)


def channel_eta_squared(by_syllable: dict[int, NDArray]) -> pd.Series:
    """Overall eta^2 per channel: how much of its onset-aligned movement
    syllable identity explains."""
    eta = {}
    for i, channel in enumerate(CHANNELS):
        per_syllable = _syllable_variance_explained(
            {s: trajs[:, :, [i]] for s, trajs in by_syllable.items()}, pre=_PRE
        )
        stats = {
            s: {f"{channel}_variance_explained": ve, f"{channel}_ss_total": ss}
            for s, (ve, ss) in per_syllable.items()
        }
        eta[channel] = _overall_variance_explained(stats, channel)
    return pd.Series(eta, name="eta_squared")


def run_head_eye_diagnostics(project_dir: str | Path, model_name: str) -> dict:
    """Save all of this module's plots and tables to
    `{project_dir}/{model_name}/head_eye_plots`."""
    output_dir = Path(project_dir) / model_name / "head_eye_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_head_eye_features(
        load_training_recording_folders(project_dir), keep_frames=KEEP_FRAMES
    )
    results = kpms.load_results(str(project_dir), model_name)
    by_syllable = _syllable_trajectories(_masked_trajectories(features), results)

    plot_syllable_fingerprints(by_syllable, output_dir / "syllable_fingerprints.png")
    coupling = plot_coupling_scatter(by_syllable, output_dir / "coupling_scatter.png")
    coupling.to_csv(output_dir / "coupling_by_syllable.csv", index=False)
    eta = channel_eta_squared(by_syllable)
    eta.to_csv(output_dir / "channel_eta_squared.csv")
    sessions = session_coupling_table(features)
    sessions.to_csv(output_dir / "coupling_by_session.csv", index=False)
    return {"coupling": coupling, "eta_squared": eta, "sessions": sessions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir")
    parser.add_argument("model_name")
    args = parser.parse_args()

    outputs = run_head_eye_diagnostics(args.project_dir, args.model_name)
    print(outputs["eta_squared"].to_string())
    print(outputs["sessions"].to_string(index=False))
