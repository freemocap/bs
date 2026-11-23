"""
Presentation version of the head/eye velocity correlograms
==========================================================

Minimal-text figure for slides: requested syllables vs. all other syllables.

- Left: mean cross-correlogram of head yaw velocity vs. conjugate horizontal
  eye velocity (Fisher-z mean, 95% bootstrap band), r(lag) =
  corr(head[t + lag], eye[t]), so a peak at lag > 0 means the eye's velocity
  leads the head's. Same qualifying gaze-shift instances and fixed windows as
  `head_eye_timing_stats` (any syllable, head and same-direction eye
  velocity >= 60 deg/s).
- Right: median per-instance peak lag (`xcorr_lag_ms`) with a hierarchical
  (animal > session > instance) bootstrap 95% CI.

This shows when velocity PEAKS, not when movement starts (see
`head_eye_onset_latency` for why onset can't be measured from the 6 Hz
filtered eye data).

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.head_eye_xcorr_presentation \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil kappa_1e+03_nlags8_seed0 9 14
"""

import argparse
from pathlib import Path

import keypoint_moseq as kpms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python_code.moseq.training_config import load_training_recording_folders
from python_code.moseq.visualization.head_eye_lag_xcorr import FPS, fisher_mean_ci, lagged_corr, load_velocities
from python_code.moseq.visualization.head_eye_onset_latency import _hierarchical_resamples
from python_code.moseq.visualization.head_eye_timing_stats import (
    _HIGHLIGHT_COLORS,
    _INK,
    MAX_LAG,
    _style,
    _subframe_peak,
    gaze_shift_windows,
    instance_timing,
)

_OTHER_COLOR = "#8a8984"
XLIM_MS = 150  # correlogram x range shown (computed out to MAX_LAG frames)


def run(project_dir: Path, model_name: str, syllables: list[int]) -> None:
    output_dir = project_dir / model_name / "eye_syllable_plots" / "head_eye_xcorr"
    output_dir.mkdir(parents=True, exist_ok=True)
    velocities = load_velocities(load_training_recording_folders(project_dir))
    results = kpms.load_results(str(project_dir), model_name)

    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    lags_ms = lags / FPS * 1000
    windows = list(gaze_shift_windows(results, velocities))
    correlograms = np.array([lagged_corr(w["head"], w["eye"], lags, min_pairs=20) for w in windows])
    labels = np.array([w["syllable"] for w in windows])
    timing = instance_timing(results, velocities).dropna(subset=["xcorr_lag_ms"]).reset_index(drop=True)

    groups = [(f"syllable {s}", color, lambda x, s=s: x == s) for s, color in zip(syllables, _HIGHLIGHT_COLORS)]
    groups.append(("other syllables", _OTHER_COLOR, lambda x: ~np.isin(x, syllables)))

    plt.rcParams.update({"font.size": 14})
    fig, (ax, ax_lag) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[2.2, 1]))

    # Correlograms, other syllables drawn first so the highlighted ones sit on top.
    for name, color, select in reversed(groups):
        mean, lo, hi = fisher_mean_ci(correlograms[select(labels)])
        ax.fill_between(lags_ms, lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.plot(lags_ms, mean, color=color, linewidth=2.5 if name != "other syllables" else 2, label=name)
        k = int(np.nanargmax(mean))
        peak_ms = (lags[0] + _subframe_peak(mean, k)) / FPS * 1000
        ax.plot(peak_ms, mean[k], "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    ax.axvline(0, color=_INK, linewidth=0.8)
    ax.axhline(0, color=_INK, linewidth=0.8)
    ax.set_xlim(-XLIM_MS, XLIM_MS)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom - 0.12 * (top - bottom), top)  # room for the lead labels under the curves
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("correlation (r)")
    ax.text(0.98, 0.04, "eye leads →", transform=ax.transAxes, ha="right", color=_INK)
    ax.text(0.02, 0.04, "← head leads", transform=ax.transAxes, ha="left", color=_INK)
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], names[::-1], frameon=False, loc="upper left")
    ax.set_title("Head vs. eye velocity", loc="left")
    _style(ax)

    # Per-instance peak lag: median(syllable) − median(other syllables), with a
    # hierarchical bootstrap CI of that difference. Plotting the difference
    # directly avoids reading overlapping per-group CIs as "no difference".
    resamples = _hierarchical_resamples(timing[["session", "animal"]])
    values = timing["xcorr_lag_ms"].to_numpy()
    syl = timing["syllable"].to_numpy()
    other = groups[-1][2]
    for x, (name, color, select) in enumerate(groups[:-1]):
        diff = np.median(values[select(syl)]) - np.median(values[other(syl)])
        boots = [np.median(values[i][select(syl[i])]) - np.median(values[i][other(syl[i])]) for i in resamples]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ax_lag.errorbar(x, diff, yerr=[[diff - lo], [hi - diff]], fmt="o", color=color, markersize=11,
                        capsize=5, linewidth=2)
        ax_lag.text(x + 0.12, diff, f"+{diff:.0f} ms", va="center", color=_INK)
    ax_lag.axhline(0, color=_OTHER_COLOR, linewidth=2)
    ax_lag.text(len(groups) - 1.6, 0.5, f"median of other syllables ({np.median(values[other(syl)]):.0f} ms)",
                color=_OTHER_COLOR, va="bottom", ha="right", fontsize=12)
    ax_lag.set_xticks(range(len(groups) - 1), [g[0] for g in groups[:-1]])
    ax_lag.set_xlim(-0.5, len(groups) - 1.4)
    ax_lag.set_ylim(-5, None)
    ax_lag.set_ylabel("extra eye lead at peak (ms)")
    ax_lag.set_title("vs. other syllables (95% CI)", loc="left")
    _style(ax_lag)

    fig.tight_layout()
    path = output_dir / f"xcorr_presentation_{'_'.join(map(str, syllables))}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("syllables", type=int, nargs="+")
    args = parser.parse_args()
    run(args.project_dir, args.model_name, args.syllables)
