"""
Does the eye or the head start moving first in a gaze shift?
============================================================

`head_eye_lag_xcorr` and `head_eye_timing_stats` show that the eye's velocity
PEAKS ~75-95 ms before the head's. A peak can come first just because the
eye accelerates faster, so this measures movement ONSET.

Instances are those of `head_eye_timing_stats.gaze_shift_windows` (any
syllable; head yaw and same-direction conjugate eye velocity both reach
`MIN_PEAK_DEG_S`; each window sign-flipped so the head turn is +), restricted
to ones that start from rest: head and eye velocity both below `still_deg_s`
for the `QUIET` frames (150 ms) before syllable onset. Without that filter
many instances start partway through a movement that began in the previous
syllable (e.g. syllable 13 in the seed-0 model), which corrupts onset
estimates.

Two estimates of head onset − eye onset (positive = eye first), both with
hierarchical bootstrap CIs (animals, then sessions, then instances):

- Mean-trace onset: average the velocity windows, then find when each mean
  trace last rises past `threshold` deg/s above its pre-onset baseline before
  its peak, linearly interpolated between frames. Averaging removes most noise,
  so low thresholds are usable; a threshold of a few deg/s is the closest this
  gets to "when did movement begin".
- Per-instance onset: the same backward walk from each instance's own peak,
  for several criteria (fraction of peak, absolute deg/s, baseline + k SD),
  then the median across instances. Noisier, so it needs higher thresholds.

If the lead shrinks toward 0 as the threshold drops, the two start together
and the eye simply accelerates faster.

Filtering caveat (important): `eye_in_head` is built from the `cleaned` eye
trajectories, which are 4th-order Butterworth low-passed at 6 Hz with
`filtfilt` (`eye_analysis/data_models/trajectory_dataset.py`). That zero-phase
filter spreads a ~30-50 ms saccade over roughly +-80 ms, so the eye's
filtered velocity starts rising ~60 ms BEFORE the true saccade onset, with a
small opposite-direction dip just before (ringing). Peak TIMES survive a
zero-phase filter; onsets do not. In simulation, a saccade and a head turn
that truly start together come out of this filter with an apparent eye lead of
~75-110 ms at these thresholds, so the observed lead of ~3-24 ms is
not evidence that the eye starts first; if anything, it suggests the saccade
starts after the head. A real onset test needs eye angles from the `raw` (or
much more lightly filtered) eye trajectories.

Sync caveat: any timing offset between the eye cameras and the mocap cameras
shifts every number here one-for-one. See `head_eye_lag_xcorr` for why that
is worth checking independently.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.head_eye_onset_latency \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil kappa_1e+03_nlags8_seed0 [--still 30]
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

from python_code.moseq.training_config import load_training_recording_folders
from python_code.moseq.visualization.head_eye_lag_xcorr import FPS, load_velocities
from python_code.moseq.visualization.head_eye_timing_stats import (
    MIN_PEAK_DEG_S,
    PRE,
    QUIET,
    _INK,
    _style,
    gaze_shift_windows,
)

N_BOOT = 1000
MEAN_TRACE_THRESHOLDS = np.arange(1, 21)  # deg/s above baseline
# Per-instance criteria: name -> function(trace, peak index) -> threshold.
INSTANCE_CRITERIA = {
    "5% of peak": lambda x, pk: 0.05 * x[pk],
    "10% of peak": lambda x, pk: 0.10 * x[pk],
    "20% of peak": lambda x, pk: 0.20 * x[pk],
    "10 deg/s": lambda x, pk: 10.0,
    "20 deg/s": lambda x, pk: 20.0,
    "30 deg/s": lambda x, pk: 30.0,
    "baseline + 3 SD": lambda x, pk: x[PRE - QUIET : PRE].mean() + 3 * x[PRE - QUIET : PRE].std(),
    "baseline + 5 SD": lambda x, pk: x[PRE - QUIET : PRE].mean() + 5 * x[PRE - QUIET : PRE].std(),
}
# Reference palette categorical slots 1 and 2.
_HEAD_COLOR = "#2a78d6"
_EYE_COLOR = "#eb6834"


def _ms(frames: float) -> float:
    return frames / FPS * 1000


def onset_frame(x: NDArray, peak: int, threshold: float) -> float | None:
    """
    Walk back from `peak` while `x` stays above `threshold`; return the
    interpolated (fractional) frame where it crosses. None if x never exceeds
    the threshold or is still above it at the start of the window.
    """
    if x[peak] <= threshold:
        return None
    j = peak
    while j > 0 and x[j - 1] > threshold:
        j -= 1
    if j == 0:
        return None
    below, above = x[j - 1], x[j]
    return j - 1 + (threshold - below) / (above - below)


def _hierarchical_resamples(meta: pd.DataFrame, seed: int = 0) -> list[NDArray]:
    """`N_BOOT` row-index arrays, resampling animals, sessions, then instances."""
    rng = np.random.default_rng(seed)
    by_animal = {
        a: [g.index.to_numpy() for _, g in ga.groupby("session")] for a, ga in meta.groupby("animal")
    }
    animals = list(by_animal)
    resamples = []
    for _ in range(N_BOOT):
        picks = []
        for a in rng.choice(animals, len(animals)):
            sessions = by_animal[a]
            for j in rng.integers(0, len(sessions), len(sessions)):
                picks.append(rng.choice(sessions[j], len(sessions[j])))
        resamples.append(np.concatenate(picks))
    return resamples


def mean_trace_onset_gap(head: NDArray, eye: NDArray, threshold: float) -> float:
    """Head − eye onset (ms) of the mean traces, `threshold` deg/s above baseline."""
    onsets = []
    for traces in (head, eye):
        m = traces.mean(axis=0)
        m = m - m[PRE - QUIET : PRE].mean()
        onset = onset_frame(m, int(np.argmax(m)), threshold)
        onsets.append(np.nan if onset is None else onset)
    return _ms(onsets[0] - onsets[1])


def instance_onset_gaps(head: NDArray, eye: NDArray) -> pd.DataFrame:
    """(n_instances, criteria) head − eye onset in ms, NaN where undefined."""
    rows = []
    for h, e in zip(head, eye):
        hp, ep = int(np.argmax(h)), int(np.argmax(e))
        row = {}
        for name, criterion in INSTANCE_CRITERIA.items():
            oh, oe = onset_frame(h, hp, criterion(h, hp)), onset_frame(e, ep, criterion(e, ep))
            row[name] = np.nan if oh is None or oe is None else _ms(oh - oe)
        rows.append(row)
    return pd.DataFrame(rows)


def run(project_dir: Path, model_name: str, still_deg_s: float = 30.0) -> None:
    output_dir = project_dir / model_name / "eye_syllable_plots" / "head_eye_xcorr"
    output_dir.mkdir(parents=True, exist_ok=True)

    velocities = load_velocities(load_training_recording_folders(project_dir))
    results = kpms.load_results(str(project_dir), model_name)
    windows = [
        w for w in gaze_shift_windows(results, velocities)
        if w["head_quiet_deg_s"] < still_deg_s and w["eye_quiet_deg_s"] < still_deg_s
    ]
    meta = pd.DataFrame([{k: w[k] for k in ("session", "animal", "syllable")} for w in windows])
    head = np.array([w["head"] for w in windows])
    eye = np.array([w["eye"] for w in windows])
    resamples = _hierarchical_resamples(meta)

    lines = [
        f"{len(windows)} gaze shifts starting from rest (head and eye < {still_deg_s:g} deg/s for "
        f"{_ms(QUIET):.0f} ms before syllable onset; peaks >= {MIN_PEAK_DEG_S:.0f} deg/s), "
        f"{meta['session'].nunique()} sessions, {meta['animal'].nunique()} animals",
        "head onset − eye onset, ms (+ = eye first); 95% CIs from an animal > session > instance bootstrap",
        "",
        "Mean-trace onset (threshold above pre-onset baseline):",
    ]
    curve = {"gap": [], "lo": [], "hi": []}
    for threshold in MEAN_TRACE_THRESHOLDS:
        gap = mean_trace_onset_gap(head, eye, threshold)
        boots = [mean_trace_onset_gap(head[i], eye[i], threshold) for i in resamples]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        curve["gap"].append(gap), curve["lo"].append(lo), curve["hi"].append(hi)
        lines.append(f"  {threshold:4.0f} deg/s: {gap:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")

    gaps = instance_onset_gaps(head, eye)
    lines += ["", "Per-instance onset (median across instances):"]
    instance_summary = {}
    for name in INSTANCE_CRITERIA:
        values = gaps[name]
        ok = values.notna().to_numpy()
        boots = [np.nanmedian(values.to_numpy()[i]) for i in resamples]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        session_eye_first = values.groupby(meta["session"]).median() > 0
        instance_summary[name] = (values.median(), lo, hi)
        lines.append(
            f"  {name:16s} n={ok.sum():5d}  median {values.median():+5.1f} [{lo:+5.1f}, {hi:+5.1f}]  "
            f"eye first {np.mean(values[ok] > 0):.0%}, same frame {np.mean(values[ok] == 0):.0%}, "
            f"head first {np.mean(values[ok] < 0):.0%}; session medians eye-first in "
            f"{session_eye_first.mean():.0%} of {len(session_eye_first)}"
        )
    text = "\n".join(lines)
    suffix = f"_still{still_deg_s:g}"
    (output_dir / f"onset_latency{suffix}.txt").write_text(text + "\n")
    print(text)

    _plot(head, eye, resamples, curve, instance_summary, len(windows), still_deg_s,
          output_dir / f"onset_latency{suffix}.png")


def _mean_ci(traces: NDArray, resamples: list[NDArray]) -> tuple[NDArray, NDArray, NDArray]:
    boots = np.array([traces[i].mean(axis=0) for i in resamples])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return traces.mean(axis=0), lo, hi


def _plot(head, eye, resamples, curve, instance_summary, n, still_deg_s, path: Path) -> None:
    t_ms = _ms(np.arange(head.shape[1]) - PRE)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1.3, 1, 1.15]))
    traces = {"head yaw": (head, _HEAD_COLOR), "eye (both, horizontal)": (eye, _EYE_COLOR)}

    # A: full mean traces. B: baseline-subtracted, zoomed on onset.
    for ax, zoom in ((axes[0], False), (axes[1], True)):
        for label, (x, color) in traces.items():
            mean, lo, hi = _mean_ci(x, resamples)
            if zoom:
                base = mean[PRE - QUIET : PRE].mean()
                mean, lo, hi = mean - base, lo - base, hi - base
            ax.fill_between(t_ms, lo, hi, color=color, alpha=0.25, linewidth=0)
            ax.plot(t_ms, mean, color=color, linewidth=2)
            peak = int(np.argmax(mean))
            if not zoom:
                ax.annotate(f"{label}\npeak {mean[peak]:.0f} deg/s at {t_ms[peak]:+.0f} ms",
                            (t_ms[peak], mean[peak]), xytext=(8, 4), textcoords="offset points",
                            fontsize=9, color=_INK)
        ax.axvline(0, color=_INK, linewidth=0.8)
        ax.axhline(0, color=_INK, linewidth=0.8)
        ax.set_xlabel("time from syllable onset (ms)")
        _style(ax)
    axes[0].axvspan(-_ms(QUIET), 0, color="#e6e5e0", zorder=0)
    axes[0].text(-_ms(QUIET) / 2, axes[0].get_ylim()[1] * 0.95, f"still\n(< {still_deg_s:g} deg/s)",
                 ha="center", va="top", fontsize=8, color=_INK)
    axes[0].set_ylabel("velocity toward the head turn (deg/s)")
    axes[0].set_title(f"A. Mean velocity, {n} gaze shifts starting from rest (95% CI)", loc="left")
    axes[1].set_xlim(-100, 100)
    axes[1].set_ylim(-3, 25)
    axes[1].set_ylabel("velocity minus pre-onset baseline (deg/s)")
    axes[1].set_title("B. Zoomed on onset, baseline subtracted", loc="left")
    for label, (_, color) in traces.items():
        axes[1].plot([], [], color=color, linewidth=2, label=label)
    axes[1].legend(fontsize=9, loc="upper left", frameon=False)

    # C: onset gap vs. threshold.
    ax = axes[2]
    ax.fill_between(MEAN_TRACE_THRESHOLDS, curve["lo"], curve["hi"], color=_INK, alpha=0.15, linewidth=0)
    ax.plot(MEAN_TRACE_THRESHOLDS, curve["gap"], color=_INK, linewidth=2, label="mean-trace onset (95% CI)")
    for name, x in (("10 deg/s", 10), ("20 deg/s", 20)):
        med, lo, hi = instance_summary[name]
        ax.errorbar(x + 0.4, med, yerr=[[med - lo], [hi - med]], fmt="o", color=_INK, mfc="white", markersize=7,
                    capsize=3, label="per-instance median, same threshold" if x == 10 else None)
    ax.axhline(0, color=_INK, linewidth=0.8)
    ax.text(MEAN_TRACE_THRESHOLDS[-1], 0.6, "eye first ↑", ha="right", va="bottom", fontsize=9, color=_INK)
    ax.text(MEAN_TRACE_THRESHOLDS[-1], -0.6, "head first ↓", ha="right", va="top", fontsize=9, color=_INK)
    ax.set_xlabel("onset threshold (deg/s above baseline)")
    ax.set_ylabel("head onset − eye onset (ms)")
    ax.set_title("C. Apparent eye lead vs. onset threshold", loc="left")
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    _style(ax)

    fig.suptitle(
        "Do the eye and head start moving at the same time? Gaze shifts starting from rest, all syllables",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("--still", type=float, default=30.0, metavar="DEG_S",
                        help="max head and eye velocity in the 150 ms before onset (default 30)")
    args = parser.parse_args()
    run(args.project_dir, args.model_name, still_deg_s=args.still)
