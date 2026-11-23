"""
Is the eye-to-head lag different in particular syllables?
=========================================================

Companion to `head_eye_lag_xcorr`. That module shows that in every syllable,
the eye's velocity peaks before the head's in a gaze shift. This one tests
whether the lag is longer in the requested syllables than in the rest.

Per instance (same signals and signs as `head_eye_lag_xcorr`):

- Window: a FIXED `PRE` frames before to `POST` frames after onset for every
  syllable. Windows that run to the syllable's offset would give long
  syllables more room for a large peak separation.
- Each instance is sign-flipped so its largest head yaw velocity is positive.
  It qualifies only if both head yaw velocity and same-direction conjugate eye
  velocity reach `MIN_PEAK_DEG_S` at an interior frame, with no NaNs.
- `peak_gap_ms`: time of head-velocity peak minus time of eye-velocity peak,
  parabolically interpolated to sub-frame precision. Positive = eye first.
- `xcorr_lag_ms`: lag of the per-instance cross-correlogram's positive peak
  (r(lag) = corr(head[t + lag], eye[t])), also interpolated. Instances whose
  peak is at the edge of the lag range are dropped from this measure.

Statistics for each requested syllable vs. all other syllables, from least to
most conservative about clustering within sessions and animals:

- Instance-level Mann-Whitney (treats every instance as independent; shown
  for reference only).
- Session-paired Wilcoxon: each session's median for the syllable vs. its
  median for every other syllable (sessions with >= `MIN_SESSION_INSTANCES`).
- Hierarchical bootstrap of the difference in medians, resampling animals,
  then sessions within animal, then instances within session.
- Linear mixed model, random intercept per session, with and without log
  head and eye peak velocity and syllable duration as covariates (does the
  difference survive controlling for movement size?).

Caveat: the model groups frames by their head and eye dynamics, so syllables
are expected to differ in timing to some degree. A difference shows that the
syllable has a longer lag, not that it is a separate kind of movement.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.head_eye_timing_stats \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil kappa_1e+03_nlags8_seed0 9 14
"""

import argparse
import re
from pathlib import Path

import keypoint_moseq as kpms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.typing import NDArray
from scipy import stats

from python_code.moseq.training_config import load_training_recording_folders
from python_code.moseq.visualization.head_eye_lag_xcorr import FPS, lagged_corr, load_velocities

PRE = 30  # frames before onset (250 ms)
POST = 45  # frames after onset (375 ms)
QUIET = 18  # frames before onset (150 ms) checked by the `still_deg_s` filter
MAX_LAG = 30  # frames, for `xcorr_lag_ms`
MIN_PEAK_DEG_S = 60.0
MIN_SESSION_INSTANCES = 10
MIN_SYLLABLE_INSTANCES = 100  # for a syllable to appear in the ranking figure
MIN_SYLLABLE_INSTANCES_STILL = 30  # the same, with the `still_deg_s` filter
N_BOOT = 2000

METRICS = {
    "peak_gap_ms": "head peak − eye peak (ms)",
    "xcorr_lag_ms": "cross-correlation peak lag (ms)",
}
# Reference palette categorical slots 1-3 (validated as all-pairs distinct,
# including for colorblind viewers); other syllables are neutral.
_HIGHLIGHT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
_NEUTRAL = "#a8a7a2"
_INK = "#52514e"


def _subframe_peak(x: NDArray, i: int) -> float:
    """Parabolic interpolation of a maximum at index i."""
    if 0 < i < len(x) - 1:
        a, b, c = x[i - 1], x[i], x[i + 1]
        curvature = a - 2 * b + c
        if curvature < 0:
            return i + 0.5 * (a - c) / curvature
    return float(i)


def gaze_shift_windows(results: dict, velocities: dict):
    """
    Yield one dict per qualifying syllable instance (see the module docstring):
    `head`/`eye` are the (PRE + POST,) sign-flipped velocity windows, onset at
    index PRE, plus session/animal/syllable, peak indices, duration, and
    `head_quiet_deg_s`/`eye_quiet_deg_s`, the largest absolute velocity in
    the `QUIET` frames before onset.
    """
    for name, res in results.items():
        syllables = res["syllable"]
        v = velocities[name]
        n = len(syllables)
        if n != len(v["head"]):
            raise ValueError(f"'{name}': syllable and kinematics frame counts differ")
        animal = re.search(r"ferret_(\d+)", name).group(1)
        onsets = np.r_[0, np.flatnonzero(np.diff(syllables)) + 1]
        offsets = np.r_[onsets[1:], n]
        for on, off in zip(onsets, offsets):
            if on - PRE < 0 or on + POST > n:
                continue
            head = v["head"][on - PRE : on + POST]
            eye = v["both eyes"][on - PRE : on + POST]
            if np.isnan(head).any() or np.isnan(eye).any():
                continue
            sign = np.sign(head[np.argmax(np.abs(head))])
            head, eye = head * sign, eye * sign
            head_i, eye_i = int(np.argmax(head)), int(np.argmax(eye))
            if head[head_i] < MIN_PEAK_DEG_S or eye[eye_i] < MIN_PEAK_DEG_S:
                continue
            if head_i in (0, len(head) - 1) or eye_i in (0, len(eye) - 1):
                continue
            yield dict(
                session=name,
                animal=animal,
                syllable=int(syllables[on]),
                head=head,
                eye=eye,
                head_i=head_i,
                eye_i=eye_i,
                duration_ms=(off - on) / FPS * 1000,
                head_quiet_deg_s=float(np.abs(head[PRE - QUIET : PRE]).max()),
                eye_quiet_deg_s=float(np.abs(eye[PRE - QUIET : PRE]).max()),
            )


def instance_timing(results: dict, velocities: dict) -> pd.DataFrame:
    """One row per qualifying syllable instance in every session."""
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    rows = []
    for w in gaze_shift_windows(results, velocities):
        head, eye, head_i, eye_i = w["head"], w["eye"], w["head_i"], w["eye_i"]
        r = lagged_corr(head, eye, lags, min_pairs=20)
        k = int(np.nanargmax(r))
        rows.append(
            dict(
                session=w["session"],
                animal=w["animal"],
                syllable=w["syllable"],
                peak_gap_ms=(_subframe_peak(head, head_i) - _subframe_peak(eye, eye_i)) / FPS * 1000,
                xcorr_lag_ms=(lags[0] + _subframe_peak(r, k)) / FPS * 1000 if 0 < k < len(lags) - 1 else np.nan,
                head_peak=head[head_i],
                eye_peak=eye[eye_i],
                duration_ms=w["duration_ms"],
                head_quiet_deg_s=w["head_quiet_deg_s"],
                eye_quiet_deg_s=w["eye_quiet_deg_s"],
            )
        )
    return pd.DataFrame(rows)


def hierarchical_bootstrap(df: pd.DataFrame, syllable: int, metric: str, seed: int = 0) -> NDArray:
    """2.5, 50, 97.5 percentiles of median(syllable) − median(other), resampling
    animals, then sessions within animal, then instances within session."""
    rng = np.random.default_rng(seed)
    by_animal = {
        a: [(g[metric].to_numpy(), (g["syllable"] == syllable).to_numpy()) for _, g in ga.groupby("session")]
        for a, ga in df.groupby("animal")
    }
    animals = list(by_animal)
    diffs = []
    for _ in range(N_BOOT):
        target, other = [], []
        for a in rng.choice(animals, len(animals)):
            sessions = by_animal[a]
            for j in rng.integers(0, len(sessions), len(sessions)):
                values, is_target = sessions[j]
                if is_target.any():
                    target.append(rng.choice(values[is_target], is_target.sum()))
                other.append(rng.choice(values[~is_target], (~is_target).sum()))
        if target:
            diffs.append(np.median(np.concatenate(target)) - np.median(np.concatenate(other)))
    return np.percentile(diffs, [2.5, 50, 97.5])


def session_medians(df: pd.DataFrame, syllable: int, metric: str) -> pd.DataFrame:
    """Per-session median for the syllable ("target") and for all others ("other")."""
    rows = []
    for session, g in df.groupby("session"):
        is_target = g["syllable"] == syllable
        if is_target.sum() >= MIN_SESSION_INSTANCES:
            rows.append(
                dict(session=session, target=g.loc[is_target, metric].median(), other=g.loc[~is_target, metric].median())
            )
    return pd.DataFrame(rows, columns=["session", "target", "other"])


def compare(df: pd.DataFrame, syllable: int, metric: str) -> dict:
    d = df.dropna(subset=[metric])
    is_target = d["syllable"] == syllable
    paired = session_medians(d, syllable, metric)
    model_df = d.assign(
        is_target=is_target.astype(int),
        log_head_peak=np.log(d["head_peak"]),
        log_eye_peak=np.log(d["eye_peak"]),
    )
    out = dict(
        syllable=syllable,
        metric=metric,
        n_target=int(is_target.sum()),
        n_other=int((~is_target).sum()),
        median_target=d.loc[is_target, metric].median(),
        median_other=d.loc[~is_target, metric].median(),
        mannwhitney_p=stats.mannwhitneyu(d.loc[is_target, metric], d.loc[~is_target, metric]).pvalue,
        n_sessions=len(paired),
        sessions_higher=float(np.mean(paired["target"] > paired["other"])) if len(paired) else np.nan,
        session_median_diff=float(np.median(paired["target"] - paired["other"])) if len(paired) else np.nan,
        # Too few sessions with enough instances (e.g. with `still_deg_s`) -> NaN.
        wilcoxon_p=stats.wilcoxon(paired["target"] - paired["other"]).pvalue if len(paired) >= 5 else np.nan,
    )
    out["boot_lo"], out["boot_mid"], out["boot_hi"] = hierarchical_bootstrap(d, syllable, metric)
    for label, covariates in [("mixed", ""), ("mixed_adj", " + log_head_peak + log_eye_peak + duration_ms")]:
        fit = smf.mixedlm(f"{metric} ~ is_target{covariates}", model_df, groups=model_df["session"]).fit()
        lo, hi = fit.conf_int().loc["is_target"]
        out[f"{label}_est"], out[f"{label}_lo"], out[f"{label}_hi"] = fit.params["is_target"], lo, hi
        out[f"{label}_p"] = fit.pvalues["is_target"]
    return out


def format_comparison(c: dict) -> str:
    return (
        f"syllable {c['syllable']} | {METRICS[c['metric']]}\n"
        f"  median {c['median_target']:.1f} (n={c['n_target']}) vs other {c['median_other']:.1f} "
        f"(n={c['n_other']}); instance-level Mann-Whitney p={c['mannwhitney_p']:.2g} (ignores clustering)\n"
        f"  session-paired: {c['n_sessions']} sessions, median diff {c['session_median_diff']:+.1f}, "
        f"{c['sessions_higher']:.0%} higher, Wilcoxon p={c['wilcoxon_p']:.2g}\n"
        f"  hierarchical bootstrap diff in medians: {c['boot_mid']:+.1f} [95% CI {c['boot_lo']:+.1f}, {c['boot_hi']:+.1f}]\n"
        f"  mixed model (1|session): {c['mixed_est']:+.1f} [{c['mixed_lo']:+.1f}, {c['mixed_hi']:+.1f}], p={c['mixed_p']:.2g}\n"
        f"  + log peak velocities + duration: {c['mixed_adj_est']:+.1f} "
        f"[{c['mixed_adj_lo']:+.1f}, {c['mixed_adj_hi']:+.1f}], p={c['mixed_adj_p']:.2g}"
    )


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK)
    ax.tick_params(colors=_INK)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot(
    df: pd.DataFrame,
    syllables: list[int],
    comparisons: list[dict],
    path: Path,
    still_deg_s: float | None = None,
    min_syllable_instances: int = MIN_SYLLABLE_INSTANCES,
) -> None:
    colors = dict(zip(syllables, _HIGHLIGHT_COLORS))
    fig, axes = plt.subplots(
        len(METRICS), 2, figsize=(14 + 1.6 * len(syllables), 5.2 * len(METRICS)),
        gridspec_kw=dict(width_ratios=[3.2, 0.5 * len(syllables)]),
    )
    for row, (metric, label) in enumerate(METRICS.items()):
        d = df.dropna(subset=[metric])
        other_median = d.loc[~d["syllable"].isin(syllables), metric].median()

        # Left: every syllable's distribution, sorted by median.
        ax = axes[row, 0]
        counts = d.groupby("syllable")[metric].agg(["median", "size"])
        order = counts[counts["size"] >= min_syllable_instances].sort_values("median", ascending=False).index
        box = ax.boxplot(
            [d.loc[d["syllable"] == s, metric] for s in order],
            positions=np.arange(len(order)), widths=0.6, showfliers=False, patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=_INK, linewidth=1), capprops=dict(color=_INK, linewidth=1),
        )
        for patch, s in zip(box["boxes"], order):
            patch.set(facecolor=colors.get(s, _NEUTRAL), edgecolor="none")
        ax.axhline(other_median, color=_INK, linestyle="--", linewidth=1)
        ax.text(len(order) - 0.4, other_median + 4, f"median of all\nother syllables\n{other_median:.0f} ms",
                va="bottom", ha="left", fontsize=8, color=_INK)
        ax.axhline(0, color=_INK, linewidth=0.8)
        ax.set_xticks(np.arange(len(order)), [str(s) for s in order])
        for tick, s in zip(ax.get_xticklabels(), order):
            if s in colors:
                tick.set_fontweight("bold")
        ax.set_xlim(-0.7, len(order) + 1.2)
        ax.set_xlabel(f"syllable (sorted by median; n ≥ {min_syllable_instances} qualifying instances)")
        ax.set_ylabel(f"{label}\n← head first | eye first →")
        ax.set_title(f"{label} by syllable (boxes: IQR, whiskers: 1.5 IQR, outliers hidden)", loc="left")
        _style(ax)

        # Right: per-session medians, syllable vs. all other syllables.
        ax = axes[row, 1]
        notes = []
        for j, s in enumerate(syllables):
            paired = session_medians(d, s, metric)
            x0, x1 = 3 * j, 3 * j + 1
            for _, p in paired.iterrows():
                ax.plot([x0, x1], [p["other"], p["target"]], color=colors.get(s, _NEUTRAL), alpha=0.45, linewidth=1.2)
            ax.scatter(np.full(len(paired), x0), paired["other"], color=_NEUTRAL, s=36, zorder=3, edgecolor="white")
            ax.scatter(np.full(len(paired), x1), paired["target"], color=colors.get(s, _NEUTRAL), s=36, zorder=3,
                       edgecolor="white")
            c = next(c for c in comparisons if c["syllable"] == s and c["metric"] == metric)
            notes.append(f"syl {s}: {c['sessions_higher']:.0%} of {c['n_sessions']} sessions higher, "
                         f"Wilcoxon p={c['wilcoxon_p']:.1g}\n  bootstrap Δ {c['boot_mid']:+.0f} ms "
                         f"[{c['boot_lo']:+.0f}, {c['boot_hi']:+.0f}]; adj. mixed model {c['mixed_adj_est']:+.0f} ms")
        ax.set_xticks([x for j in range(len(syllables)) for x in (3 * j, 3 * j + 1)],
                      [t for s in syllables for t in ("other", f"syl {s}")])
        ax.set_xlim(-0.6, 3 * len(syllables) - 1.4)
        ax.set_ylabel(f"session median, {label}")
        ax.set_title("per session (one line each)", loc="left")
        ax.text(0, -0.2, "\n".join(notes), transform=ax.transAxes, fontsize=8, color=_INK, va="top")
        _style(ax)

    fig.suptitle(
        "How much earlier does the eye's velocity peak than the head's, per syllable?\n"
        f"Instances with head yaw and same-direction eye velocity ≥ {MIN_PEAK_DEG_S:.0f} deg/s, "
        f"fixed window onset −{PRE / FPS * 1000:.0f} to +{POST / FPS * 1000:.0f} ms"
        + (f"\nOnly instances starting from stillness: head and eye < {still_deg_s:g} deg/s for "
           f"{QUIET / FPS * 1000:.0f} ms before onset" if still_deg_s is not None else ""),
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(path)


def run(project_dir: Path, model_name: str, syllables: list[int], still_deg_s: float | None = None) -> None:
    """
    still_deg_s:
        If given, keep only instances whose head and eye velocities both stay
        below this for the `QUIET` frames before onset, i.e. gaze shifts that
        start from rest rather than partway through a movement that began in
        the previous syllable. The ranking figure's per-syllable minimum drops
        to `MIN_SYLLABLE_INSTANCES_STILL` since far fewer instances qualify.
    """
    if len(syllables) > len(_HIGHLIGHT_COLORS):
        raise ValueError(f"at most {len(_HIGHLIGHT_COLORS)} syllables can be highlighted at once")
    output_dir = project_dir / model_name / "eye_syllable_plots" / "head_eye_xcorr"
    output_dir.mkdir(parents=True, exist_ok=True)

    velocities = load_velocities(load_training_recording_folders(project_dir))
    results = kpms.load_results(str(project_dir), model_name)
    df = instance_timing(results, velocities)
    df.to_csv(output_dir / "instance_timing.csv", index=False)
    suffix, min_instances = "", MIN_SYLLABLE_INSTANCES
    if still_deg_s is not None:
        n_all = len(df)
        df = df[(df["head_quiet_deg_s"] < still_deg_s) & (df["eye_quiet_deg_s"] < still_deg_s)]
        suffix, min_instances = f"_still{still_deg_s:g}", MIN_SYLLABLE_INSTANCES_STILL
    header = (f"{len(df)} qualifying instances, {df['session'].nunique()} sessions, "
              f"{df['animal'].nunique()} animals")
    if still_deg_s is not None:
        header += (f" (of {n_all}; head and eye < {still_deg_s:g} deg/s for "
                   f"{QUIET / FPS * 1000:.0f} ms before onset)")

    comparisons = [compare(df, s, metric) for metric in METRICS for s in syllables]
    text = "\n\n".join([header, *map(format_comparison, comparisons)])
    (output_dir / f"timing_stats{suffix}.txt").write_text(text + "\n")
    print(text)
    plot(
        df, syllables, comparisons,
        output_dir / f"timing_by_syllable_{'_'.join(map(str, syllables))}{suffix}.png",
        still_deg_s=still_deg_s, min_syllable_instances=min_instances,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("syllables", type=int, nargs="+")
    parser.add_argument("--still", type=float, default=None, metavar="DEG_S",
                        help="only instances with head and eye velocity below this before onset")
    args = parser.parse_args()
    run(args.project_dir, args.model_name, args.syllables, still_deg_s=args.still)
