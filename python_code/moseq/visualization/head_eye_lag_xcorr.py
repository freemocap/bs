"""
Does the eye or the head lead within a syllable?
================================================

For each requested syllable, cross-correlates head yaw velocity with
horizontal eye-in-head velocity within every instance of the syllable, over
a range of lags, and averages the per-instance correlograms.

Signals (both deg/s, both positive = toward the animal's LEFT):

- head: `angular_velocity_local` yaw of the skull (+Z up, +Y left, so
  right-hand-rule +yaw is a left turn). Central-differenced by the solver.
- eye: horizontal eye-in-head velocity, `np.gradient` of the horizontal angle
  (also central differences, so neither signal is shifted by half a frame).
  `eye_in_head.adduction` flips sign per eye (see `analyzable_output_csvs.md`),
  so leftward = -adduction for the left eye and +adduction for the right eye.
  "both eyes" is the mean of the two (the conjugate/version component).

Lag convention: r(tau) = corr(head[t + tau], eye[t]). A peak at tau > 0
means the head reproduces what the eye did tau frames earlier, i.e. the EYE
LEADS. tau < 0 means the head leads.

Positive r = eye and head turn the same way (gaze shift); negative r = they
turn opposite ways (compensatory / VOR-like). An eye-leading gaze shift should
show a positive peak at tau > 0.

Each instance's window is its onset - `pad` to its offset + `pad`. Syllables
are short (median ~7 frames at 120 Hz) so most of each window is padding;
the padding is what lets the correlation see a movement that starts before
or ends after the label.

Two references are drawn alongside each syllable:

- other syllables: the same per-instance correlogram for a random sample of
  instances of every other syllable.
- whole session: the correlogram over every valid frame of every session.
  Ferret head rotation is dominated by VOR, so this should be a negative
  trough at or just after tau = 0 (the eye follows the head by a VOR latency).
  A trough well away from 0 would point to an eye/head camera sync offset,
  which would also shift the syllable correlograms by the same amount.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.head_eye_lag_xcorr \\
        /mnt/data/moseq/head_eye_arhmm_17_sessions_no_pupil kappa_1e+03_nlags8_seed0 9 14
"""

import argparse
from pathlib import Path

import keypoint_moseq as kpms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from python_code.moseq.head_eye_features import _skull_angles
from python_code.moseq.kpms_loader import load_eye_in_head_kpms
from python_code.moseq.training_config import load_training_recording_folders

FPS = 120.0
_YAW_RATE_COLUMN = 2  # of `_skull_angles`
EYES = ["left eye", "right eye", "both eyes"]
_EYE_COLORS = {"left eye": "tab:blue", "right eye": "tab:orange", "both eyes": "k"}


def load_velocities(recording_folders) -> dict[str, dict[str, NDArray]]:
    """`{recording: {"head": (n,), "left eye": (n,), ...}}`, deg/s, + = leftward, NaN where invalid."""
    left, right = load_eye_in_head_kpms(recording_folders, quality_level="medium")
    out = {}
    for rf in recording_folders:
        name = rf.recording_name
        head = _skull_angles(rf)[:, _YAW_RATE_COLUMN]
        left_h = -left[name][:, 0]  # leftward horizontal angle, deg
        right_h = right[name][:, 0]
        if not len(head) == len(left_h) == len(right_h):
            raise ValueError(f"'{name}': frame counts differ")
        velocities = {
            "head": head,
            "left eye": np.gradient(left_h) * FPS,
            "right eye": np.gradient(right_h) * FPS,
        }
        velocities["both eyes"] = (velocities["left eye"] + velocities["right eye"]) / 2
        out[name] = velocities
    return out


def lagged_corr(head: NDArray, eye: NDArray, lags: NDArray, min_pairs: int) -> NDArray:
    """r(tau) = corr(head[t + tau], eye[t]) over the t where both are finite."""
    n = len(head)
    r = np.full(len(lags), np.nan)
    for i, tau in enumerate(lags):
        if tau >= 0:
            h, e = head[tau:], eye[: n - tau]
        else:
            h, e = head[: n + tau], eye[-tau:]
        ok = np.isfinite(h) & np.isfinite(e)
        if ok.sum() < min_pairs:
            continue
        h, e = h[ok], e[ok]
        if h.std() == 0 or e.std() == 0:
            continue
        r[i] = np.corrcoef(h, e)[0, 1]
    return r


def syllable_windows(syllables: NDArray, pad: int) -> dict[int, list[tuple[int, int, int]]]:
    """`{syllable: [(window_start, onset, window_stop), ...]}` for one session."""
    onsets = np.r_[0, np.flatnonzero(np.diff(syllables)) + 1]
    offsets = np.r_[onsets[1:], len(syllables)]
    out: dict[int, list] = {}
    for on, off in zip(onsets, offsets):
        out.setdefault(int(syllables[on]), []).append((max(0, on - pad), on, min(len(syllables), off + pad)))
    return out


def instance_correlograms(
    windows: list[tuple[str, int, int, int]], velocities, lags: NDArray, min_pairs: int
) -> dict[str, NDArray]:
    """`{eye: (n_instances, n_lags)}`."""
    out = {eye: [] for eye in EYES}
    for name, start, _, stop in windows:
        v = velocities[name]
        for eye in EYES:
            out[eye].append(lagged_corr(v["head"][start:stop], v[eye][start:stop], lags, min_pairs))
    return {eye: np.array(rows) for eye, rows in out.items()}


def fisher_mean_ci(r: NDArray, n_boot: int = 1000, seed: int = 0) -> tuple[NDArray, NDArray, NDArray]:
    """Fisher-z mean across instances (rows) and a bootstrap 95% CI, per lag."""
    z = np.arctanh(np.clip(r, -0.999, 0.999))
    z = z[np.isfinite(z).all(axis=1)]
    rng = np.random.default_rng(seed)
    boots = np.array([z[rng.integers(0, len(z), len(z))].mean(axis=0) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return np.tanh(z.mean(axis=0)), np.tanh(lo), np.tanh(hi)


def triggered_average(windows, velocities, pre: int, post: int) -> dict[str, NDArray]:
    """
    Onset-aligned (n, pre + post) velocities, each instance sign-flipped so its
    net head yaw over [onset, onset + post) is positive: a head turn in either
    direction then reads as "+" and the eye's sign is relative to the head.
    """
    out = {k: [] for k in ["head", *EYES]}
    for name, _, on, _ in windows:
        v = velocities[name]
        if on - pre < 0 or on + post > len(v["head"]):
            continue
        sign = np.sign(np.nansum(v["head"][on : on + post]))
        if sign == 0:
            continue
        for k in out:
            out[k].append(sign * v[k][on - pre : on + post])
    return {k: np.array(rows) for k, rows in out.items()}


def peak(lags_ms: NDArray, r: NDArray, positive: bool) -> tuple[float, float]:
    i = np.nanargmax(r) if positive else np.nanargmin(r)
    return lags_ms[i], r[i]


def plot_syllable(
    syllable: int,
    corrs: dict[str, NDArray],
    reference: dict[str, NDArray],
    session_r: dict[str, NDArray],
    triggered: dict[str, NDArray],
    lags: NDArray,
    pre: int,
    pad: int,
    output_dir: Path,
) -> list[str]:
    lags_ms = lags / FPS * 1000
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    lines = []
    n = len(next(iter(corrs.values())))
    fig.suptitle(
        f"syllable {syllable}: head yaw velocity vs. horizontal eye velocity ({n} instances, "
        f"window = onset - {pad} to offset + {pad} frames)\n"
        "r(lag) = corr(head[t + lag], eye[t]);  lag > 0: EYE LEADS,  lag < 0: HEAD LEADS",
        fontsize=13,
    )

    for col, eye in enumerate(EYES):
        ax = axes[0, col]
        mean, lo, hi = fisher_mean_ci(corrs[eye])
        ref_mean, ref_lo, ref_hi = fisher_mean_ci(reference[eye])
        ax.fill_between(lags_ms, ref_lo, ref_hi, color="0.7", alpha=0.4)
        ax.plot(lags_ms, ref_mean, color="0.5", label="other syllables (instance mean)")
        ax.plot(lags_ms, session_r[eye], color="tab:green", linestyle=":", label="whole session (all frames)")
        ax.fill_between(lags_ms, lo, hi, color=_EYE_COLORS[eye], alpha=0.25)
        ax.plot(lags_ms, mean, color=_EYE_COLORS[eye], linewidth=2, label=f"syllable {syllable} (instance mean, 95% CI)")
        ax.axvline(0, color="0.3", linewidth=0.8)
        ax.axhline(0, color="0.3", linewidth=0.8)
        pos_lag, pos_r = peak(lags_ms, mean, True)
        neg_lag, neg_r = peak(lags_ms, mean, False)
        ax.plot(pos_lag, pos_r, "^", color="crimson", markersize=9)
        ax.plot(neg_lag, neg_r, "v", color="tab:purple", markersize=9)
        ses_lag, ses_r = peak(lags_ms, session_r[eye], False)
        ax.set_title(
            f"{eye}\nmax r = {pos_r:+.2f} at {pos_lag:+.0f} ms,  min r = {neg_r:+.2f} at {neg_lag:+.0f} ms"
        )
        ax.set_xlabel("lag (ms)   ← head leads | eye leads →")
        ax.set_ylabel("r (Fisher-z mean)")
        if col == 0:
            ax.legend(fontsize=8, loc="lower left")
        lines.append(
            f"syllable {syllable:2d} {eye:9s}: max r {pos_r:+.2f} @ {pos_lag:+5.1f} ms | "
            f"min r {neg_r:+.2f} @ {neg_lag:+5.1f} ms | whole-session min r {ses_r:+.2f} @ {ses_lag:+5.1f} ms"
        )

        # Per-instance lag of the positive peak.
        ax = axes[1, col] if col < 2 else None
        if ax is not None:
            r = corrs[eye]
            r = r[np.isfinite(r).all(axis=1)]
            peak_lags = lags_ms[np.argmax(r, axis=1)]
            ax.hist(peak_lags, bins=np.arange(lags_ms[0], lags_ms[-1] + 8.4, 8.33), color=_EYE_COLORS[eye], alpha=0.7)
            ax.axvline(0, color="0.3", linewidth=0.8)
            ax.axvline(np.median(peak_lags), color="crimson", linestyle="--",
                       label=f"median {np.median(peak_lags):+.0f} ms")
            frac = np.mean(peak_lags > 0)
            ax.set_title(f"{eye}: lag of each instance's max r\n{frac:.0%} eye-leads, "
                         f"{np.mean(peak_lags < 0):.0%} head-leads, n = {len(r)}")
            ax.set_xlabel("lag (ms)   ← head leads | eye leads →")
            ax.set_ylabel("instances")
            ax.legend(fontsize=8)

    # Onset-triggered average, sign-normalized to the head's turn direction.
    ax = axes[1, 2]
    t_ms = (np.arange(triggered["head"].shape[1]) - pre) / FPS * 1000
    ax2 = ax.twinx()
    head = np.nanmean(triggered["head"], axis=0)
    ax.plot(t_ms, head, color="tab:red", linewidth=2, label="head yaw velocity")
    for eye in EYES:
        ax2.plot(t_ms, np.nanmean(triggered[eye], axis=0), color=_EYE_COLORS[eye],
                 linewidth=2 if eye == "both eyes" else 1, label=eye)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("time from syllable onset (ms)")
    ax.set_ylabel("head yaw velocity (deg/s)", color="tab:red")
    ax2.set_ylabel("eye-in-head velocity (deg/s)")
    ax.set_title(f"onset-triggered mean (n = {len(triggered['head'])}),\n"
                 "each instance flipped so its head turn is +")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.tight_layout()
    path = output_dir / f"syllable_{syllable:02d}_head_eye_xcorr.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path)
    return lines


def run(
    project_dir: Path,
    model_name: str,
    syllables: list[int],
    pad: int = 30,
    max_lag: int = 30,
    min_pairs: int = 20,
    n_reference: int = 3000,
    seed: int = 0,
) -> None:
    output_dir = project_dir / model_name / "eye_syllable_plots" / "head_eye_xcorr"
    output_dir.mkdir(parents=True, exist_ok=True)

    recording_folders = load_training_recording_folders(project_dir)
    velocities = load_velocities(recording_folders)
    results = kpms.load_results(str(project_dir), model_name)
    lags = np.arange(-max_lag, max_lag + 1)

    windows: dict[int, list] = {}
    for name, res in results.items():
        if len(res["syllable"]) != len(velocities[name]["head"]):
            raise ValueError(f"'{name}': syllable and kinematics frame counts differ")
        for syl, wins in syllable_windows(res["syllable"], pad).items():
            windows.setdefault(syl, []).extend((name, *w) for w in wins)

    # Reference correlograms: random instances of every other syllable, and
    # the whole session's frames.
    rng = np.random.default_rng(seed)
    others = [w for syl, wins in windows.items() if syl not in syllables for w in wins]
    others = [others[i] for i in rng.choice(len(others), min(n_reference, len(others)), replace=False)]
    reference = instance_correlograms(others, velocities, lags, min_pairs)
    session_r = {}
    for eye in EYES:
        # Concatenate sessions with a NaN gap so no lag spans two sessions.
        gap = np.full(max_lag, np.nan)
        head = np.concatenate([x for v in velocities.values() for x in (v["head"], gap)])
        eye_v = np.concatenate([x for v in velocities.values() for x in (v[eye], gap)])
        session_r[eye] = lagged_corr(head, eye_v, lags, min_pairs)

    summary = []
    for syllable in syllables:
        corrs = instance_correlograms(windows[syllable], velocities, lags, min_pairs)
        triggered = triggered_average(windows[syllable], velocities, pre=pad, post=pad)
        summary += plot_syllable(syllable, corrs, reference, session_r, triggered, lags, pad, pad, output_dir)

    text = "\n".join(summary)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("syllables", type=int, nargs="+")
    parser.add_argument("--pad", type=int, default=30, help="frames added before onset and after offset")
    parser.add_argument("--max-lag", type=int, default=30, help="frames")
    args = parser.parse_args()
    run(args.project_dir, args.model_name, args.syllables, pad=args.pad, max_lag=args.max_lag)
