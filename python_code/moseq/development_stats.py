"""
Developmental statistics on fitted multi-session kappa sweeps.

For one model (`head`, `eyes`, `scaled_eyes` from `run_multi_session_kappa_sweeps.SWEEPS`)
and one or more kappas, computes per-session syllable statistics from the
`results/*.csv` syllable sequences and tests how they change with days since
eye opening (the EO/E number in the session name).

Each metric gets two age slopes:

- `within`: OLS with an animal fixed effect, so the slope comes only from
  age differences within the same animal. This is the cleaner developmental
  estimate, since the three cohorts (405/407 at EO0-1, 402/420 at E02-08,
  753/757 at EO7/E11) cover different ages and were recorded months apart.
- `mixed`: linear mixed model with an animal random intercept, which also
  uses between-animal age differences (and so is exposed to cohort effects).

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.development_stats head --kappa 1e4
    python -m python_code.moseq.development_stats eyes --kappa all

Outputs go to `<project_dir>/development_stats/<model_name>/`, plus a
`kappa_robustness.csv` in `<project_dir>/development_stats/` when more than one
kappa is run.
"""
import argparse
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from python_code.moseq.run_multi_session_kappa_sweeps import KEEP_FRAMES, SWEEPS

FPS = 120
# Same cutoff keypoint-moseq uses for "used" syllables; rarer ones are pooled
# into one "other" bin so per-syllable tests aren't run on near-empty states.
MIN_USAGE = 0.005
WINDOW_S = 60

SCALAR_METRICS = {
    "onset_rate_hz": "Syllable onsets per second",
    "median_duration_s": "Median syllable duration (s)",
    "mean_duration_s": "Mean syllable duration (s)",
    "usage_entropy_bits": "Usage entropy (bits)",
    "transition_entropy_bits": "Transition entropy H(next|current) (bits)",
    "n_syllables_1pct": "Syllables with usage > 1%",
    "unique_per_min": "Unique syllables per minute",
    "usage_pc1": "Usage composition PC1",
}

# Cohorts share a color (3 categorical slots validate for every pair); the
# animal within a cohort gets the marker.
COHORT_OF_ANIMAL = {"405": 0, "407": 0, "402": 1, "420": 1, "753": 2, "757": 2}
COHORT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
COHORT_LABELS = ["405/407", "402/420", "753/757"]
MARKER_OF_ANIMAL = {"405": "o", "402": "o", "753": "o", "407": "s", "420": "s", "757": "s"}


def parse_session(name: str) -> tuple[str, int]:
    animal = re.search(r"ferret_(\d+)", name).group(1)
    eo_day = int(re.search(r"_EO?(\d+)$", name).group(1))
    return animal, eo_day


def load_syllables(results_dir: Path) -> dict[str, np.ndarray]:
    syllables = {}
    for csv in sorted(results_dir.glob("*.csv")):
        seq = pd.read_csv(csv, usecols=["syllable"])["syllable"].to_numpy()
        if csv.stem in KEEP_FRAMES:
            start, stop = KEEP_FRAMES[csv.stem]
            seq = seq[start:stop]
        syllables[csv.stem] = seq
    return syllables


def run_lengths(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Syllable label and length (frames) of each run, dropping the first and
    last runs since they're cut off by the recording edges."""
    change = np.flatnonzero(np.diff(seq)) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [len(seq)]]))
    return seq[starts][1:-1], lengths[1:-1]


def entropy_bits(counts: np.ndarray) -> float:
    """Plug-in entropy with the Miller-Madow bias correction, so sessions with
    fewer onsets aren't biased toward lower entropy."""
    counts = counts[counts > 0]
    n = counts.sum()
    p = counts / n
    return float(-(p * np.log2(p)).sum() + (len(counts) - 1) / (2 * n * np.log(2)))


def unique_per_window(seq: np.ndarray, window_frames: int = WINDOW_S * FPS) -> float:
    """Mean number of distinct syllables per non-overlapping window (the
    trailing partial window is dropped), so sessions of different lengths
    are comparable."""
    n_windows = len(seq) // window_frames
    windows = seq[: n_windows * window_frames].reshape(n_windows, window_frames)
    return float(np.mean([len(np.unique(w)) for w in windows]))


def session_metrics(syllables: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-session scalar metrics, and per-session onset usage of each
    commonly used syllable (columns) plus an `other` column."""
    runs = {name: run_lengths(seq) for name, seq in syllables.items()}

    pooled = pd.Series(np.concatenate([labels for labels, _ in runs.values()])).value_counts(normalize=True)
    common = sorted(pooled.index[pooled >= MIN_USAGE])
    # Rare syllables share one label so they still count as states in the
    # entropy terms without each being its own sparse category.
    other = -1

    rows, usage_rows = [], []
    for name, (labels, lengths) in runs.items():
        labels = np.where(np.isin(labels, common), labels, other)
        animal, eo_day = parse_session(name)

        counts = pd.Series(labels).value_counts()
        usage = (counts / counts.sum()).reindex(common + [other], fill_value=0.0)

        # Consecutive runs can share a label once rare syllables are merged
        # into `other`; those aren't real transitions.
        prev, nxt = labels[:-1], labels[1:]
        keep = prev != nxt
        bigrams = pd.crosstab(prev[keep], nxt[keep])
        row_totals = bigrams.sum(axis=1)
        transition_entropy = sum(
            (row_totals[s] / row_totals.sum()) * entropy_bits(bigrams.loc[s].to_numpy()) for s in bigrams.index
        )

        rows.append(
            {
                "session": name,
                "animal": animal,
                "eo_day": eo_day,
                "n_frames": len(syllables[name]),
                "n_onsets": len(labels),
                "onset_rate_hz": len(labels) / (len(syllables[name]) / FPS),
                "median_duration_s": np.median(lengths) / FPS,
                "mean_duration_s": np.mean(lengths) / FPS,
                "usage_entropy_bits": entropy_bits(counts.to_numpy()),
                "transition_entropy_bits": transition_entropy,
                "n_syllables_1pct": int((usage.drop(other) > 0.01).sum()),
                "unique_per_min": unique_per_window(syllables[name]),
            }
        )
        usage_rows.append(usage.rename(name))

    metrics = pd.DataFrame(rows).set_index("session")
    usage = pd.DataFrame(usage_rows)
    usage.columns = [str(c) if c != other else "other" for c in usage.columns]

    # Hellinger (sqrt) transform before PCA so common syllables don't dominate
    # the composition axis.
    hellinger = np.sqrt(usage.to_numpy())
    centered = hellinger - hellinger.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vt[0]
    # Sign PC1 so positive means "older-like", which makes slopes comparable
    # across kappas.
    if spearmanr(pc1, metrics["eo_day"]).statistic < 0:
        pc1 = -pc1
    metrics["usage_pc1"] = pc1
    return metrics, usage


def age_slopes(df: pd.DataFrame, column: str) -> dict:
    """Age slope of `column` (per eye-opening day) within animals and from a
    mixed model, plus a pooled Spearman correlation."""
    data = df[[column, "eo_day", "animal"]].rename(columns={column: "y"})
    out = {"metric": column}

    within = smf.ols("y ~ eo_day + C(animal)", data=data).fit()
    out.update(
        within_slope=within.params["eo_day"],
        within_ci_low=within.conf_int().loc["eo_day", 0],
        within_ci_high=within.conf_int().loc["eo_day", 1],
        within_p=within.pvalues["eo_day"],
    )

    with warnings.catch_warnings():
        # Six groups with 1-6 sessions each: boundary/convergence warnings are
        # expected when the animal variance is near zero.
        warnings.simplefilter("ignore")
        mixed = smf.mixedlm("y ~ eo_day", data=data, groups=data["animal"]).fit(reml=True)
    out.update(mixed_slope=mixed.params["eo_day"], mixed_p=mixed.pvalues["eo_day"])

    rho = spearmanr(data["eo_day"], data["y"])
    out.update(spearman_rho=rho.statistic, spearman_p=rho.pvalue)
    return out


def syllable_slopes(metrics: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    df = metrics[["eo_day", "animal"]].join(usage.add_prefix("syl_"))
    rows = [age_slopes(df, f"syl_{s}") | {"mean_usage": usage[s].mean()} for s in usage.columns]
    table = pd.DataFrame(rows)
    table["syllable"] = table.pop("metric").str.removeprefix("syl_")
    for test in ("within", "mixed"):
        table[f"{test}_q"] = multipletests(table[f"{test}_p"], method="fdr_bh")[1]
    return table.set_index("syllable").sort_values("within_p")


def _scatter_by_animal(ax, metrics: pd.DataFrame, column: str) -> None:
    for animal, group in metrics.sort_values("eo_day").groupby("animal"):
        color = COHORT_COLORS[COHORT_OF_ANIMAL[animal]]
        ax.plot(group["eo_day"], group[column], color=color, linewidth=2, alpha=0.6)
        ax.scatter(
            group["eo_day"], group[column], s=64, color=color, marker=MARKER_OF_ANIMAL[animal],
            edgecolor="white", linewidth=2, zorder=3, label=animal,
        )


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9a9994")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_metrics(metrics: pd.DataFrame, slopes: pd.DataFrame, title: str, path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for ax, (column, label) in zip(axes.flat, SCALAR_METRICS.items()):
        _scatter_by_animal(ax, metrics, column)
        s = slopes.loc[column]
        ax.set_title(
            f"{label}\nwithin-animal slope {s.within_slope:+.3g}/day, p={s.within_p:.2g}",
            fontsize=10, color="#0b0b0b", loc="left",
        )
        ax.set_xlabel("Days since eye opening", fontsize=9, color="#52514e")
        _style(ax)

    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (COHORT_OF_ANIMAL[labels[i]], labels[i]))
    legend_ax.legend(
        [handles[i] for i in order], [f"ferret {labels[i]}" for i in order],
        loc="center left", frameon=False, fontsize=10, title="Cohorts share a color", title_fontsize=10,
    )
    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_usage(metrics: pd.DataFrame, usage: pd.DataFrame, syl_slopes: pd.DataFrame, title: str, path: Path) -> None:
    order = metrics.sort_values(["eo_day", "animal"]).index
    syllables = syl_slopes.sort_values("within_slope").index
    data = usage.loc[order, syllables].T.to_numpy()

    fig, (ax_heat, ax_slope) = plt.subplots(
        1, 2, figsize=(14, max(5, 0.22 * len(syllables) + 2)), width_ratios=[3, 1], sharey=True,
        constrained_layout=True,
    )
    im = ax_heat.imshow(np.sqrt(data), aspect="auto", cmap="Blues", interpolation="nearest")
    ax_heat.set_xticks(
        range(len(order)), [f"{metrics.loc[s, 'animal']} E{metrics.loc[s, 'eo_day']}" for s in order],
        rotation=90, fontsize=8,
    )
    ax_heat.set_yticks(range(len(syllables)), syllables, fontsize=8)
    ax_heat.set_ylabel("Syllable (sorted by within-animal age slope)")
    ax_heat.set_title("Onset usage (sqrt), sessions sorted by age", loc="left", fontsize=10)
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.6)
    cbar.set_label("sqrt(usage)")

    s = syl_slopes.loc[syllables]
    y = np.arange(len(syllables))
    significant = s["within_q"] < 0.05
    ax_slope.axvline(0, color="#9a9994", linewidth=1)
    ax_slope.hlines(y, s["within_ci_low"], s["within_ci_high"], color="#9a9994", linewidth=2)
    ax_slope.scatter(s["within_slope"][~significant], y[~significant], s=36, color="#9a9994", zorder=3)
    ax_slope.scatter(
        s["within_slope"][significant], y[significant], s=48, color="#2a78d6", zorder=3, label="FDR q < 0.05",
    )
    ax_slope.set_xlabel("Within-animal usage slope per day (95% CI)", fontsize=9)
    ax_slope.legend(loc="lower right", frameon=False, fontsize=8)
    _style(ax_slope)
    ax_slope.grid(False)
    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def time_fractions(syllables: dict[str, np.ndarray], columns: list[str]) -> pd.DataFrame:
    """Fraction of each session's frames spent in each syllable, with
    syllables outside `columns` (the commonly used set from
    `session_metrics`) pooled into `other`."""
    rows = {}
    for name, seq in syllables.items():
        labels = pd.Series(seq.astype(str))
        labels[~labels.isin(columns)] = "other"
        rows[name] = labels.value_counts(normalize=True).reindex(columns, fill_value=0.0)
    return pd.DataFrame(rows).T


def plot_time_fractions(metrics: pd.DataFrame, fractions: pd.DataFrame, title: str, path: Path) -> None:
    """One panel per syllable: percent of each session's recording spent in
    it, sessions sorted by age and colored by cohort."""
    order = metrics.sort_values(["eo_day", "animal"]).index
    syllables = [s for s in fractions.mean().sort_values(ascending=False).index if s != "other"] + ["other"]
    percent = 100 * fractions.loc[order]
    colors = [COHORT_COLORS[COHORT_OF_ANIMAL[a]] for a in metrics.loc[order, "animal"]]
    x = np.arange(len(order))

    n_cols = 6
    n_rows = int(np.ceil(len(syllables) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), sharex=True, constrained_layout=True,
        squeeze=False,
    )
    for ax, syllable in zip(axes.flat, syllables):
        ax.bar(x, percent[syllable], width=0.85, color=colors)
        ax.set_title(f"syllable {syllable} · mean {percent[syllable].mean():.1f}%", loc="left", fontsize=9)
        _style(ax)
    for ax in axes.flat[len(syllables):]:
        ax.axis("off")
    # Bottom labels on the last filled panel of each column.
    for col in range(n_cols):
        filled = [r for r in range(n_rows) if r * n_cols + col < len(syllables)]
        if filled:
            ax = axes[filled[-1], col]
            ax.tick_params(labelbottom=True)
            ax.set_xticks(
                x, [f"{metrics.loc[s, 'animal']} E{metrics.loc[s, 'eo_day']}" for s in order],
                rotation=90, fontsize=7,
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("% of recording", fontsize=9, color="#52514e")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COHORT_COLORS]
    fig.legend(handles, [f"ferrets {label}" for label in COHORT_LABELS], loc="outside upper right",
               ncols=3, frameon=False, fontsize=9)
    fig.suptitle(f"{title}\nTime in each syllable per session (sessions sorted by age)", fontsize=12,
                 x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(model: str, model_name: str) -> pd.DataFrame:
    project_dir = Path(SWEEPS[model]["project_dir"])
    out_dir = project_dir / "development_stats" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    syllables = load_syllables(project_dir / model_name / "results")
    metrics, usage = session_metrics(syllables)
    fractions = time_fractions(syllables, list(usage.columns))
    slopes = pd.DataFrame([age_slopes(metrics, m) for m in SCALAR_METRICS]).set_index("metric")
    syl_slopes = syllable_slopes(metrics, usage)

    metrics.to_csv(out_dir / "session_metrics.csv")
    usage.to_csv(out_dir / "session_usage.csv")
    fractions.to_csv(out_dir / "session_time_fraction.csv")
    slopes.to_csv(out_dir / "age_slopes.csv")
    syl_slopes.to_csv(out_dir / "syllable_age_slopes.csv")

    title = f"{model} · {model_name} · {len(metrics)} sessions, {metrics['animal'].nunique()} animals"
    plot_metrics(metrics, slopes, title, out_dir / "metrics_vs_age.png")
    plot_usage(metrics, usage, syl_slopes, title, out_dir / "usage_vs_age.png")
    plot_time_fractions(metrics, fractions, title, out_dir / "syllable_time_by_session.png")

    print(f"\n=== {model} {model_name} -> {out_dir}")
    print(slopes[["within_slope", "within_p", "mixed_slope", "mixed_p", "spearman_rho"]].round(4).to_string())
    n_sig = int((syl_slopes["within_q"] < 0.05).sum())
    print(f"syllables with within-animal age effect at FDR q<0.05: {n_sig}/{len(syl_slopes)}")

    slopes["model_name"] = model_name
    slopes["n_sig_syllables"] = n_sig
    slopes["n_syllables"] = len(syl_slopes)
    return slopes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(SWEEPS))
    parser.add_argument("--kappa", nargs="+", default=["all"], help="kappas like 1e4 3e3, or 'all'")
    args = parser.parse_args()

    project_dir = Path(SWEEPS[args.model]["project_dir"])
    available = sorted((p.name for p in project_dir.glob("kappa_sweep_*") if (p / "results").is_dir()),
                       key=lambda n: float(n.removeprefix("kappa_sweep_")))
    if args.kappa == ["all"]:
        model_names = available
    else:
        model_names = [f"kappa_sweep_{float(k):.0e}" for k in args.kappa]

    all_slopes = pd.concat([analyze(args.model, name) for name in model_names])
    if len(model_names) > 1:
        robustness = all_slopes.reset_index().pivot(index="metric", columns="model_name", values="within_p")
        all_slopes.to_csv(project_dir / "development_stats" / "kappa_robustness.csv")
        print("\nwithin-animal age p-values across kappas:")
        print(robustness[model_names].round(4).to_string())
