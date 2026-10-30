"""
How much animal identity is in each model's latent (PC) space?

Splits each session's `latent_state` trajectory (from one fitted kappa's
`results/*.csv`) into non-overlapping 1 s windows, summarizes each window by
the mean (posture) and std (movement) of every latent dim, and predicts the
ferret with a multinomial logistic regression under leave-one-session-out
cross-validation. Holding out whole sessions means the classifier must
recognize an animal on a day it never saw; animals with a single session can
only be trained on, never tested.

Also reports, per latent dim, the fraction of window-mean variance explained
by animal (eta squared), to show which PCs carry identity.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.animal_identity_decoding head
    python -m python_code.moseq.animal_identity_decoding eyes --kappa 1e3
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from python_code.moseq.development_stats import COHORT_OF_ANIMAL, FPS, parse_session
from python_code.moseq.run_multi_session_kappa_sweeps import KEEP_FRAMES, SWEEPS

WINDOW_FRAMES = FPS
# Latents barely depend on kappa, so one representative fit per model.
DEFAULT_KAPPA = {"head": "1e4", "eyes": "1e3", "eyes_recentered": "1e3", "scaled_eyes": "1e3",
                 "head_recentered": "1e4", "scaled_eyes_recentered": "1e3"}


def window_features(latents_by_session: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-window mean and std of each latent dim, for (n_frames, n_dims)
    trajectories keyed by session name. Applies `KEEP_FRAMES`."""
    rows = []
    for session, latents in latents_by_session.items():
        if session in KEEP_FRAMES:
            start, stop = KEEP_FRAMES[session]
            latents = latents[start:stop]
        n_windows = len(latents) // WINDOW_FRAMES
        windows = latents[: n_windows * WINDOW_FRAMES].reshape(n_windows, WINDOW_FRAMES, -1)
        animal, eo_day = parse_session(session)
        frame = pd.DataFrame(
            np.hstack([windows.mean(axis=1), windows.std(axis=1)]),
            columns=[f"mean_{i}" for i in range(latents.shape[1])] + [f"std_{i}" for i in range(latents.shape[1])],
        )
        frame["session"], frame["animal"], frame["eo_day"] = session, animal, eo_day
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def load_latents(results_dir: Path) -> dict[str, np.ndarray]:
    return {
        csv.stem: pd.read_csv(csv).filter(like="latent_state").to_numpy()
        for csv in sorted(results_dir.glob("*.csv"))
    }


def leave_one_session_out(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Window-level predictions for every session whose animal also has other
    sessions to train on."""
    sessions_per_animal = features.groupby("animal")["session"].nunique()
    predictions = []
    for session, test in features.groupby("session"):
        if sessions_per_animal[test["animal"].iloc[0]] < 2:
            continue
        train = features[features["session"] != session]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(train[columns], train["animal"])
        predictions.append(test[["session", "animal", "eo_day"]].assign(predicted=model.predict(test[columns])))
    return pd.concat(predictions, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> dict:
    """Window and session (majority vote) accuracy, and how much of the error
    lands on the cage-mate from the same cohort."""
    animals = sorted(predictions["animal"].unique())
    wrong = predictions[predictions["animal"] != predictions["predicted"]]
    same_cohort = (wrong["animal"].map(COHORT_OF_ANIMAL) == wrong["predicted"].map(COHORT_OF_ANIMAL)).mean()
    votes = predictions.groupby("session").agg(
        animal=("animal", "first"), predicted=("predicted", lambda p: p.value_counts().index[0])
    )
    return {
        "window_balanced_acc": balanced_accuracy_score(predictions["animal"], predictions["predicted"]),
        "chance": 1 / len(animals),
        "cohort_acc": (predictions["animal"].map(COHORT_OF_ANIMAL) == predictions["predicted"].map(COHORT_OF_ANIMAL)).mean(),
        "errors_to_same_cohort": same_cohort,
        "sessions_correct": f"{(votes['animal'] == votes['predicted']).sum()}/{len(votes)}",
    }


def eta_squared_by_dim(features: pd.DataFrame, columns: list[str]) -> pd.Series:
    def eta(col: str) -> float:
        x = features[col]
        between = features.groupby("animal")[col].agg(["mean", "size"])
        return float((between["size"] * (between["mean"] - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())

    return pd.Series({c: eta(c) for c in columns})


def plot(confusions: dict[str, pd.DataFrame], etas: dict[str, pd.Series], path: Path) -> None:
    fig, axes = plt.subplots(
        2, len(confusions), figsize=(5 * len(confusions), 9), constrained_layout=True, squeeze=False
    )
    for (title, cm), ax in zip(confusions.items(), axes[0]):
        ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cm.columns)), cm.columns)
        ax.set_yticks(range(len(cm.index)), cm.index)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if value > 0.6 else "#0b0b0b")
        ax.set_xlabel("Predicted ferret")
        ax.set_ylabel("True ferret (held-out sessions)")
        ax.set_title(f"{title}\nrow-normalized confusion", loc="left", fontsize=10)
    for (title, eta), ax in zip(etas.items(), axes[1]):
        dims = np.arange(len(eta))
        ax.bar(dims, eta.to_numpy(), color="#2a78d6", width=0.8)
        ax.set_xticks(dims, [c.removeprefix("mean_") for c in eta.index], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Latent dim (PC)")
        ax.set_ylabel("Variance explained by animal (η²)")
        ax.set_title(f"{title}\nidentity per PC (window means)", loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help=f"any of {sorted(SWEEPS)} (default: all)")
    parser.add_argument("--kappa", default=None, help="kappa to read latents from (default per model)")
    parser.add_argument("--out", default="/mnt/data/moseq/animal_identity_decoding")
    args = parser.parse_args()
    if unknown := set(args.models) - set(SWEEPS):
        parser.error(f"unknown models {sorted(unknown)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, confusions, etas = [], {}, {}
    for model in args.models or sorted(SWEEPS):
        kappa = args.kappa or DEFAULT_KAPPA[model]
        results_dir = Path(SWEEPS[model]["project_dir"]) / f"kappa_sweep_{float(kappa):.0e}" / "results"
        features = window_features(load_latents(results_dir))
        means = [c for c in features.columns if c.startswith("mean_")]
        stds = [c for c in features.columns if c.startswith("std_")]

        for feature_set, columns in {"posture (mean)": means, "posture + movement": means + stds}.items():
            predictions = leave_one_session_out(features, columns)
            summaries.append({"model": model, "features": feature_set} | summarize(predictions))
            if feature_set == "posture + movement":
                predictions.to_csv(out_dir / f"{model}_window_predictions.csv", index=False)
                animals = sorted(features["animal"].unique(), key=lambda a: (COHORT_OF_ANIMAL[a], a))
                cm = confusion_matrix(predictions["animal"], predictions["predicted"], labels=animals)
                cm = pd.DataFrame(cm, index=animals, columns=animals)
                confusions[model] = cm.loc[cm.sum(axis=1) > 0].div(cm.sum(axis=1)[cm.sum(axis=1) > 0], axis=0)
        etas[model] = eta_squared_by_dim(features, means)

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(etas).to_csv(out_dir / "eta_squared_by_pc.csv")
    plot(confusions, etas, out_dir / "identity_decoding.png")
    print(summary.round(3).to_string(index=False))
    print("\nper-PC variance explained by animal:")
    print(pd.DataFrame(etas).round(2).to_string())
    for model, cm in confusions.items():
        print(f"\n{model} confusion (rows = true, row-normalized):\n{cm.round(2).to_string()}")
