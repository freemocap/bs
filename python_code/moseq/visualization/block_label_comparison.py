"""
Syllable agreement between joint, eyes-only and head-only AR-HMMs
=================================================================

Compares frame-level syllable labels from `run_head_eye_arhmm.py` fits of the
joint model and its `--block eyes` / `--block head` counterparts, which train
and label on the same frames.

Per pair of models (A, B), on frames usable by every model (`training_mask`):

- `mi_bits`: mutual information between the two label sequences.
- `mi_shifted_bits`: the same with B circularly shifted against A per session
  (>= 60 s), the chance level given each sequence's autocorrelation and
  plug-in estimator bias. `mi_excess_bits` is the difference.
- `u_a_given_b`, `u_b_given_a`: uncertainty coefficients, MI / H(A) and
  MI / H(B): the fraction of one model's label entropy the other explains.
- `boundary_match`, `boundary_match_shifted`: fraction of A's syllable
  transitions with a B transition within `_BOUNDARY_TOLERANCE` frames.

Seed-vs-seed pairs of the same model are the ceiling for cross-model
agreement. `joint_vs_eyes_x_head` treats the (eyes, head) label pair as one
label, so `u_a_given_b` says how much of the joint label is explained by
knowing both block labels.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.visualization.block_label_comparison
"""

import argparse
from itertools import combinations
from pathlib import Path

import keypoint_moseq as kpms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from python_code.moseq.head_eye_features import load_head_eye_features, training_mask
from python_code.moseq.run_multi_session_kappa_sweeps import KEEP_FRAMES
from python_code.moseq.training_config import load_training_recording_folders

FPS = 120
NLAGS = 8
_BOUNDARY_TOLERANCE = 2
_MIN_SHIFT_SECONDS = 60
_N_SHIFTS = 5
_MIN_FREQUENCY = 0.005

_ROOT = Path("/mnt/data/moseq")
PROJECTS = {
    "joint": _ROOT / "head_eye_arhmm_17_sessions_no_pupil",
    "eyes": _ROOT / "head_eye_arhmm_17_sessions_eyes_only",
    "head": _ROOT / "head_eye_arhmm_17_sessions_head_only",
}
SEEDS = [0, 1, 2]
OUTPUT_DIR = _ROOT / "head_eye_arhmm_17_sessions_block_comparison"


def _model_name(seed: int) -> str:
    return f"kappa_1e+03_nlags{NLAGS}_seed{seed}"


def _load_labels(project_dir: Path, seed: int) -> dict[str, NDArray[np.int64]]:
    results = kpms.load_results(str(project_dir), _model_name(seed))
    return {name: np.asarray(r["syllable"]) for name, r in results.items()}


def _entropy(counts: NDArray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _mutual_information(a: NDArray, b: NDArray) -> tuple[float, float, float]:
    """(MI, H(a), H(b)) in bits for paired integer label arrays."""
    _, a_idx = np.unique(a, return_inverse=True)
    _, b_idx = np.unique(b, return_inverse=True)
    joint = np.zeros((a_idx.max() + 1, b_idx.max() + 1))
    np.add.at(joint, (a_idx, b_idx), 1)
    h_a, h_b = _entropy(joint.sum(axis=1)), _entropy(joint.sum(axis=0))
    return h_a + h_b - _entropy(joint.ravel()), h_a, h_b


def _transitions(labels: NDArray, keep: NDArray) -> NDArray[np.bool_]:
    """True on frames where the label changes from the previous frame, both kept."""
    changed = np.zeros(len(labels), dtype=bool)
    changed[1:] = (labels[1:] != labels[:-1]) & keep[1:] & keep[:-1]
    return changed


def _boundary_match(a: dict, b: dict, keep: dict) -> float:
    matched = total = 0
    width = 2 * _BOUNDARY_TOLERANCE + 1
    for name in a:
        a_edges = _transitions(a[name], keep[name])
        b_near = np.convolve(_transitions(b[name], keep[name]), np.ones(width), mode="same") > 0
        matched += (a_edges & b_near).sum()
        total += a_edges.sum()
    return matched / total


def _shifted(labels: dict, rng: np.random.Generator) -> dict:
    min_shift = _MIN_SHIFT_SECONDS * FPS
    return {name: np.roll(x, int(rng.integers(min_shift, len(x) - min_shift))) for name, x in labels.items()}


def _pooled(labels: dict, keep: dict) -> NDArray:
    return np.concatenate([labels[name][keep[name]] for name in sorted(labels)])


def compare(a: dict, b: dict, keep: dict, seed: int = 0) -> dict:
    """All metrics for one pair of label dicts. For the shifted baseline only
    B moves, and `keep` is applied to A's frames, so shifted B labels may come
    from frames that were masked."""
    mi, h_a, h_b = _mutual_information(_pooled(a, keep), _pooled(b, keep))
    rng = np.random.default_rng(seed)
    shifted_mi, shifted_boundary = [], []
    for _ in range(_N_SHIFTS):
        b_shifted = _shifted(b, rng)
        shifted_mi.append(_mutual_information(_pooled(a, keep), _pooled(b_shifted, keep))[0])
        shifted_boundary.append(_boundary_match(a, b_shifted, keep))
    return {
        "h_a_bits": h_a,
        "h_b_bits": h_b,
        "mi_bits": mi,
        "mi_shifted_bits": np.mean(shifted_mi),
        "mi_excess_bits": mi - np.mean(shifted_mi),
        "u_a_given_b": mi / h_a,
        "u_b_given_a": mi / h_b,
        "boundary_match": _boundary_match(a, b, keep),
        "boundary_match_shifted": np.mean(shifted_boundary),
    }


def _paired_labels(a: dict, b: dict) -> dict:
    """One label per (a, b) pair."""
    offset = max(x.max() for x in b.values()) + 1
    return {name: a[name] * offset + b[name] for name in a}


def plot_contingency(a: dict, b: dict, keep: dict, a_label: str, b_label: str, ax: plt.Axes) -> None:
    """P(B syllable | A syllable) for syllables above `_MIN_FREQUENCY`, B
    columns ordered by the A row where they peak."""
    pooled_a, pooled_b = _pooled(a, keep), _pooled(b, keep)
    common = lambda x: [s for s, n in zip(*np.unique(x, return_counts=True)) if n / len(x) >= _MIN_FREQUENCY]
    rows, cols = common(pooled_a), common(pooled_b)
    table = pd.crosstab(pooled_a, pooled_b).reindex(index=rows, columns=cols, fill_value=0)
    conditional = table.div(table.sum(axis=1), axis=0)
    cols = sorted(cols, key=lambda c: (conditional[c].values.argmax(), -conditional[c].max()))
    conditional = conditional[cols]

    image = ax.imshow(conditional.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols, fontsize=6)
    ax.set_yticks(np.arange(len(rows)), rows, fontsize=6)
    ax.set_xlabel(f"{b_label} syllable")
    ax.set_ylabel(f"{a_label} syllable")
    ax.set_title(f"P({b_label} | {a_label})")
    plt.colorbar(image, ax=ax, fraction=0.046)


def run_comparison(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_head_eye_features(
        load_training_recording_folders(PROJECTS["joint"]), keep_frames=KEEP_FRAMES
    )
    keep = {name: training_mask(valid, NLAGS) for name, valid in features.valid.items()}
    labels = {(model, seed): _load_labels(path, seed) for model, path in PROJECTS.items() for seed in SEEDS}
    for key, by_name in labels.items():
        if set(by_name) != set(keep) or any(len(by_name[n]) != len(keep[n]) for n in keep):
            raise ValueError(f"{key} labels don't match the feature frames")

    rows = []
    for seed in SEEDS:
        pairs = {
            "eyes_vs_head": (labels["eyes", seed], labels["head", seed]),
            "joint_vs_eyes": (labels["joint", seed], labels["eyes", seed]),
            "joint_vs_head": (labels["joint", seed], labels["head", seed]),
            "joint_vs_eyes_x_head": (
                labels["joint", seed],
                _paired_labels(labels["eyes", seed], labels["head", seed]),
            ),
        }
        for pair, (a, b) in pairs.items():
            rows.append({"pair": pair, "seeds": f"{seed}", **compare(a, b, keep, seed)})
            print(rows[-1])
    for model in PROJECTS:
        for s1, s2 in combinations(SEEDS, 2):
            rows.append({
                "pair": f"{model}_seed_vs_seed",
                "seeds": f"{s1}-{s2}",
                **compare(labels[model, s1], labels[model, s2], keep, s1),
            })
            print(rows[-1])

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "label_agreement.csv", index=False)
    summary = table.drop(columns="seeds").groupby("pair", sort=False).agg(["mean", "std"])
    summary.to_csv(output_dir / "label_agreement_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (a, b) in zip(axes, [("joint", "eyes"), ("joint", "head"), ("eyes", "head")]):
        plot_contingency(labels[a, 0], labels[b, 0], keep, a, b, ax)
    fig.suptitle("Seed 0, frames usable by all models; syllables >= 0.5% of frames")
    plt.tight_layout()
    fig.savefig(output_dir / "contingency_seed0.png", dpi=150)
    plt.close(fig)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    table = run_comparison(args.output_dir)
    cols = ["mi_bits", "mi_excess_bits", "u_a_given_b", "u_b_given_a", "boundary_match", "boundary_match_shifted"]
    print(table.groupby("pair", sort=False)[cols].mean().round(3).to_string())
