"""
Does removing per-session static shape remove animal identity from a
model's PC space?

For one sweep in `run_multi_session_kappa_sweeps.SWEEPS`, loads its sessions
with and without the loader's `recenter_static_shape` option, fits a
kpms-style PCA on each (kpms outlier removal, then the same
`preprocess_for_pca` -- egocentric alignment unless the loader config fixes
the heading, low-confidence keypoints interpolated), and runs the
leave-one-session-out identity decoder from `animal_identity_decoding` on
both. This skips the AR-HMM fit, so it's a quick check before refitting the
full model.

Run in the `keypoint_moseq` env (CPU is fine):

    JAX_PLATFORMS=cpu python -m python_code.moseq.static_shape_check eyes
    JAX_PLATFORMS=cpu python -m python_code.moseq.static_shape_check head
"""
import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax_moseq.models.keypoint_slds.alignment import preprocess_for_pca
from keypoint_moseq.util import find_medoid_distance_outliers, interpolate_keypoints
from sklearn.decomposition import PCA

from python_code.moseq.animal_identity_decoding import (
    eta_squared_by_dim,
    leave_one_session_out,
    plot,
    summarize,
    window_features,
)
from python_code.moseq.development_stats import COHORT_OF_ANIMAL
from python_code.moseq.run_moseq_pipeline import _LOADER_CONFIG, load_keypoints
from python_code.moseq.run_multi_session_kappa_sweeps import SWEEPS, recording_folders

DEFAULT_LATENT_DIM = 10
CONF_THRESHOLD = 0.5
PCA_FITTING_NUM_FRAMES = 1_000_000
OUTLIER_SCALE_FACTOR = 6.0


def remove_outliers(coordinates: dict[str, np.ndarray], confidences: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """`kpms.outlier_removal` without its QA plots."""
    cleaned, confs = {}, {}
    for name, coords in coordinates.items():
        mask = find_medoid_distance_outliers(coords, outlier_scale_factor=OUTLIER_SCALE_FACTOR)["mask"]
        cleaned[name] = interpolate_keypoints(coords, mask)
        confs[name] = np.where(mask, 0, confidences[name])
    return cleaned, confs


def pcs_by_session(
    coordinates: dict[str, np.ndarray],
    confidences: dict[str, np.ndarray],
    anterior_idxs: list[int],
    posterior_idxs: list[int],
    fix_heading: bool,
    latent_dim: int,
) -> tuple[dict, PCA]:
    flat = {
        name: np.asarray(
            # Leading batch dim: `interpolate` works along axis 1, i.e. time
            # in kpms's (n_segments, T, k, d) layout.
            preprocess_for_pca(
                jnp.asarray(coordinates[name])[None], anterior_idxs, posterior_idxs,
                jnp.asarray(confidences[name])[None], conf_threshold=CONF_THRESHOLD, fix_heading=fix_heading,
            )[0][0]
        )
        for name in coordinates
    }
    stacked = np.concatenate(list(flat.values()))
    rng = np.random.default_rng(0)
    fit_frames = stacked[rng.permutation(len(stacked))[:PCA_FITTING_NUM_FRAMES]]
    pca = PCA(n_components=latent_dim).fit(fit_frames)
    return {name: pca.transform(x) for name, x in flat.items()}, pca


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(SWEEPS))
    parser.add_argument("--out", default="/mnt/data/moseq/animal_identity_decoding/static_shape_check")
    args = parser.parse_args()
    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = SWEEPS[args.model]
    cfg = _LOADER_CONFIG[sweep["loader"]]
    folders = recording_folders(sweep["exclude_sessions"])
    latent_dim = sweep.get("latent_dim") or DEFAULT_LATENT_DIM

    summaries, confusions, etas = [], {}, {}
    for variant, options in {"raw": {}, "recentered": {"recenter_static_shape": True}}.items():
        coordinates, confidences, bodyparts = load_keypoints(sweep["loader"], folders, **options)
        use = [bodyparts.index(bp) for bp in cfg["use_bodyparts"]]
        coordinates = {name: c[:, use] for name, c in coordinates.items()}
        confidences = {name: c[:, use] for name, c in confidences.items()}
        anterior = [cfg["use_bodyparts"].index(bp) for bp in cfg["anterior_bodyparts"]]
        posterior = [cfg["use_bodyparts"].index(bp) for bp in cfg["posterior_bodyparts"]]

        # Same order as the pipeline: the loader (incl. recentering) runs
        # before kpms's outlier removal.
        pcs, pca = pcs_by_session(
            *remove_outliers(coordinates, confidences), anterior, posterior,
            cfg.get("fix_heading", False), latent_dim,
        )
        features = window_features(pcs)
        means = [c for c in features.columns if c.startswith("mean_")]
        stds = [c for c in features.columns if c.startswith("std_")]
        for feature_set, columns in {"posture (mean)": means, "posture + movement": means + stds}.items():
            predictions = leave_one_session_out(features, columns)
            summaries.append(
                {"variant": variant, "features": feature_set}
                | summarize(predictions)
                | {"pc_var_explained": pca.explained_variance_ratio_.sum()}
            )
            if feature_set == "posture + movement":
                animals = sorted(predictions["animal"].unique(), key=lambda a: (COHORT_OF_ANIMAL[a], a))
                cm = pd.crosstab(predictions["animal"], predictions["predicted"], normalize="index")
                confusions[f"{args.model} PCA, {variant}"] = cm.reindex(index=animals, columns=animals, fill_value=0)
        etas[f"{args.model} PCA, {variant}"] = eta_squared_by_dim(features, means)
        print(f"{variant}: variance explained per PC {np.round(pca.explained_variance_ratio_, 3)}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(etas).to_csv(out_dir / "eta_squared_by_pc.csv")
    plot(confusions, etas, out_dir / "identity_decoding.png")
    print(summary.round(3).to_string(index=False))
    print("\nper-PC variance explained by animal:")
    print(pd.DataFrame(etas).round(2).to_string())
