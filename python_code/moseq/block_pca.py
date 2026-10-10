"""
Separate PCA per keypoint group, assembled into one block-diagonal PCA object
==============================================================================

Implements Option B from `eye_movement_in_pcs.md`: instead of kpms's default
single joint PCA over all keypoints (in which head displacement dominates
variance and crowds eye movement out of the top `latent_dim` components),
fit an independent PCA per keypoint group (e.g. head / left eye / right eye),
normalize each group's chosen components to unit variance, and assemble them
into one block-diagonal loading matrix. Each group is then guaranteed a fixed
share of the latent space, regardless of its raw physical amplitude relative
to the other groups.

The result is a genuine `sklearn.decomposition.PCA` instance living in the
exact same embedded space kpms's own `fit_pca` produces (see
`center_embedding` in `jax_moseq.models.keypoint_slds.alignment`), so it's a
drop-in replacement for `kpms.fit_pca`'s return value -- `kpms.save_pca`,
`fit_ar_model`, `fit_full_model`, `extract_results`, and all of kpms's own
plotting (`plot_pcs`, `plot_scree`) work unchanged.  Only `latent_dim` in
config.yml needs to be set to the sum of the per-group dims before fitting
the AR/full model (see `run_moseq_pipeline.fit_pca`'s `keypoint_groups`/
`block_latent_dims` parameters, which do this automatically).

Why this needs to operate in the embedded space, not just pad+concatenate
keypoint-space components
----------------------------------------------------------------------------
kpms doesn't feed raw keypoint coordinates to the AR-HMM; `align_egocentric`
removes translation and yaw, and `center_embedding(k)` then changes basis
from `k` (redundant, translation-free) keypoints to `k-1` independent
dimensions. That embedding is baked into the rest of the pipeline (AR-HMM
init, `extract_results`, etc. all recompute it internally from the same raw
`Y`/`mask`), so a substitute PCA's `components_`/`mean_` must live in that
same `(k-1)*d`-dimensional space for `pca.transform` to give the right
answer downstream.

A keypoint-space component with nonzero values only on one group's keypoints
does *not* automatically land in that space unchanged: the embedding is a
projection that discards each frame's mean position across *all* `k`
keypoints (not per-group), so a candidate component must itself be
zero-mean across all `k` keypoints (per spatial axis) for the projection to
preserve it exactly. `_pad_and_center` below enforces that before the
embedding step -- it's an unavoidable, mathematically necessary side effect
of embedding "block-diagonal in keypoint space" is not quite achievable
exactly, but the leakage this introduces onto other groups' keypoints is
tiny (the padded component's own mean divided across ~20+ keypoints).
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def _validate_groups(use_bodyparts: Sequence[str], keypoint_groups: dict[str, list[str]]) -> None:
    covered = {bp for bps in keypoint_groups.values() for bp in bps}
    if covered != set(use_bodyparts):
        missing = set(use_bodyparts) - covered
        extra = covered - set(use_bodyparts)
        raise ValueError(
            "keypoint_groups must partition use_bodyparts exactly. "
            f"Missing: {sorted(missing)}. Not in use_bodyparts: {sorted(extra)}."
        )
    if len(covered) != sum(len(bps) for bps in keypoint_groups.values()):
        raise ValueError("keypoint_groups must not list any bodypart more than once.")


def fit_pca_by_keypoint_group(
    Y: NDArray,
    mask: NDArray,
    anterior_idxs,
    posterior_idxs,
    use_bodyparts: Sequence[str],
    keypoint_groups: dict[str, list[str]],
    latent_dims: dict[str, int],
    conf: NDArray | None = None,
    conf_threshold: float = 0.5,
    PCA_fitting_num_frames: int = 1_000_000,
    verbose: bool = False,
    **kwargs,
) -> PCA:
    """
    Fit one PCA per keypoint group and assemble them into a single
    block-diagonal `sklearn.decomposition.PCA`, in the same embedded space
    as `kpms.fit_pca` (see module docstring).

    Parameters
    ----------
    Y, mask, anterior_idxs, posterior_idxs, conf, conf_threshold:
        Same as `kpms.fit_pca`/`jax_moseq...alignment.fit_pca` -- pass
        `data["Y"]`, `data["mask"]`, `config()["anterior_idxs"]`,
        `config()["posterior_idxs"]`, `data["conf"]`.
    use_bodyparts:
        `config()["use_bodyparts"]`, in order -- the bodypart list `Y`'s
        keypoint axis is indexed by.
    keypoint_groups:
        `{group_name: [bodypart names]}`. Must partition `use_bodyparts`
        exactly (every bodypart in exactly one group).
    latent_dims:
        `{group_name: n_dims}` -- each group's share of the combined latent
        space. Must have the same keys as `keypoint_groups`. The resulting
        PCA's total component count is `sum(latent_dims.values())`; this
        must equal config.yml's `latent_dim` for `fit_ar_model` to pick up
        exactly these components (`run_moseq_pipeline.fit_pca` sets this
        automatically).
    PCA_fitting_num_frames:
        As in `kpms.fit_pca` -- max frames sampled (per group) to fit each
        group's PCA. Same fixed seed (42) as kpms's own `fit_pca` for
        reproducibility.

    Returns
    -------
    pca: sklearn.decomposition.PCA
        `components_` ordered by group, in the order `keypoint_groups` is
        given (each group's own components ordered by that group's
        explained variance), each row normalized so its score has unit
        variance within its own group's fit. `explained_variance_ratio_` is
        each component's fraction of *its own group's* total variance, not
        comparable across groups (that comparability is exactly what this
        function removes) -- it's cosmetic only, for `plot_scree`/
        `print_dims_to_explain_variance`.
    """
    from jax_moseq.models.keypoint_slds.alignment import center_embedding, preprocess_for_pca

    _validate_groups(use_bodyparts, keypoint_groups)
    if set(latent_dims.keys()) != set(keypoint_groups.keys()):
        raise ValueError(
            f"latent_dims keys {sorted(latent_dims)} must match "
            f"keypoint_groups keys {sorted(keypoint_groups)}."
        )

    k = len(use_bodyparts)
    d = Y.shape[-1]
    name_to_index = {name: i for i, name in enumerate(use_bodyparts)}

    # Reuse kpms's own alignment + outlier interpolation exactly (translation
    # + yaw removal, then change of basis to the (k-1)*d embedded space), so
    # this is bit-for-bit consistent with what the rest of the pipeline
    # recomputes from Y/mask later. Then decode back to real (k, d) keypoint
    # space -- lossless, since Y_aligned is already zero-mean across all k
    # keypoints (translation was removed) and Gamma @ Gamma_inv is identity
    # on that subspace.
    Y_flat, _, _ = preprocess_for_pca(
        Y, anterior_idxs, posterior_idxs, conf, conf_threshold, fix_heading=False, verbose=verbose
    )
    Gamma = np.array(center_embedding(k))  # (k, k-1)
    Gamma_inv = Gamma.T  # (k-1, k) -- encodes (k, d) -> (k-1, d)

    mask = np.asarray(mask)
    valid = mask > 0
    Y_flat_valid = np.asarray(Y_flat)[valid]  # (N_valid, (k-1)*d)
    Y_embedded_valid = Y_flat_valid.reshape(-1, k - 1, d)
    Y_aligned_valid = np.einsum("ij,njd->nid", Gamma, Y_embedded_valid)  # (N_valid, k, d)

    rng = np.random.default_rng(42)
    mean_full = np.zeros((k, d))
    component_rows_full: list[NDArray] = []
    explained_variance = []
    explained_variance_ratio = []

    for group_name, group_bodyparts in keypoint_groups.items():
        idx = [name_to_index[bp] for bp in group_bodyparts]
        n_dims = latent_dims[group_name]
        Y_group = Y_aligned_valid[:, idx, :].reshape(len(Y_aligned_valid), -1)  # (N_valid, k_g*d)

        n_sample = min(PCA_fitting_num_frames, Y_group.shape[0])
        sample = rng.choice(Y_group.shape[0], n_sample, replace=False)
        group_pca = PCA(random_state=42).fit(Y_group[sample])

        if n_dims > group_pca.components_.shape[0]:
            raise ValueError(
                f"Group '{group_name}' has only {group_pca.components_.shape[0]} "
                f"components available but latent_dims requested {n_dims}."
            )

        mean_full[idx] = group_pca.mean_.reshape(len(idx), d)

        comps = group_pca.components_[:n_dims]  # (n_dims, k_g*d)
        var = group_pca.explained_variance_[:n_dims]
        comps_unit_variance = comps / np.sqrt(var)[:, None]

        for row in comps_unit_variance:
            full_row = np.zeros((k, d))
            full_row[idx] = row.reshape(len(idx), d)
            # Project out this component's own mean across ALL k keypoints
            # (not just its group) -- required for the embedding step below
            # to preserve it exactly; see module docstring.
            full_row -= full_row.mean(axis=0, keepdims=True)
            component_rows_full.append(full_row)

        explained_variance.extend(var.tolist())
        explained_variance_ratio.extend((var / group_pca.explained_variance_.sum()).tolist())

    components_full = np.stack(component_rows_full, axis=0)  # (total_latent_dim, k, d)
    components_embedded = np.einsum("ij,pjd->pid", Gamma_inv, components_full).reshape(
        len(component_rows_full), -1
    )
    mean_embedded = (Gamma_inv @ mean_full).reshape(-1)

    total_latent_dim = len(component_rows_full)
    n_fit_sample = min(PCA_fitting_num_frames, Y_flat_valid.shape[0])
    fit_sample = rng.choice(Y_flat_valid.shape[0], n_fit_sample, replace=False)
    # Fit a real PCA on the actual embedded data purely to populate sklearn's
    # internal fitted-state attributes (n_features_in_, noise_variance_,
    # etc.) with the right shapes; components_/mean_/explained_variance_* are
    # then overwritten below with our block-diagonal decomposition.
    pca = PCA(n_components=total_latent_dim, random_state=42).fit(Y_flat_valid[fit_sample])
    pca.components_ = components_embedded.astype(pca.components_.dtype)
    pca.mean_ = mean_embedded.astype(pca.mean_.dtype)
    pca.explained_variance_ = np.array(explained_variance)
    pca.explained_variance_ratio_ = np.array(explained_variance_ratio)

    return pca
