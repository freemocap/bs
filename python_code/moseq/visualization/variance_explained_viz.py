"""
Variance-explained diagnostics for a fitted kp-moseq model
============================================================

`fit_pca` (see `run_moseq_pipeline.py`) already reports how much keypoint
variance is captured by the low-dimensional continuous pose representation
(`pca_scree.pdf`, `kpms.print_dims_to_explain_variance`) -- that's a property
of the data and the chosen `latent_dimension`, not of the fitted
syllable/AR-HMM model itself.

This module answers the model-fit question instead: given the syllable
labels (z) and switching AR dynamics the model learned, how much of the
variance in the continuous pose trajectory (x, in PCA-latent space) do they
actually account for?

Two complementary measures, computed per latent dimension and pooled across
all valid frames of all sequences:

`compute_ar_r2`
    One-step-ahead AR prediction R^2: at each frame, predict x[t] from its
    `nlags` lagged history using the AR matrix of the syllable active at
    that frame, `Ab[z[t]]`, and compare to the actual x[t].
    R^2 = 1 - SS_residual / SS_total. This credits the model for both
    *which* syllable is active and the shape of that syllable's dynamics,
    so it's the more complete "variance explained by the behavior output"
    number.

`compute_syllable_eta_squared`
    Between-syllable eta^2: how much of the variance in x is explained
    just by which syllable is active (per-syllable mean vs. grand mean),
    ignoring within-syllable dynamics/trajectory shape. Cheaper to reason
    about, and pairs with `ar_r2` as a sanity check -- `ar_r2` should
    exceed `eta_squared` unless the fitted dynamics contribute nothing
    beyond a per-syllable offset.

Entry point
-----------
    from python_code.moseq.visualization.variance_explained_viz import (
        plot_variance_explained,
    )

    stats, fig = plot_variance_explained(project_dir, model_name)
    # stats["ar_r2"]["per_dim"], stats["ar_r2"]["pooled"]
    # stats["eta_squared"]["per_dim"], stats["eta_squared"]["pooled"]

Called automatically at the end of `run_pipeline` in `run_moseq_pipeline.py`,
right after `extract_results`.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from jax_moseq.utils import apply_affine
from jax_moseq.utils.autoregression import get_lags, get_nlags


def _ar_predictable_region(model, data):
    """
    Pull out the continuous latent trajectory, syllable labels, AR matrices,
    and a frame mask, all aligned to the region where one-step AR
    predictions exist -- i.e. after dropping each sequence's first `nlags`
    frames, which have no history to condition on. (kpms's `extract_results`
    pads `z` by `nlags` frames to align it with `x` for saving; here we do
    the opposite and trim `x` to align with the unpadded `z`.)

    Returns
    -------
    x, z, Ab, mask, nlags
        `x`: (n_seq, T - nlags, latent_dim) actual latent state
        `z`: (n_seq, T - nlags) syllable labels
        `Ab`: (n_states, latent_dim, latent_dim * nlags + 1) AR matrices
        `mask`: (n_seq, T - nlags) validity mask
    """
    x = np.asarray(model["states"]["x"])  # (n_seq, T, latent_dim)
    z = np.asarray(model["states"]["z"])  # (n_seq, T - nlags)
    Ab = np.asarray(model["params"]["Ab"])  # (n_states, latent_dim, latent_dim*nlags + 1)
    mask = np.asarray(data["mask"])  # (n_seq, T)

    nlags = get_nlags(Ab)
    assert x.shape[1] - z.shape[1] == nlags, (
        f"Unexpected shapes: x has {x.shape[1]} frames, z has {z.shape[1]}, "
        f"but Ab implies nlags={nlags}."
    )
    return x, z, Ab, mask[:, nlags:], nlags


def compute_ar_r2(model, data) -> dict:
    """
    One-step-ahead AR prediction R^2, per latent dimension and pooled.

    At each valid frame t, predicts x[t] from x[t-nlags:t] using the AR
    matrix of the syllable active at that frame, Ab[z[t]], and compares
    the prediction to the actual x[t].

    Returns
    -------
    dict with keys "per_dim" (array, shape (latent_dim,)) and "pooled" (float).
    """
    x, z, Ab, mask, nlags = _ar_predictable_region(model, data)

    x_lags = get_lags(x, nlags)  # (n_seq, T - nlags, latent_dim * nlags)
    Ab_z = Ab[z]  # (n_seq, T - nlags, latent_dim, latent_dim*nlags + 1)
    x_pred = np.asarray(apply_affine(x_lags, Ab_z))  # (n_seq, T - nlags, latent_dim)
    x_actual = x[:, nlags:]  # (n_seq, T - nlags, latent_dim)

    keep = mask.astype(bool)
    resid = (x_actual - x_pred)[keep]  # (n_valid, latent_dim)
    actual = x_actual[keep]  # (n_valid, latent_dim)

    ss_res = np.sum(resid**2, axis=0)
    ss_tot = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    per_dim = 1 - ss_res / ss_tot
    pooled = 1 - ss_res.sum() / ss_tot.sum()
    return {"per_dim": per_dim, "pooled": float(pooled)}


def compute_syllable_eta_squared(model, data) -> dict:
    """
    Between-syllable eta^2, per latent dimension and pooled: how much of
    the variance in x is explained by syllable identity alone (per-syllable
    mean vs. grand mean), ignoring within-syllable dynamics/trajectory
    shape. Computed over the same AR-predictable region as `compute_ar_r2`
    so the two numbers are directly comparable.

    Returns
    -------
    dict with keys "per_dim" (array, shape (latent_dim,)) and "pooled" (float).
    """
    x, z, _, mask, nlags = _ar_predictable_region(model, data)
    x_actual = x[:, nlags:]

    keep = mask.astype(bool)
    actual = x_actual[keep]  # (n_valid, latent_dim)
    labels = z[keep]  # (n_valid,)

    grand_mean = actual.mean(axis=0)
    ss_tot = np.sum((actual - grand_mean) ** 2, axis=0)

    ss_between = np.zeros_like(ss_tot)
    for k in np.unique(labels):
        in_k = labels == k
        syllable_mean = actual[in_k].mean(axis=0)
        ss_between += in_k.sum() * (syllable_mean - grand_mean) ** 2

    per_dim = ss_between / ss_tot
    pooled = float(ss_between.sum() / ss_tot.sum())
    return {"per_dim": per_dim, "pooled": pooled}


def plot_variance_explained(
    project_dir,
    model_name,
    model=None,
    data=None,
    savefig=True,
    fig_size=(5, 3),
):
    """
    Compute `compute_ar_r2` and `compute_syllable_eta_squared` for a
    checkpoint and plot them side by side as a per-latent-dimension bar
    chart, with pooled values in the title.

    Parameters
    ----------
    project_dir, model_name:
        Used to load the most recent checkpoint if `model`/`data` aren't
        passed, and as the save location for `variance_explained.pdf`.
    model, data:
        Optionally pass an already-loaded checkpoint (as returned by
        `kpms.load_checkpoint`) to avoid loading it again.
    savefig:
        If True, save the figure to
        `{project_dir}/{model_name}/variance_explained.pdf`.
    fig_size:
        Size of the figure in inches.

    Returns
    -------
    stats: dict
        {"ar_r2": {...}, "eta_squared": {...}}, each as returned by the
        corresponding `compute_*` function above.
    fig: matplotlib.figure.Figure
    """
    import keypoint_moseq as kpms  # deferred: heavy, JAX-backed import

    if model is None or data is None:
        model, data, _, _ = kpms.load_checkpoint(str(project_dir), model_name)

    ar_r2 = compute_ar_r2(model, data)
    eta_sq = compute_syllable_eta_squared(model, data)

    latent_dim = len(ar_r2["per_dim"])
    dims = np.arange(latent_dim)
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(dims - width / 2, ar_r2["per_dim"], width, label="AR one-step R²")
    ax.bar(dims + width / 2, eta_sq["per_dim"], width, label="syllable η²")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(dims)
    ax.set_xticklabels([f"PC{d + 1}" for d in dims])
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("variance explained")
    ax.set_title(
        f"pooled: AR R²={ar_r2['pooled']:.2f}, "
        f"η²={eta_sq['pooled']:.2f}"
    )
    ax.legend()
    ax.grid(axis="y")
    fig.set_size_inches(fig_size)
    fig.tight_layout()

    if savefig:
        path = os.path.join(str(project_dir), model_name, "variance_explained.pdf")
        fig.savefig(path)

    return {"ar_r2": ar_r2, "eta_squared": eta_sq}, fig


if __name__ == "__main__":
    # Only needs the checkpoint (states + AR params) and data mask -- no
    # coordinates/loader required -- so it can be pointed at any already-fit
    # model directly, unlike `replot_session.py`'s trajectory
    # plots/grid movies.
    stats, _ = plot_variance_explained(
        project_dir="/home/scholab/moseq/head_with_pupil_points_405_407_test",
        model_name="2026_09_23-01_08_09",
    )
    print(f"AR one-step R² (pooled): {stats['ar_r2']['pooled']:.3f}")
    print(f"AR one-step R² (per dim): {stats['ar_r2']['per_dim']}")
    print(f"syllable η² (pooled): {stats['eta_squared']['pooled']:.3f}")
    print(f"syllable η² (per dim): {stats['eta_squared']['per_dim']}")
