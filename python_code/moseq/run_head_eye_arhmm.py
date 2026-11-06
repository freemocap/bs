"""
Joint head + eye AR-HMM
=======================

Fits one sticky HDP AR-HMM directly on the 9 kinematic channels from
`head_eye_features.py`, so head and eye dynamics share one syllable sequence
with no keypoint PCA in between (`eye_movement_in_pcs.md`, Option C).

`kpms.fit_model(ar_only=True)` only ever calls `arhmm.resample_model`, which
never touches `x`, so the channels go straight into `states["x"]` and kpms does
the fitting. `states["v"]`/`["h"]` are the real head centroid and heading, so
checkpoints and `results.h5` have the usual schema and `plot_variance_explained`,
`eye_syllable_viz`, grid movies and `moseq_model_viewer` all work on these models
(`training_recordings.yaml` records the `head` loader for their keypoints).

Each model trains on the first `TRAIN_FRACTION` of every session. The rest is
held out for `heldout_ll_per_frame` in `sweep_summary.csv`, then the fitted
model labels every frame, so results cover the whole recording.

`--shifted-control` fits the same grid with the eye block circularly shifted
against the head block inside the training window only
(`head_eye_features.shift_eye_block`), so both fits are scored on identical
held-out frames. Held-out likelihood, real minus shifted, is the test of
whether the joint model finds head-eye coupling beyond what each block does
alone. Control fits with whole-session shifts are not comparable.

Run in the `keypoint_moseq` env:

    python -m python_code.moseq.run_head_eye_arhmm pilot
    python -m python_code.moseq.run_head_eye_arhmm pilot --shifted-control
    python -m python_code.moseq.run_head_eye_arhmm full --kappas 1e4 --nlags 8 --seeds 0 1 2

Fits already listed in `sweep_summary.csv` are skipped, so rerun to resume.
"""
import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import keypoint_moseq as kpms
import numpy as np
import pandas as pd
from jax_moseq.models import arhmm
from jax_moseq.models.keypoint_slds.alignment import vector_to_angle
from jax_moseq.utils import batch

from python_code.moseq.head_eye_features import (
    NOISE_FLOOR_SD,
    HeadEyeFeatures,
    add_noise_floor,
    drop_pupil,
    load_head_eye_features,
    shift_eye_block,
    training_mask,
    use_head_rates,
)
from python_code.moseq.kpms_loader import load_head_kpms
from python_code.moseq.run_moseq_pipeline import (
    _KPMS_MIN_FREQUENCY,
    _LOADER_CONFIG,
    KPMS_Loader,
    _format_duration,
    _resolve_registered_loader,
    _syllable_usage_stats,
    configure_project,
    extract_results,
    variance_explained,
)
from python_code.moseq.run_multi_session_kappa_sweeps import (
    KEEP_FRAMES,
    RECORDINGS_ROOT,
    SESSION_NAMES,
)
from python_code.moseq.utils.bodyparts import HEAD_BODYPARTS
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder

FPS = 120
TRAIN_FRACTION = 0.8
SEG_LENGTH = 10000
SEG_OVERLAP = 30
# Fit in chunks of this many iterations, with a held-out likelihood after each,
# so `heldout_trace.json` shows whether a fit converged.
TRACE_EVERY_N_ITERS = 50

# kpms defaults (see any project's config.yml); only kappa and nlags vary.
TRANS_HYPPARAMS = {"num_states": 100, "gamma": 1e3, "alpha": 5.7}
AR_HYPPARAMS = {"S_0_scale": 0.01, "K_0_scale": 10.0}

# Left eye passes the medium eye-data-quality threshold on only 0.1% of frames.
_EXCLUDED_SESSIONS = ["session_2025-07-03_ferret_753_EyeCameras_P35_EO7"]

SWEEPS = {
    "pilot": {
        "project_dir": "/mnt/data/moseq/head_eye_arhmm_405_407_pilot",
        "sessions": [name for name in SESSION_NAMES if "ferret_405" in name or "ferret_407" in name],
        # Short end: eye movements are the focus. 1e5 gives ~230 ms syllables.
        "kappas": [1e2, 1e3, 1e4, 1e5, 1e6],
        # 3 lags is the kpms default but only 25 ms at 120 Hz; the eye lags
        # head yaw by 17-83 ms, which 8 lags (67 ms) mostly covers.
        "nlags": [3, 8],
        "seeds": [0],
        "num_iters": 300,
    },
    "full": {
        "project_dir": "/mnt/data/moseq/head_eye_arhmm_17_sessions",
        "sessions": [name for name in SESSION_NAMES if name not in _EXCLUDED_SESSIONS],
        "kappas": [1e2, 1e3, 1e4, 1e5, 1e6],
        "nlags": [3, 8],
        "seeds": [0, 1, 2],
        "num_iters": 500,
    },
}


def _head_centroids_and_headings(
    recording_folders: list[RecordingFolder],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Head keypoint centroid and heading per recording, computed as in
    `align_egocentric` but in numpy, since the keypoints aren't modeled."""
    coordinates, _, bodyparts = load_head_kpms(recording_folders)
    head = _LOADER_CONFIG[KPMS_Loader.HEAD]
    anterior_idxs = [bodyparts.index(bp) for bp in head["anterior_bodyparts"]]
    posterior_idxs = [bodyparts.index(bp) for bp in head["posterior_bodyparts"]]

    centroids, headings = {}, {}
    for name, coords in coordinates.items():
        centroids[name] = coords.mean(axis=1)
        forward = coords[:, anterior_idxs, :2].mean(axis=1) - coords[:, posterior_idxs, :2].mean(axis=1)
        headings[name] = np.asarray(vector_to_angle(forward))
    return centroids, headings


def _batched(by_recording: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, tuple]:
    return batch(by_recording, keys=sorted(by_recording), seg_length=SEG_LENGTH, seg_overlap=SEG_OVERLAP)


def _train_and_heldout_masks(
    features: HeadEyeFeatures, nlags: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-recording boolean masks: frames to train on, frames to score on,
    and all usable frames (which get labels). Each excludes frames with
    invalid AR history (`training_mask`)."""
    train, heldout, usable = {}, {}, {}
    for name, valid in features.valid.items():
        usable[name] = training_mask(valid, nlags)
        split = int(TRAIN_FRACTION * len(valid))
        train[name] = usable[name].copy()
        train[name][split:] = False
        heldout[name] = usable[name].copy()
        heldout[name][: split + nlags] = False
    return train, heldout, usable


def _heldout_ll_per_frame(model: dict, x: jnp.ndarray, heldout_mask: jnp.ndarray, nlags: int) -> float:
    params = model["params"]
    ll = arhmm.marginal_log_likelihood(heldout_mask, x, params["Ab"], params["Q"], params["pi"])
    return float(ll / heldout_mask[:, nlags:].sum())


def fit_one(
    project_dir: Path,
    model_name: str,
    features: HeadEyeFeatures,
    centroids: dict[str, np.ndarray],
    headings: dict[str, np.ndarray],
    kappa: float,
    nlags: int,
    seed: int,
    num_iters: int,
) -> float:
    """
    Fit one model, label every frame with it, and save a kpms checkpoint and
    results under `{project_dir}/{model_name}`.

    Returns
    -------
    Held-out log likelihood per frame.
    """
    x_batched, padding, metadata = _batched(features.features)
    train, heldout, usable = _train_and_heldout_masks(features, nlags)
    to_mask = lambda masks: jnp.asarray(_batched(masks)[0] * padding, dtype=float)
    train_mask, heldout_mask, usable_mask = to_mask(train), to_mask(heldout), to_mask(usable)

    x = jnp.asarray(x_batched, dtype=float)
    model = arhmm.init_model(
        {"x": x, "mask": train_mask},
        trans_hypparams={**TRANS_HYPPARAMS, "kappa": kappa},
        ar_hypparams={**AR_HYPPARAMS, "latent_dim": x.shape[-1], "nlags": nlags},
        seed=jr.PRNGKey(seed),
    )
    model["states"].update(
        x=x,
        v=jnp.asarray(_batched(centroids)[0], dtype=float),
        h=jnp.asarray(_batched(headings)[0], dtype=float),
        # Unused with ar_only, but keypoint_slds.resample_model takes them.
        s=jnp.ones(x.shape[:2] + (1,)),
    )
    model["noise_prior"] = jnp.ones(1)

    data = {"mask": train_mask}
    trace = {}
    for start in range(0, num_iters, TRACE_EVERY_N_ITERS):
        stop = min(start + TRACE_EVERY_N_ITERS, num_iters)
        model, _ = kpms.fit_model(
            model, data, metadata, str(project_dir), model_name,
            ar_only=True, start_iter=start, num_iters=stop - 1,
            save_every_n_iters=None, generate_progress_plots=False,
        )
        trace[stop] = _heldout_ll_per_frame(model, x, heldout_mask, nlags)
        print(f"{model_name} iter {stop}: held-out LL {trace[stop]:.4f} nats/frame")

    # Label held-out frames too; parameters stay as fit on the training frames.
    model["states"]["z"] = arhmm.resample_discrete_stateseqs(
        model["seed"], x, usable_mask, **model["params"]
    )

    save_dir = project_dir / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    kpms.save_hdf5(
        str(save_dir / "checkpoint.h5"),
        {
            "model_snapshots": {str(num_iters): model},
            "metadata": metadata,
            "data": {"mask": usable_mask},
        },
    )
    with open(save_dir / "heldout_trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    return trace[max(trace)]


def run_sweep(
    project_dir: Path,
    session_names: list[str],
    kappas: list[float],
    nlags_values: list[int],
    seeds: list[int],
    num_iters: int,
    shifted_control: bool = False,
    noise_sd: float = NOISE_FLOOR_SD,
    no_pupil: bool = False,
    head_rates: bool = False,
) -> pd.DataFrame:
    """Fit every kappa x nlags x seed combination, rewriting
    `sweep_summary.csv` after each so an interrupted sweep keeps its results."""
    recording_folders = [
        RecordingFolder.from_folder_path(RECORDINGS_ROOT / name / "full_recording") for name in session_names
    ]
    head = _LOADER_CONFIG[KPMS_Loader.HEAD]
    _, _, video_dir = _resolve_registered_loader(project_dir, recording_folders, KPMS_Loader.HEAD)
    configure_project(
        project_dir,
        video_dir=video_dir,
        bodyparts=HEAD_BODYPARTS,
        use_bodyparts=head["use_bodyparts"],
        anterior_bodyparts=head["anterior_bodyparts"],
        posterior_bodyparts=head["posterior_bodyparts"],
        fps=FPS,
        skeleton=head["skeleton"],
    )

    features = load_head_eye_features(recording_folders, keep_frames=KEEP_FRAMES)
    if head_rates:
        features = use_head_rates(features, recording_folders)
    features = add_noise_floor(features, noise_sd)
    if no_pupil:
        features = drop_pupil(features)
    centroids, headings = _head_centroids_and_headings(recording_folders)
    pd.DataFrame({"channel": features.channels, "scale": features.scales}).to_csv(
        project_dir / "feature_scales.csv", index=False
    )

    summary_path = project_dir / "sweep_summary.csv"
    rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    finished = {row["model_name"] for row in rows}

    for kappa in kappas:
        for nlags in nlags_values:
            for seed in seeds:
                prefix = "shifted_" if shifted_control else ""
                model_name = f"{prefix}kappa_{kappa:.0e}_nlags{nlags}_seed{seed}"
                if model_name in finished:
                    print(f"[skip] {model_name}")
                    continue

                start = time.perf_counter()
                heldout_ll = fit_one(
                    project_dir, model_name,
                    shift_eye_block(features, seed, TRAIN_FRACTION, fps=FPS) if shifted_control else features,
                    centroids, headings, kappa, nlags, seed, num_iters,
                )
                results = extract_results(project_dir, model_name)
                stats = variance_explained(project_dir, model_name)
                fit_seconds = time.perf_counter() - start
                print(f"[done] {model_name}: {_format_duration(fit_seconds)}")

                rows.append({
                    "model_name": model_name,
                    "shifted_control": shifted_control,
                    "kappa": kappa,
                    "nlags": nlags,
                    "seed": seed,
                    "noise_sd": noise_sd,
                    "heldout_ll_per_frame": heldout_ll,
                    **_syllable_usage_stats(results, _KPMS_MIN_FREQUENCY),
                    "ar_r2_pooled": stats["ar_r2"]["pooled"],
                    "eta_sq_pooled": stats["eta_squared"]["pooled"],
                    "fit_seconds": fit_seconds,
                })
                summary = pd.DataFrame(rows)
                summary.to_csv(summary_path, index=False)
                print(summary.to_string(index=False))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", choices=sorted(SWEEPS))
    parser.add_argument("--shifted-control", action="store_true", help="shift eyes against head (control)")
    parser.add_argument("--kappas", type=float, nargs="+", default=None)
    parser.add_argument("--nlags", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--noise-sd", type=float, default=NOISE_FLOOR_SD)
    parser.add_argument("--head-rates", action="store_true", help="use pitch and roll rates, not angles")
    parser.add_argument("--no-pupil", action="store_true", help="fit without the pupil channels")
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--project-dir", default=None, help="override the sweep's project dir")
    parser.add_argument("--sessions", type=int, default=None, help="use only the first N sessions")
    args = parser.parse_args()

    sweep = SWEEPS[args.sweep]
    run_sweep(
        project_dir=Path(args.project_dir or sweep["project_dir"]),
        session_names=sweep["sessions"][: args.sessions],
        kappas=args.kappas or sweep["kappas"],
        nlags_values=args.nlags or sweep["nlags"],
        seeds=args.seeds or sweep["seeds"],
        num_iters=args.num_iters or sweep["num_iters"],
        shifted_control=args.shifted_control,
        noise_sd=args.noise_sd,
        no_pupil=args.no_pupil,
        head_rates=args.head_rates,
    )
