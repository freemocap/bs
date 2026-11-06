"""
Pre-AR-HMM eye_scale (alpha) sweep for `kpms_loader.load_head_with_scaled_eyes_kpms`.

Cheap check (minutes, no AR-HMM) for choosing `eye_scale` before fitting a
model -- see `eye_movement_in_pcs.md`.

    python -m python_code.moseq.eye_scale_sweep 405_407
    python -m python_code.moseq.eye_scale_sweep 2025
    python -m python_code.moseq.eye_scale_sweep 18_sessions 3 10 30 50

Optional trailing args override the alphas to sweep.

For each alpha: kpms outlier removal + yaw alignment (preprocess_for_pca), fit a
standard joint PCA, then score the top-N PCs on
  - eye-in-head angle decodability (4 channels)
  - head pitch/roll decodability, and head-keypoint variance reconstructed
using within-recording, time-blocked 5-fold CV linear decoding.
"""
import json
import sys
from pathlib import Path

import numpy as np
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds.alignment import preprocess_for_pca, center_embedding
from sklearn.decomposition import PCA
from python_code.utilities.folder_utilities.recording_folder import RecordingFolder
from python_code.moseq.kpms_loader import (
    load_head_with_scaled_eyes_kpms, load_eye_in_head_kpms, load_eye_data_quality, _fit_rigid_poses)
from python_code.moseq.training_config import load_training_recording_folders
from python_code.moseq.run_multi_session_kappa_sweeps import KEEP_FRAMES, SWEEPS, recording_folders
from python_code.moseq.utils.bodyparts import (
    HEAD_BODYPARTS, HEAD_WITH_PUPIL_POINTS_BODYPARTS as BODYPARTS,
    HEAD_WITH_PUPIL_POINTS_LEFT_EYE_BODYPARTS as LEFT, HEAD_WITH_PUPIL_POINTS_RIGHT_EYE_BODYPARTS as RIGHT)
from python_code.kinematics_core.reference_geometry_model import ReferenceGeometry

DATASET = sys.argv[1]
ALPHAS = [float(a) for a in sys.argv[2:]] or [1.0, 3.0, 10.0, 30.0]
OUT = Path("/home/scholab/moseq/eye_scale_sweep") / DATASET; OUT.mkdir(parents=True, exist_ok=True)
if DATASET == "405_407":
    folders = load_training_recording_folders("/home/scholab/moseq/head_with_pupil_points_405_407_test")
elif DATASET == "18_sessions":
    # Same exclusions as the 18-session eye-only sweep (no usable left-eye data).
    folders = recording_folders(SWEEPS["eyes"]["exclude_sessions"])
else:
    # session_2025-07-05_ferret_753 excluded: no usable left-eye data, which
    # dominates the top PCs as a rigid shift of that eye's points.
    folders = [RecordingFolder.from_folder_path(f"/mnt/data/ferret_recordings/{n}/full_recording") for n in [
        "session_2025-07-05_ferret_757_EyeCameras_P37_EO9",
        "session_2025-07-07_ferret_753_EyeCameras_P39_E11", "session_2025-07-07_ferret_757_EyeCameras_P39_E11"]]


def trim(d):
    return {n: v[slice(*KEEP_FRAMES[n])] if n in KEEP_FRAMES else v for n, v in d.items()}


cfg = kpms.load_config("/home/scholab/moseq/head_with_pupil_points_405_407_test")
k = len(BODYPARTS)
ant = [BODYPARTS.index(b) for b in cfg["anterior_bodyparts"]]
post = [BODYPARTS.index(b) for b in cfg["posterior_bodyparts"]]
head_idx = [BODYPARTS.index(b) for b in HEAD_BODYPARTS]
eye_idx = [BODYPARTS.index(b) for b in LEFT + RIGHT]
Gamma = np.array(center_embedding(k))

# ground truth: eye-in-head angles + head pitch/roll
eih_l, eih_r = load_eye_in_head_kpms(folders)
eye_truth, head_truth = {}, {}
eih_l, eih_r = trim(eih_l), trim(eih_r)
base_coords, _, base_bps = load_head_with_scaled_eyes_kpms(folders, eye_scale=1.0)
base_coords = trim(base_coords)
for rf in folders:
    n = rf.recording_name
    gl, gr = (trim({n: g})[n] for g in load_eye_data_quality(rf, "medium"))
    l, r = eih_l[n].astype(float).copy(), eih_r[n].astype(float).copy()
    l[~gl] = np.nan; r[~gr] = np.nan
    eye_truth[n] = np.c_[l, r]
    geo = ReferenceGeometry.from_json_file(rf.skull_kinematics / "skull_reference_geometry.json")
    H = base_coords[n][:, :8]
    R, _ = _fit_rigid_poses(np.stack([geo.get_keypoint_position(b) for b in base_bps[:8]]), H)
    pitch = np.degrees(np.arcsin(np.clip(-R[:, 2, 0], -1, 1)))
    roll = np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    head_truth[n] = np.c_[pitch, roll]
recs = [rf.recording_name for rf in folders]
for n in recs:
    print(n, "eye finite:", np.round(np.isfinite(eye_truth[n]).mean(0), 2),
          "pitch/roll std deg:", np.round(np.nanstd(head_truth[n], 0), 1), flush=True)


def cv_r2(feat, truth, nfold=5):
    out = []
    for c in range(next(iter(truth.values())).shape[1]):
        ss_res = ss_tot = 0.0
        for r in recs:
            X, y = feat[r], truth[r][:, c]
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            for f in np.array_split(np.arange(len(y)), nfold):
                te = np.zeros(len(y), bool); te[f] = True
                tr, te = ok & ~te, ok & te
                if tr.sum() < 100 or te.sum() < 10: continue
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                w, *_ = np.linalg.lstsq(np.c_[(X[tr] - mu) / sd, np.ones(tr.sum())], y[tr], rcond=None)
                pred = np.c_[(X[te] - mu) / sd, np.ones(te.sum())] @ w
                ss_res += ((y[te] - pred) ** 2).sum(); ss_tot += ((y[te] - y[te].mean()) ** 2).sum()
        out.append(float(1 - ss_res / ss_tot))
    return out


results = {}
for alpha in ALPHAS:
    coords, confs, bps = load_head_with_scaled_eyes_kpms(folders, eye_scale=alpha)
    coords, confs = trim(coords), trim(confs)
    order = [bps.index(b) for b in BODYPARTS]
    coords = {n: c[:, order] for n, c in coords.items()}
    confs = {n: c[:, order] for n, c in confs.items()}
    scratch = OUT / f"alpha_{alpha:g}"; scratch.mkdir(exist_ok=True)
    conf_before = {n: (c > 0).mean() for n, c in confs.items()}
    coords, confs = kpms.outlier_removal(coords, confs, str(scratch), overwrite=False,
                                         outlier_scale_factor=6.0, bodyparts=BODYPARTS)
    flagged = {n: float(conf_before[n] - (confs[n] > 0).mean()) for n in recs}
    flats, aligned = {}, {}
    for n in recs:
        Yf, _, _ = preprocess_for_pca(coords[n][None], ant, post, confs[n][None], cfg["conf_threshold"])
        flats[n] = np.asarray(Yf)[0]
        aligned[n] = np.einsum("ij,tjd->tid", Gamma, flats[n].reshape(len(flats[n]), k - 1, 3))
    allf = np.concatenate([flats[n] for n in recs]); allf = allf[np.isfinite(allf).all(1)]
    rng = np.random.default_rng(42)
    pca = PCA(random_state=42).fit(allf[rng.choice(len(allf), min(len(allf), 1_000_000), replace=False)])
    res = {"outlier_frac_flagged": flagged, "evr_top10": float(pca.explained_variance_ratio_[:10].sum())}
    # per-PC loading energy on eye keypoints (top 10)
    comps_kp = np.einsum("ij,pjd->pid", Gamma, pca.components_[:10].reshape(10, k - 1, 3))
    energy = (comps_kp ** 2).sum(-1)
    res["eye_loading_frac_per_pc"] = (energy[:, eye_idx].sum(1) / energy.sum(1)).round(3).tolist()
    for n_pc in (10, 20):
        scores = {n: pca.transform(flats[n])[:, :n_pc] for n in recs}
        recon = {n: np.einsum("ij,tjd->tid", Gamma, (scores[n] @ pca.components_[:n_pc] + pca.mean_).reshape(-1, k - 1, 3)) for n in recs}
        # head keypoint variance reconstructed (in-sample, pooled)
        Ht = np.concatenate([aligned[n][:, head_idx] for n in recs]); Hr = np.concatenate([recon[n][:, head_idx] for n in recs])
        ok = np.isfinite(Ht).all((1, 2))
        head_recon = 1 - ((Ht[ok] - Hr[ok]) ** 2).sum() / ((Ht[ok] - Ht[ok].mean(0)) ** 2).sum()
        res[f"top{n_pc}"] = {
            "eye_angle_r2": cv_r2(scores, eye_truth),
            "head_pitch_roll_r2": cv_r2(scores, head_truth),
            "head_keypoint_var_reconstructed": float(head_recon),
        }
        if n_pc == 10:
            Tt = np.concatenate([eye_truth[n] for n in recs]); S = np.concatenate([scores[n] for n in recs])
            best = []
            for j in range(10):
                r2s = [np.corrcoef(S[v, j], Tt[v, c])[0, 1] ** 2 for c in range(4) for v in [np.isfinite(S[:, j]) & np.isfinite(Tt[:, c])]]
                best.append(round(float(max(r2s)), 3))
            res["best_single_pc_eye_r2"] = best
    results[alpha] = res
    print(f"\nalpha={alpha:g}", json.dumps(res, indent=1), flush=True)

json.dump(results, open(OUT / "results.json", "w"), indent=1)
print("\nSUMMARY", DATASET)
print(f"{'alpha':>6} {'nPC':>4} | {'L add':>6} {'L elv':>6} {'R add':>6} {'R elv':>6} | {'pitch':>6} {'roll':>6} | {'headVar':>7}")
for a, r in results.items():
    for n_pc in (10, 20):
        t = r[f"top{n_pc}"]
        print(f"{a:6g} {n_pc:4d} | " + " ".join(f"{x:6.3f}" for x in t["eye_angle_r2"]) + " | "
              + " ".join(f"{x:6.3f}" for x in t["head_pitch_roll_r2"]) + f" | {t['head_keypoint_var_reconstructed']:7.3f}")
