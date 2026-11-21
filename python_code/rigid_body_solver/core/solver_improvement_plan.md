# Rigid Body Solver — Improvement Plan

Source of these findings: review of `python_code/rigid_body_solver/core/{optimization.py, main_solver_interface.py, calculate_reference_geometry.py}`.
Excludes the quaternion double-cover discontinuity issue (acceptable for normal ferret head motion; revisit only if fast-rotation jumps appear after the items below are fixed).

Ordered by impact on **accuracy and frame-to-frame jumps first**, efficiency work last — efficiency gains don't matter if the output is still biased or unstable.

---

## Baseline jump measurement (pre-implementation, 2026-11-14)

Before implementing any of the fixes below, a jump detector was built
(`python_code/rigid_body_solver/core/jump_detection.py`) and run against every
`skull_kinematics.csv` under
`/Users/philipqueen/Library/CloudStorage/Dropbox/VisBehavDev/data/analyzable_outputs/`,
to quantify the current level of the issue and give something to re-measure
against after each fix lands.

**Method:** per-frame quaternion "triangle discrepancy" —
`angle(q[t-1],q[t]) + angle(q[t],q[t+1]) - angle(q[t-1],q[t+1])`. This isolates
a frame whose solved orientation deviates from the geodesic path through its
immediate neighbors (the "wrong local minimum for one frame" signature),
independent of how fast the body is genuinely rotating. Frames are flagged
when discrepancy exceeds `median + 8×MAD` (floor 2°).

**Results (65 of 66 sessions processed — one, `ferret_7572_..._P43_E15`, has
no kinematics files on disk):**
- 3,854 flagged frames out of 2,336,397 total frames (~0.16% overall).
- 20 of 65 sessions are completely clean (zero flagged frames).
- Highly concentrated, not uniform: the single worst session
  (`2025-07-11 · ferret 757 · P43 E15`) has 745 flagged frames (1.6% of its
  frames); the top 4 sessions account for over half of all flagged frames in
  the dataset.
- Worst sessions by flagged-frame rate per 1000 frames: `ferret_757_P43_E15__1`
  (16.27), `ferret_403_P46_E4` (13.65), `ferret_416_P42_E3` (13.56),
  `ferret_402_E07` (10.64), `ferret_407_E11` (8.58).
- **Open question, not yet explained by this plan:** the 2025 recordings
  (ferret 753/757) are consistently worse than the 2026 recordings (ferret
  403/405/407/411/416/420) — most 2026 sessions are clean or near-clean, while
  most of the bad 2025 sessions land in the top 10 by jump rate. Worth
  checking whether something changed between those batches (camera setup,
  marker placement, topology) independent of whatever items 1-3 below fix.

**Re-measurement:** after each item below lands, re-run
`jump_detection.py` (or the batch script pattern used for this baseline —
iterate `skull_kinematics` dirs, call `detect_orientation_jumps`) across the
same dataset and compare `total_flagged`, `n_clean`, and the worst-session
tail against these numbers.

---

## 1. Replace identity-rotation initialization with per-frame Procrustes/Kabsch

**Why first:** this is the most direct lever on "jumps between frames." Every frame currently starts bundle adjustment from `quat = [1,0,0,0]` (`optimization.py:298`) regardless of the body's actual orientation. Rotation optimization is non-convex; starting far from the true answer on every single frame means Ceres can converge different frames into different local minima before the (comparatively weak, by design) smoothness term pulls them into line — this is exactly the mechanism that produces visible jumps, especially after a frame where the head/body orientation differs substantially from identity.

**Where:** `optimization.py::optimize_rigid_body`, the pose initialization loop (~line 294-302):
```python
for frame_idx in range(n_frames):
    quat_ceres = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    translation = np.nanmean(original_trajectories[frame_idx], axis=0)
    ...
```

**Implementation:**
1. Write a helper `estimate_initial_pose(measured_points, reference_points, valid_mask)` in `optimization.py` (or a new `pose_initialization.py` module next to it) that:
   - Takes the per-frame measured keypoints (with NaNs), the fixed `reference_geometry_array`, and a boolean mask of which keypoints are valid in that frame.
   - Requires at least 3 non-collinear valid points (if fewer, fall back — see step 3).
   - Centers both the valid measured points and the corresponding valid reference points on their own centroids.
   - Computes the optimal rotation via SVD (Kabsch algorithm): `H = ref_centered.T @ measured_centered`; `U, S, Vt = np.linalg.svd(H)`; `d = np.sign(np.linalg.det(Vt.T @ U.T))`; `R = Vt.T @ diag(1,1,d) @ U.T`. The `d` sign-correction is required to avoid returning a reflection instead of a rotation when the point configuration is near-planar or SVD lands on a degenerate case.
   - Converts `R` to a `[w,x,y,z]` quaternion via `scipy.spatial.transform.Rotation.from_matrix(R).as_quat()` (remember to reorder from scipy's `xyzw` to Ceres's `wxyz`, matching the convention already used elsewhere in this file).
   - Translation is then `measured_centroid - R @ reference_centroid` (not the raw marker mean used today — the raw mean is only a translation estimate when rotation is identity, which it generally isn't).
2. Fallback for frames with <3 valid keypoints: carry forward the *previous* frame's solved initial pose (not identity) — a stale-but-plausible orientation is a far better warm start than identity. Track this with a simple "last good pose" variable while iterating `frame_idx` in increasing order.
3. Replace the initialization loop with this per-frame Kabsch estimate, iterating frames in order so the carry-forward fallback has something valid to use.
4. Keep this purely as *initialization* — the existing bundle adjustment (data term + smoothness) still runs afterward and remains the source of truth; this just gives it a much better starting point.

**Test:** on a synthetic rotating rigid body test (see `cube_solver_demo.py` for the existing pattern), verify that initial residual cost (`summary.initial_cost`) is dramatically lower than with identity init, and that frame-to-frame rotation angle change (`Rotation.from_quat` angle between consecutive solved quaternions) has no outlier spikes compared to before the fix.

---

## 2. Add a robust loss function to the measurement (data) term

**Why second:** once initialization (step 1) is fixed, this addresses the remaining source of jumps — a single occluded/mislabeled marker in one frame currently gets full squared-error weight in `MeasurementFactor`, and because rotation/translation are shared across all markers in a frame, one bad detection can drag the whole frame's pose toward a bad fit.

**Where:** `optimization.py::optimize_rigid_body`, residual block construction (~line 326-331):
```python
cost = MeasurementFactor(
    measured_point=measured_point,
    reference_point=reference_geometry_array[keypoint_idx],
    weight=config.lambda_data,
)
problem.add_residual_block(cost, None, [quat, trans])
```
The `None` is the loss function argument.

**Implementation:**
1. Add a `robust_loss_scale: float | None` field to `OptimizationConfig` (default e.g. `None` = disabled, so this is opt-in and doesn't silently change behavior for existing configs/tuned weights — but recommend enabling it as the new default once validated).
2. When set, build `loss = pyceres.HuberLoss(config.robust_loss_scale)` once per call to `optimize_rigid_body` and pass it as the loss argument instead of `None`:
   ```python
   problem.add_residual_block(cost, loss, [quat, trans])
   ```
3. **Picking the scale:** `HuberLoss(a)` transitions from quadratic to linear at residual magnitude `a`. Since `MeasurementFactor`'s residual is already `weight * (measured - predicted)` (i.e. pre-multiplied by `lambda_data`), the Huber scale must be in the *same scaled units* as the residual, not raw mm. Practical approach: set `a = lambda_data * expected_inlier_error_mm`, where `expected_inlier_error_mm` is your typical marker tracking noise (e.g. 1-2mm from triangulation) — so inlier residuals stay in the quadratic region and only genuine outliers (occlusion artifacts, mislabeled markers) get down-weighted. Expose this as `robust_loss_scale` directly in mm-equivalent-of-residual terms in the config docstring so it's tunable per rigid body without needing to know Ceres internals.
4. Do **not** apply a robust loss to the smoothness factors — those are regularization terms, not noisy measurements, and should stay quadratic.
5. **Validation:** re-run on a known problematic recording (one that currently shows jumps) and check two things: (a) `summary.final_cost` doesn't get dramatically worse on clean frames (would indicate the scale is too aggressive), and (b) the frame(s) that previously jumped now track smoothly — inspect per-marker residuals at that frame before/after to confirm an outlier marker was actually down-weighted rather than the jump having a different cause.

**Implementation note (2026-11-15), correcting the scale-picking guidance above:** tried `robust_loss_scale` at `lambda_data * 1.5mm` (150) and `lambda_data * 4mm` (400) on `ferret_757_P43_E15` — both *regressed* jump counts relative to step 1 alone (146 → 394 → 555 flagged frames). Diagnosed by comparing real per-marker residuals (measured markers vs. the solved rigid-body fit's prediction) on flagged vs. non-flagged frames: residuals are typically **70-90mm even on clean, non-flagged frames** (soft-tissue flex / imperfect rigidity, not occlusion noise) — nowhere near the assumed 1-2mm triangulation noise. At both scales tried, virtually every residual on every frame (flagged or not) already sat in the Huber linear region, so the robust loss wasn't selectively suppressing outliers — it was uniformly weakening the data term vs. the smoothness term across the whole recording, which is consistent with the observed regression. **Correct approach:** calibrate `expected_inlier_error_mm` from the actual residual distribution of clean frames for the topology in question (here, ~70-90mm), not a hand-guessed marker-noise number — e.g. `a ≈ lambda_data * 100mm` as a starting point for this skull topology, re-validate per rigid body.

**Outcome:** re-ran at the recalibrated scale (`lambda_data * 100mm` = 10000) — jump count was neutral (159 flagged, vs. step 1 alone's 146), not a clear improvement. This session's jumps appear driven by structural soft-tissue flex (every marker on every frame has large residuals, flagged or not), not isolated outlier/occlusion spikes — the failure mode Huber loss is actually designed to catch. **Status: parked, `robust_loss_scale=None` in `ferret_skull_solver.py` as of 2026-11-15.** Revisit with a session whose jumps are demonstrably occlusion/outlier-driven (check per-marker residuals first, same method as above) before re-enabling. The `OptimizationConfig`/`optimize_rigid_body` plumbing is still correct and in place — just unused in production pending a session where it actually helps.

---

## 3. Add per-keypoint confidence/likelihood weighting to the measurement term

**Why third:** this is a refinement on top of step 2, not a replacement — Huber loss handles the "detected point is a flat-out outlier" case; confidence weighting handles the gentler case where a marker's detection is systematically less trustworthy even when not an outright outlier, e.g. a camera-angle-grazing marker with persistently lower DLC likelihood. This is ordered after the robust loss because it depends on confidence scores actually being threaded through the data pipeline, which may need upstream plumbing (DLC/triangulation output → `load_measured_trajectories_csv` → `optimize_rigid_body`); do steps 1-2 first since they need no new upstream data.

**Where:**
- `data_io/load_measured_trajectories.py` — confirm whether per-keypoint confidence/likelihood is already available in the loaded `trajectory_dict` (DLC typically emits per-point `likelihood` alongside x/y/z). If not present, this needs to be added at the loader level first.
- `optimization.py::MeasurementFactor.__init__` currently takes a single scalar `weight=100.0` applied uniformly.
- `optimization.py::optimize_rigid_body`, the measurement factor construction loop (~line 318-332).

**Implementation:**
1. Extend `optimize_rigid_body`'s signature to optionally accept `confidence: NDArray[np.float64] | None` of shape `(n_frames, n_keypoints)` (values in `[0, 1]`), threaded through from `process_tracking_data` / `RigidBodySolverConfig` if available from the loader.
2. When constructing each `MeasurementFactor`, compute the effective weight as `weight = config.lambda_data * confidence[frame_idx, keypoint_idx]` instead of the flat `config.lambda_data`. When `confidence is None`, preserve current behavior (`weight = config.lambda_data`) so this is backward compatible.
3. Pick a floor (e.g. treat confidence below some threshold, say 0.1-0.2, the same as a missing/NaN point and skip the residual block entirely) — a near-zero-confidence point contributes mostly noise to the solve and isn't worth the residual block.
4. **Interaction with step 2's Huber loss:** since weight now varies per-residual, the Huber scale picked in step 2 should be re-validated — a low-confidence point's residual is scaled down by its weight, so it will sit in the quadratic region more easily than before; this is fine (desired) but worth re-checking the chosen `robust_loss_scale` still makes sense across the confidence range you actually see in data.

**Test:** on a recording with known low-confidence stretches (e.g. markers near occlusion), verify the solved trajectory smooths through that stretch more than with uniform weighting, without needing to crank up `lambda_rot_smooth`/`lambda_trans_smooth` (over-smoothing elsewhere) to compensate.

---

## 4. Replace finite-difference quaternion Jacobian with the analytic derivative

**Why fourth (efficiency, not accuracy-critical):** this affects solve speed and introduces a small, probably-negligible bias from first-order forward differencing, but it does not drive the visible jumps — it's ordered last among the "real" fixes because the previous three items change *what* is being fit; this changes *how fast and how precisely* the same fit converges, and it's easiest to validate against a stable baseline (verify cost curves match the old solver on a reference recording after items 1-3 land, before further perturbing the math).

**Where:** `optimization.py::MeasurementFactor.Evaluate` (~line 113-126):
```python
if jacobians[0] is not None:
    for j in range(4):
        quat_plus = quat.copy()
        quat_plus[j] += eps
        quat_plus = quat_plus / np.linalg.norm(quat_plus)
        R_plus = Rotation.from_quat(quat_plus[[1, 2, 3, 0]]).as_matrix()
        predicted_plus = R_plus @ self.reference_point + translation
        residual_plus = self.weight * (self.measured_point - predicted_plus)
        for i in range(3):
            jacobians[0][i * 4 + j] = (residual_plus[i] - residuals[i]) / eps
```

**Implementation:**
1. Derive the analytic Jacobian of `R(q) @ p` with respect to `q = [w, x, y, z]` (unnormalized, since Ceres's `QuaternionManifold` handles the unit-norm constraint on the manifold tangent space separately — differentiate the standard quaternion-to-rotation-matrix formula directly, not a normalized version).
   For `R(q) @ p` where `p` is the fixed reference point, the closed-form per-component derivative is well-documented (e.g. in Ceres's own `rotation.h` — `QuaternionRotatePoint` and its Jacobian, or any robotics quaternion-kinematics reference). Concretely, for `q = (w, x, y, z)` and `p = (px, py, pz)`:
   ```
   d(R(q)p)/dw = 2 * [ w*px + y*pz - z*py,  w*py + z*px - x*pz,  w*pz + x*py - y*px ] ... (see derivation note below)
   ```
   Rather than re-deriving by hand in this plan, the implementer should either:
   - Port the closed-form Jacobian from Ceres's C++ `rotation.h` (`QuaternionRotatePoint` + `UnitQuaternionRotatePoint` Jacobian helpers) into Python/NumPy, or
   - Symbolically derive it once with `sympy` (`sympy.diff` on the quaternion-to-rotation-matrix formula times `p`, matrix form), simplify, and hand-transcribe the resulting closed-form expressions into `Evaluate` — this is the safer option since it produces a formula you can unit-test against the current finite-difference output before replacing it.
2. Residual is `weight * (measured - R(q)@ref - trans)`, so `d(residual)/dq = -weight * d(R(q)@ref)/dq` — apply the sign and weight after computing the rotation-only derivative.
3. The translation Jacobian is already correct and analytic (constant `∓weight·I`); while touching this function, hoist it to be computed once (e.g. cache `-weight * np.eye(3)` / `weight * np.eye(3)` as instance attributes in `__init__`) rather than rebuilt with nested Python loops on every `Evaluate` call.
4. **Validation step (do this before trusting the new Jacobian):** temporarily keep both implementations side by side in a test, and assert the analytic Jacobian matches the existing finite-difference Jacobian to within a small tolerance (e.g. `1e-4`) across a batch of random quaternions/points. Only delete the finite-difference code after this passes — this is exactly the kind of math that's easy to get subtly wrong (sign/order conventions), and the existing finite-difference version is a correct (if slow) reference implementation to check against.
5. After replacing, re-run the full solve on a reference recording and confirm `final_cost`, solved trajectories, and iteration count are the same or better (fewer iterations to converge is expected, since the gradient is now exact rather than a biased first-order approximation) — don't expect wildly different *answers*, since Ceres was already converging with the approximate gradient, just expect faster/cheaper convergence.

**Test:** unit test comparing analytic vs. finite-difference Jacobian on randomized inputs (per item 4 above); end-to-end timing comparison on a representative recording (same `max_iter`, compare wall-clock and `summary.num_successful_steps`).

**Outcome (2026-11-15): implemented.** Closed form derived via `sympy.diff` on `R(q/|q|) @ p` w.r.t. raw `[w,x,y,z]` (differentiating through the renormalization, matching the finite-difference code's semantics), evaluated at `|q|=1` (true whenever `Evaluate` is called, since `QuaternionManifold` keeps the parameter block on the unit sphere), validated against central differences to ~1e-7 before transcription, then transcribed verbatim (CSE'd sympy-generated code, not hand-retyped) into `_quaternion_point_jacobian` in `optimization.py`. `test_measurement_jacobian.py` checks it against the old forward-difference implementation (kept there as a reference, tolerance loosened to `1e-3` to account for that reference's own ~4e-4 forward-difference truncation error, not analytic error) — passes. Translation Jacobian hoisted to `__init__` as `self._translation_jacobian` (constant per instance). End-to-end validated on a synthetic rotating-rigid-body solve (noise-free ground truth + 1mm marker noise, 300 frames): `initial_cost`/`final_cost`/`iterations`/solved trajectories identical to the pre-change implementation to numerical precision, wall-clock ~2x faster (0.80s → 0.40s) on that synthetic case.

**Re-measurement (2026-11-17): confirmed on real data, fully closed out.** Lab machine re-ran the solver with the new Jacobian on two real recordings (`ferret_757_P43_E15__1` and `ferret_757_P41_E13`). For `P41_E13` (51,108 frames, 0 jumps flagged both before and after), a new frame-by-frame comparison tool (`compare_kinematics.py`, added alongside `batch_jump_detection.py` — see note below) diffed the full pre/post solved trajectories, not just jump counts: body-frame position delta mean 0.0000mm / p99 0.0002mm / max 0.043mm, orientation delta mean 0.0001° / p99 0.0004° / max 0.14°, across the entire recording — deviation only in one tiny ~10-frame cluster where the exact gradient nudged Ceres to a very slightly different point than the old biased finite-difference gradient, settling back to exact agreement everywhere else. This is the expected no-op-with-rounding-noise signature for a change that only affects *how* the same objective is optimized, not *what* is being optimized. `P43_E15__1` showed a small, bounded flagged-frame change (159 → 145) consistent with the same mechanism (a few frames near the jump threshold tipping either way). Confirms item 4 is safe and accuracy-neutral; no further action needed on this item.

**Tooling added while validating this item** (useful for re-measuring later plan items too): `batch_jump_detection.py` (runs `detect_orientation_jumps` across every `skull_kinematics.csv` under a dataset root and saves a JSON report — the original baseline's batch script was never committed, so this replaces a one-off with something re-runnable) and `compare_kinematics.py` (frame-by-frame position/orientation/per-marker delta between two solved kinematics outputs, for when jump-count parity isn't enough evidence on its own). Snapshot reports live in `jump_detection_reports/`. Note for future re-measurements: the solver overwrites `skull_kinematics.csv`/`.parquet` in place, so back up the directory you care about *before* re-running if you want a real before/after diff — the `P43_E15__1` before-state was lost this way and had to be worked around with a different session.

**Pre-change snapshot for comparison (2026-11-17):** before any item-4 re-runs on real data, every session's `skull_kinematics/` folder (as it stood after items 1-3, before item 4's analytic Jacobian) was copied to `/Users/philipqueen/Documents/skull_kinematics_backup_pre_step4/<session_dir>/skull_kinematics/` (outside Dropbox and outside git, ~8GB, not committed). As of 2026-11-17 this backup is still a valid "before" for every session except `P43_E15__1` and `P41_E13` (both already re-run — see above). To check whether re-running any other session changed its output, point `compare_kinematics.py` at the backup copy as "before" and the live Dropbox copy as "after":
```
BACKUP=/Users/philipqueen/Documents/skull_kinematics_backup_pre_step4/<session_dir>/skull_kinematics
LIVE=/path/to/analyzable_outputs/<session_dir>/skull_kinematics
uv run python -m python_code.rigid_body_solver.core.compare_kinematics \
  "$BACKUP/skull_kinematics.csv" "$BACKUP/skull_reference_geometry.json" \
  "$LIVE/skull_kinematics.csv" "$LIVE/skull_reference_geometry.json"
```
Expect near-zero deltas (sub-0.1mm position, sub-0.2deg orientation) matching the `P41_E13` result above — if a session shows materially larger deltas, that's worth investigating before trusting it. This only works once per session (the backup isn't refreshed after a session is re-run), so check the backup's mtime is still older than the live file before relying on it as "before."

---

## 5. Batch per-frame residuals into a single vectorized `CostFunction`

**Why fifth:** purely a performance optimization once the math (items 1-4) is trusted; this is the most invasive structural change (changes the residual-block layout of the whole problem) so it should be done near-last, against a known-good baseline, so any regression is easy to attribute.

**Where:** `optimization.py::optimize_rigid_body`, the per-keypoint loop building `MeasurementFactor`s (~line 318-332), and the `MeasurementFactor` class itself (~line 83-134).

**Implementation:**
1. Create a new `FrameMeasurementFactor(pyceres.CostFunction)` that takes *all* valid (non-NaN) measured points for one frame plus their corresponding reference points (and, if item 3 landed, their confidence weights) in its constructor, and sets `set_num_residuals(3 * n_valid)` with `set_parameter_block_sizes([4, 3])` (still just one quaternion + one translation per frame).
2. In `Evaluate`, compute `R(q)` once, then vectorize: `predicted = (R @ reference_points.T).T + translation` for all `n_valid` points in one shot (NumPy broadcasting, no Python-level per-point loop), and `residuals[:] = (weights * (measured_points - predicted)).ravel()`.
3. Jacobian: using the analytic per-point Jacobian from item 4, stack it: the quaternion Jacobian block is now `(3*n_valid, 4)` — compute the single-point analytic Jacobian for each of the `n_valid` reference points via broadcasting (same closed-form formula, vectorized over points) and tile into the output buffer; the translation Jacobian block `(3*n_valid, 3)` is just the single-point `∓weight·I` tiled `n_valid` times (can precompute the pattern once per frame in `__init__` since `n_valid` is fixed at construction time).
4. Replace the per-keypoint loop in `optimize_rigid_body` with one `FrameMeasurementFactor` construction + one `add_residual_block` call per frame.
5. This reduces the number of Python-level `Evaluate` calls from `n_frames * n_keypoints` to `n_frames` — the dominant cost for pyceres (Python/C++ boundary crossings), so expect the biggest wall-clock win here, larger than item 4 alone.

**Test:** confirm `problem.num_residual_blocks()` drops as expected (from roughly `n_frames*n_keypoints + 2*(n_frames-1)` down to `2*n_frames - 1`), confirm total residual count (`problem.num_residuals()` or equivalent) is unchanged, and confirm final solved trajectories match the pre-batching baseline to numerical tolerance — this change should be bitwise-equivalent in the limit of exact arithmetic, since it's the same residuals just grouped differently; any mismatch indicates a vectorization bug, not a modeling difference.

**Outcome (2026-11-19): implemented and validated.** Added `FrameMeasurementFactor` to `optimization.py` (kept `MeasurementFactor` in place as a permanent reference implementation, not deleted) and replaced the per-keypoint loop in `optimize_rigid_body` with one `FrameMeasurementFactor` per frame over that frame's valid keypoints (reusing the already-computed `valid_mask` instead of re-checking NaN per keypoint). The analytic per-point Jacobian (`_quaternion_point_jacobian` from item 4) is called once per valid keypoint inside a single `Evaluate`, not re-derived/vectorized — this still collapses the per-frame call count from `n_keypoints` to `1`, which is the actual cost win (Python/C++ boundary crossings dominate, not the per-point math), while carrying zero risk of a new transcription bug in a second copy of that formula.

Found and worked around an unrelated pre-existing latent bug while building the synthetic test harness: `_estimate_initial_pose`'s fallback (line ~32) returns the *same* array object as the previous frame's pose when a frame has fewer than 3 valid keypoints, instead of a copy. If that happens, the smoothness factor between that frame and its neighbor gets the identical array object passed twice, which pyceres rejects (`Duplicate parameter blocks in a residual parameter are not allowed`). Not triggered in production today since ferret skull frames apparently always have >=3 valid keypoints, and out of scope for item 5 — flagging here for a future fix (copy `fallback_pose`'s arrays before returning).

Validation performed:
- **Unit test** (`test_frame_measurement_factor.py`): 503 trials (fixed edge cases `n_valid` in {1,2,8} plus 500 random trials, `n_valid` in [1,12)) comparing `FrameMeasurementFactor`'s residuals/Jacobians against the equivalent stack of individual `MeasurementFactor` calls — max abs error `3.64e-12` (machine precision, as expected since both paths share the same analytic code). PASS.
- **Synthetic end-to-end test** (`test_frame_measurement_batching_e2e.py`): 250-frame synthetic rotating rigid body (8 keypoints, 1mm noise, random single-keypoint NaN gaps) solved both ways (old per-keypoint loop vs. new batched loop, reimplemented locally since both classes coexist). Residual block count dropped from 2478 to 748 as expected; total residual count unchanged (7683); `initial_cost`/`final_cost`/`iterations` identical; solved quaternions/translations identical to `1e-9` (actual max deltas `2.22e-16`/`4.44e-15`). A separate isolated check (`test_zero_valid_frame_skip`) confirms a frame with zero valid keypoints gets no measurement factor, matching prior behavior.
- **Real-data validation:** no raw per-session triangulated input (`head_freemocap_data_by_frame.csv`) was available on this machine for the live Dropbox `analyzable_outputs` sessions — only one example recording ships with raw data, in `data/example_recordings/session_2025-07-01_ferret_757_EyeCameras_P33_EO5.zip` (`full_recording`, ferret 757, 20,243 frames). Ran `run_ferret_skull_solver` on that recording's raw input both before (stashed the `optimization.py` change, old per-keypoint loop) and after (restored, new batched loop), then diffed with `compare_kinematics.py`: body-frame position delta mean 0.0006mm / p99 0.0055mm / max 0.108mm, orientation delta mean 0.0011° / p99 0.0079° / max 1.127° (one outlier frame; all others sub-0.2°) — consistent with the same no-op-with-rounding-noise signature documented for item 4, confirming this is a pure regrouping of the same objective. Wall-clock: 429.9s (old, per-keypoint) → 165.7s (new, batched) on this recording — **~2.6x speedup**, the actual point of this item.
- **Not done:** the dataset-wide `batch_jump_detection.py` sweep across all ~65 sessions requires re-running the solver on sessions whose raw triangulated input only exists on the lab machine (same constraint noted in item 4's outcome). Recommend running it from the lab machine before considering item 5 fully closed out, mirroring item 4's precedent.

**Re-measurement (2026-11-19): clean-session and worst-session spot check, lab machine.** `skull_kinematics/` backups taken pre-emptively at `/Users/philipqueen/Documents/skull_kinematics_backup_pre_step5/<session_dir>/skull_kinematics/` for `ferret_757_P41_E13` (clean, 0 flagged jumps in the original baseline) and `ferret_403_P46_E4` (worst session by flagged-rate in the original baseline) were used as "before" after the lab machine re-ran both sessions with the batched code.

- **`P41_E13`** (51,108 frames): position delta mean `1.7e-8mm` / max `6.3e-6mm`, orientation delta mean `4.4e-8°` / max `4.5e-5°` — floating-point-noise level, tighter even than item 4's own validation on this same session. Confirms the batching change is a true no-op in a well-conditioned (clean, no soft-tissue-flex-driven ambiguity) session.
- **`P46_E4`** (32,606 frames): **not** a no-op — position delta mean 1.29mm / p99 14.9mm / max 40.8mm, orientation delta mean 4.65° / p99 88.8° / max 179.7°, with ~8% of frames exceeding 5mm/20°. However, jump-count *improved*: 445 flagged frames before -> **50 after** (`jump_detection_reports/compare_403_P46_E4_item5.json`). The large-delta frames are strongly concentrated where the old solver was already uncertain (58% land within 15 frames of an old jump-flagged frame, vs. 11.5% expected by chance — 5x enrichment), not scattered across well-constrained frames. Mechanism: this session's residuals are dominated by soft-tissue flex (70-90mm on every marker, every frame, per item 2's notes), which makes parts of its optimization landscape genuinely non-unique. Batched vs. per-keypoint residual summation is mathematically identical but evaluated in a different order; in well-conditioned regions that's inconsequential (as `P41_E13` shows), but in this session's already-ambiguous frames the same tiny perturbation is enough to tip Ceres into a *different* nearby local minimum — one that happened to be less jumpy more often than not, but is not guaranteed to be.

**Conclusion: item 5 is a no-op in well-conditioned sessions, but can re-select which local minimum an already-ambiguous frame lands in in highly ill-posed sessions (dominated by soft-tissue flex / heavy occlusion).** This is a pre-existing property of the underlying non-convex problem exposed by this change, not a bug introduced by it — but it means "accuracy-neutral" should be understood per-session, not as a blanket guarantee, and the dataset-wide sweep (still not done — see above) matters more for this item than it would for a change with no interaction with solver numerics.

---

## 6. Wire up (or remove) the dead `rigid_edges` parameter in `estimate_reference_geometry`

**Why last, and why this is cleanup rather than a bug fix:** `optimize_rigid_body` accepts `rigid_edges` (`optimization.py:220`) but never forwards it to `estimate_reference_geometry` (~line 262) — only `display_edges` goes through. `estimate_distance_matrix` (`calculate_reference_geometry.py`) always computes distances for *every* keypoint pair regardless.

Checked against the actual live caller: `ferret_skull_solver.py:46` already sets `rigid_edges = list(combinations(keypoint_names, 2))` — i.e. the one real topology in production already treats every marker as mutually rigid, identical to what `estimate_distance_matrix` computes with or without the forwarding fix. **So today this causes no actual bias** — it's dead plumbing, not an active accuracy bug. (The one place a genuine non-complete `rigid_edges` subset appears is `cube_solver_demo.py`, which is currently broken against the API — `from python_code.rigid_body_solver.core import TrackingConfig` fails, `TrackingConfig` no longer exists — so no live path is exercising partial rigidity today either.)

The reason to still fix this: it's misleading API surface. A future topology that mixes rigid and non-rigid markers (e.g. adding a skin/eye marker to the skull set) would silently have `rigid_edges` ignored and get all-pairs distances instead, corrupting that topology's reference geometry with no warning. Do this last since it has no effect on current behavior and isn't blocking anything above.

**Implementation (pick one):**
- **Option A — wire it up:** pass `rigid_edges=rigid_edges` through to `estimate_reference_geometry`; in `estimate_distance_matrix`, add a `rigid_edges` parameter so that when given, only `(i, j)` pairs present in it get a real distance (match by name, both orderings), with all other entries left as `np.nan`. `reconstruct_from_distances`'s classical MDS needs a complete matrix, so either (a) require `rigid_edges` to span a connected graph over all keypoints and fill missing pairs via shortest-path through the known rigid distances (`scipy.sparse.csgraph.shortest_path`) before MDS, or (b) raise a clear `ValueError` if `rigid_edges` doesn't connect every keypoint. Also fix `cube_solver_demo.py`'s broken imports (`TrackingConfig` → `RigidBodySolverConfig`/`process_tracking_data`'s current signature) so the one test case that actually exercises partial rigidity runs again, and use it to validate this path.
- **Option B — remove it:** if there is no near-term plan to support mixed rigid/non-rigid topologies, delete the unused `rigid_edges` parameter from `optimize_rigid_body`'s signature (and stop passing it from `main_solver_interface.py:282`) so the API doesn't imply a capability that doesn't exist. Keep `rigid_edges` on `StickFigureTopology`/`ReferenceGeometry` only for its current real uses (display/bookkeeping) if any remain after removal.

Prefer **Option A** if mixed-rigidity topologies (e.g. skull + a non-rigidly-attached marker) are on the roadmap; otherwise **Option B** keeps the code honest about what it does.

**Test:** if Option A — same synthetic test as before (one marker given a slowly time-varying offset from an otherwise-rigid cluster; confirm the rigid markers' reconstructed geometry is unaffected once excluded via `rigid_edges`) — plus fixing and re-running `cube_solver_demo.py`. If Option B — confirm nothing downstream actually reads `rigid_edges` off `optimize_rigid_body`'s signature before deleting it.

**Outcome (2026-11-19): implemented, Option A.**
- `optimization.py::optimize_rigid_body` now forwards `rigid_edges=rigid_edges` to `estimate_reference_geometry` (was silently dropped before).
- `calculate_reference_geometry.py::estimate_distance_matrix` gained `keypoint_names`/`rigid_edges` params: when `rigid_edges` is given, only those (name-matched, either ordering) pairs' distances are computed directly from the data; every other pair is filled via `scipy.sparse.csgraph.shortest_path` over the known rigid distances (new helper `_fill_missing_distances_via_shortest_path`), which raises `ValueError` if `rigid_edges` doesn't connect every keypoint — implementing the plan's Option (a) branch directly rather than requiring the caller to pre-check connectivity. `estimate_reference_geometry` threads `keypoint_names`/`rigid_edges` through to it; when `rigid_edges` is `None` it still defaults to all-pairs first (unchanged), so production behavior (`ferret_skull_solver.py`'s all-pairs topology) is bit-for-bit identical to before — this is purely additive for topologies that actually supply a partial `rigid_edges`.
- Fixed `cube_solver_demo.py`'s broken imports/API usage (`TrackingConfig` → `RigidBodySolverConfig`, `process_tracking_data(config=...)` with no `ground_truth_data` kwarg, `OptimizationConfig` without the nonexistent `lambda_rigid` field, the nonexistent `data_savers.save_simple_csv` replaced with a local tidy-CSV writer matching `load_measured_trajectories_csv`'s expected format) and switched its `rigid_edges` from index pairs to keypoint-name pairs (the model requires names). Ran it end-to-end: solves successfully, mean reconstruction error vs. ground truth 2.09mm / max 4.57mm over 200 frames (noise_std=0.1) — this is now the one live exercised case of a genuine partial-rigidity topology (cube's 19 edges, not all 55 pairs among its 11 keypoints).
- New test `test_rigid_edges_partial_rigidity.py`: a 4-point rigid cluster (a,b,c,d) plus a hinge-attached marker e (fixed distance to `a` only, free-wobbling relative to b/c/d). With `rigid_edges` restricted to the honest pairs (all of a/b/c/d, plus a-e), the reconstructed rigid-cluster geometry matches a clean a/b/c/d-only baseline at least as closely as the legacy all-pairs path (which feeds e's inconsistent raw median distances into the same joint MDS solve) — `0.246mm` vs `0.330mm` max error in one representative run. Also directly verifies `estimate_distance_matrix`'s excluded-pair values exactly match an independent shortest-path computation (not the raw median), that directly-specified pairs are untouched, and that a `rigid_edges` set leaving a keypoint disconnected raises `ValueError` rather than silently producing a nonsensical matrix.
- All pre-existing tests (`test_frame_measurement_factor.py`, `test_measurement_jacobian.py`, `test_frame_measurement_batching_e2e.py`) still pass — confirms this change has zero effect on the all-pairs (current production) path.

---

## Summary table

| # | Fix | Primary benefit | Risk if skipped |
|---|-----|------------------|------------------|
| 1 | Kabsch/Procrustes per-frame pose initialization | Removes local-minima jumps | Frames can converge to wrong orientation, especially after large motion |
| 2 | Robust loss (Huber) on measurement residuals | Limits damage from single bad marker detections | One outlier marker drags the whole frame's pose |
| 3 | Confidence-weighted measurement residuals | Smooths through low-confidence stretches without over-smoothing elsewhere | Low-confidence detections trusted as much as high-confidence ones |
| 4 | Analytic quaternion Jacobian | Faster, more precise convergence | 5x redundant rotation computation per residual; slight gradient bias |
| 5 | Per-frame batched residual blocks | Major wall-clock speedup at scale | Python call overhead dominates solve time for long recordings |
| 6 | Wire up or remove dead `rigid_edges` parameter | Honest API; ready for mixed-rigidity topologies if ever needed | No current risk — purely a future-proofing/clarity cleanup |
