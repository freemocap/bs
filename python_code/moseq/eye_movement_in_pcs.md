# Is eye movement captured in the HEAD_WITH_PUPIL_POINTS latent space?

Notes from investigating why eye movement is hard to see in the `HEAD_WITH_PUPIL_POINTS`
model's PCs, what's actually going on, and options for fixing it. Diagnostic tooling
referenced here lives in `visualization/eye_pc_viz.py`.

## Background: latent_dim and kappa

Keypoint-MoSeq doesn't model keypoints directly. Two separate hyperparameters govern
different parts of the model, and neither one is "the number of syllables":

- **`latent_dim`** (`config.yml`, currently `10`) is the dimensionality of the
  *continuous* pose trajectory `x(t)`: the top `latent_dim` principal components of the
  egocentrically-aligned keypoints. It's a property of the pose representation, not of
  syllables.
- **`kappa`** (`run_moseq_pipeline.fit_full_model`, currently `1e4`) controls how
  "sticky" the syllable transitions are -- higher kappa means longer, fewer syllable
  bouts; lower kappa means shorter, more numerous ones. The actual number of syllables
  used is capped by a separate truncation (`num_states`, default 100 in
  `keypoint_moseq`), and how many of those 100 slots end up meaningfully occupied is an
  emergent property of the fit, not something either hyperparameter sets directly.

A syllable is a discrete label `z(t)` that owns its own AR (autoregressive) dynamics on
`x(t)`. Fitting the model means jointly inferring which syllable is active at each
frame and what each syllable's dynamics look like.

## PCs are not per-keypoint

Each of the 10 latent dimensions is a principal component of *all* keypoints
combined -- a linear combination of every keypoint's coordinates, not one PC per
keypoint. PCs are ordered by variance explained, not by anatomical meaning. `kpms.plot_pcs`
(called at the end of `run_moseq_pipeline.fit_pca`) draws each PC as a keypoint
perturbation so you can see what it moves, but with a 26-keypoint set spanning head and
both eyes, head displacement (tens of mm) makes eye displacement (a couple mm) hard to
see by eye in that plot.

## Two diagnostics: is eye movement actually in there?

`visualization/eye_pc_viz.py` answers this two ways instead of reading loading plots by eye:

1. **`compute_pc_keypoint_group_loadings`** -- decomposes each PC's loading vector (in
   real keypoint space, undoing kpms's `center_embedding` change of basis) into the
   fraction of loading energy on `head` vs. `left_eye` vs. `right_eye` keypoints. Needs
   only the fitted PCA object, no AR-HMM required.
2. **`compute_pc_eye_angle_correlation`** -- correlates each PC's fitted latent
   trajectory (`results[...]["latent_state"]`) against the *independently* computed
   `eye_in_head` adduction/elevation angles from the eye kinematics pipeline (not
   derived from these pupil-point keypoints at all). This is the check that actually
   matters: it's possible for a PC to have eye keypoints in its loading without
   tracking real eye movement well, or vice versa.

Both are run automatically after fitting any `KPMS_Loader.HEAD_WITH_PUPIL_POINTS` model
(wired into `run_moseq_pipeline.run_pipeline`/`main`), saved to
`{project_dir}/{model_name}/eye_pc_plots/`.

### Result on the `head_with_pupil_points_405_407_test` model

Loading energy is head-majority (55-90%) on every one of the 10 PCs, no PC is
eye-dominant, and best-case correlation against ground-truth eye angle is R² = 0.19
(PC 8 vs. right adduction) -- everything else is ≤ 0.08. Eye movement is essentially
not being resolved in this latent space.

## Why: the alignment only removes yaw

Traced through kpms's actual PCA fitting
(`jax_moseq.models.keypoint_slds.alignment.fit_pca`/`align_egocentric`):

1. Subtract the centroid (mean position across all 26 keypoints) -- removes translation.
2. Rotate so the nose→base vector points along a fixed axis -- removes **yaw only**.
3. Change basis via `center_embedding(k)` (removes the now-redundant translational DOF)
   and flatten, then fit an ordinary `sklearn.decomposition.PCA`.

Head pitch and roll are **not** removed by step 2 (only yaw is). Since the head
keypoints are a rigid body (from `skull_kinematics`), all of their yaw variance is
normalized away, but head nodding/tilting is real, large-amplitude variance (mm of
keypoint displacement over the skull's lever arm) that survives and competes directly
against pupil deflection (a couple mm) in one shared, variance-maximizing PCA. Head
wins essentially every time. This isn't a fixable scale quirk in the current design --
it's the direct, expected consequence of a single joint PCA over both keypoint groups.

## Options going forward

The goal is to capture head and eye movement *together* (saccades, head direction,
compensatory/VOR-like eye-head coordination) -- not just eye movement in isolation. That
requirement rules out fitting two completely independent models (one per modality):
separate models can't discover coupled syllables at all. Any fix needs one shared
syllable/AR-HMM operating on a latent space where neither modality can drown out the
other.

### Option A: increase `latent_dim`

Cheapest thing to try. If eye signal is only slightly below the current cutoff, giving
PCA more components to keep might surface it further down the ranking. Doesn't fix the
underlying imbalance (head still wins on variance) -- it just gives eye signal a chance
to show up in a PC further down the list rather than being discarded outright.
Low effort, worth trying first, but likely a partial fix at best given the fraction of
loading energy is already low across *all* current PCs, not just concentrated near the
current cutoff.

### Option B: separate PCA per keypoint group, concatenated into one latent (recommended)

Fit two (or three) independent PCAs -- one on the 8 head keypoints, one on the eye
keypoints -- each normalized to unit variance *before* combining, so neither's raw
physical amplitude decides how much of the shared latent space it gets. Assemble a
single block-diagonal loading matrix:

```
combined_components = [[head_components,          0                    ],
                       [0,                 left_eye_components,        0],
                       [0,                        0,   right_eye_components]]
```

This is still a valid linear PCA-shaped object (same `components_`/`mean_` shape kpms
expects), so it's a drop-in replacement for `kpms.fit_pca`'s output -- `fit_ar_model`,
`fit_full_model`, `extract_results`, and all existing visualization work unchanged.
`latent_dim` becomes `head_dim + left_eye_dim + right_eye_dim` (e.g. 6 + 2 + 2 = 10),
each block guaranteed a fair share instead of head eating the whole budget.

Critically, the concatenated vector still feeds into **one** joint AR-HMM -- one
syllable label per frame, dynamics free to couple head and eye dimensions -- so
compensatory eye-head syllables remain discoverable. What must NOT happen is two
separate AR-HMMs; that would throw away the coupling entirely.

A further refinement: fit the eye block's PCA on `eye_in_head` adduction/elevation (2
numbers/eye, already head-relative, already used in `eye_syllable_viz.py`) instead of
the 18 raw pupil-boundary keypoints, which mostly encode the same 2 DOF per eye plus
tracking noise. A small PCA over the 4 `eye_in_head` numbers (left/right ×
adduction/elevation) would more directly surface conjugate vs. vergence eye-movement
modes.

### Option C: hand-engineered feature space with a bare AR-HMM (bigger redesign)

`jax_moseq.models.arhmm` is a standalone sticky HDP-AR-HMM with no keypoint/PCA emission
layer at all. The most faithful-to-the-actual-question version of this project would fit
that directly on a hand-built feature vector: head orientation (Euler angles or
quaternion log-map from `skull_kinematics`) concatenated with `eye_in_head` angles --
skipping keypoint PCA (and the pupil-point keypoints) entirely.

This is the option most directly aimed at "saccades + head direction + VOR-style
compensation" as the object of study, rather than an incidental byproduct of pose PCA.
Tradeoffs: bigger lift (no more `kpms.fit_pca`/`fit_ar_model`/`fit_full_model`
convenience layer -- would need a custom Gibbs-sampling loop, syllable extraction, and
plotting), and loses kpms's built-in keypoint-based video/grid-movie visualizations
(`generate_grid_movies`, `generate_trajectory_plots`), since those are built around
keypoint reconstruction from the latent state, which no longer applies to a
non-keypoint feature space.

## Recommendation

Start with **Option B**, restricted to eye keypoints as `eye_in_head` angles rather than
raw pupil points. It's a drop-in for the existing pipeline (no loss of any current
tooling), directly fixes the variance-domination mechanism identified above, and keeps
the joint-syllable structure needed for compensatory eye-head behavior. Revisit **Option
C** if Option B's eye block still isn't resolving enough for the specific behaviors of
interest (e.g. individual saccades within a syllable, rather than syllable-level eye
tendencies).
