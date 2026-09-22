# Analyzable Output CSVs


## Common Format

Most files use a **tidy format**: one row per observation, with these columns:

| column | type | description                                                           |
|---|---|-----------------------------------------------------------------------|
| `frame` | int | 0-based frame index                                                   |
| `timestamp_s` | float | Time in seconds, timebase = zeroed to the start of the recording      |
| `trajectory` | str | What is being measured (see tables below)                             |
| `component` | str | Spatial dimension,quaternion component, etc (e.g. `x`, `y`, `z`, `w`) |
| `value` | float | The measurement value                                                 |
| `units` | str | Unit string (e.g. `mm`, `rad_s`, `quaternion`)                        |

> The resampled trajectory CSVs (listed at the end) use `timestamp` instead of `timestamp_s`.

Each `(frame, trajectory, component)` combination produces exactly one row.

**`roll`/`pitch`/`yaw` convention:** these are Euler angles extracted from the rotation quaternion, in body-frame axis order `roll = rotation about local X, pitch = about local Y, yaw = about local Z`. For the skull and gaze body frames used in this pipeline (see below), that means roll is rotation about the forward/nose axis, pitch is about the left-right axis, and yaw is about the vertical axis, following the right-hand rule.

**World-frame axes are not anatomically fixed.** `world coordinates` below means the arena coordinate system set by camera calibration — it's consistent within one recording, but which direction is "up" or "forward" in the room is arbitrary and can differ between recordings/calibrations. Anatomical directions (nose/left/up) are only meaningful for **local/body-frame** quantities (`angular_velocity_local`, `angular_acceleration_local`, and the eye/gaze coordinate frames described below), not for raw world-frame `position`/`orientation` x,y,z values.

---

## `skull_kinematics/skull_kinematics.csv`

Skull rigid-body pose over time. All trajectories are in **world coordinates**.

The difference between global and local frames matters when rotational axes are combined.
Local frames are what would be measured by a device on the skull, like an IMU.
Global frames are measured in relation to the arena axes.
Ex. When the head is level, looking left is a yaw in both frames. But when the head is already rotated, a LOCAL yaw will contain pitch and yaw components in the GLOBAL frame.

**Skull body frame** (used for `_local` trajectories, and for the anatomical meaning of roll/pitch/yaw): origin at the midpoint of the eyes; **+X points toward the nose** (forward), **+Y points toward the left eye** (subject's left), **+Z points up** (X × Y).

| trajectory | components | units | meaning |
|---|---|---|---|
| `position` | x, y, z | mm | Position of the skull center in world space |
| `orientation` | w, x, y, z | quaternion | Skull rotation as a quaternion (world frame) |
| `linear_velocity` | x, y, z | mm_s | Rate of change of skull center position |
| `linear_acceleration` | x, y, z | mm_s2 | Second derivative of skull center position |
| `angular_velocity_global` | roll, pitch, yaw | rad_s | Angular velocity in the world frame |
| `angular_velocity_local` | roll, pitch, yaw | rad_s | Angular velocity in the skull's own body frame |
| `angular_acceleration_global` | roll, pitch, yaw | rad_s2 | Angular acceleration in the world frame |
| `angular_acceleration_local` | roll, pitch, yaw | rad_s2 | Angular acceleration in the skull body frame |
| `keypoint__nose` | x, y, z | mm | Nose marker position in world space |
| `keypoint__left_ear` | x, y, z | mm | Left ear marker in world space |
| `keypoint__right_ear` | x, y, z | mm | Right ear marker in world space |
| `keypoint__left_eye` | x, y, z | mm | Left eye socket center in world space |
| `keypoint__right_eye` | x, y, z | mm | Right eye socket center in world space |
| `keypoint__base` | x, y, z | mm | Skull base marker in world space |
| `keypoint__left_cam_tip` | x, y, z | mm | Left eye camera mount tip in world space |
| `keypoint__right_cam_tip` | x, y, z | mm | Right eye camera mount tip in world space |

---

## `*_eye_kinematics/*_eye_kinematics.csv`

Eye orientation and pupil tracking data. **Coordinates are in the eye camera frame, not world space.** The eye is modeled as a sphere rotating in place, so position is always [0, 0, 0] and is not saved.

**Eye camera frame convention** (at rest, right-handed): **+Z = rest gaze direction** ("north pole" of the eyeball, i.e. straight ahead), **+Y = superior (up)**, **+X = subject's left** (computed as Y × Z). This applies to the `orientation` quaternion axes and to every keypoint's x/y/z components.

> **This +X = subject's left convention is the same for both eyes — it is not nose-relative and does not flip between `left_eye_kinematics.csv` and `right_eye_kinematics.csv`.** It only tells you left/right in an absolute anatomical sense, not toward/away from the nose. This is different from `eye_in_head.adduction` below, which *is* nose-relative and *does* flip sign per eye (see that row).

| trajectory | components | units | meaning |
|---|---|---|---|
| `orientation` | w, x, y, z | quaternion | Eyeball rotation quaternion (camera frame) |
| `angular_velocity_local` | x, y, z | rad_s | Angular velocity in the eyeball's own frame (camera frame) |
| `angular_acceleration_local` | x, y, z | rad_s2 | Angular acceleration in the eyeball's own frame (camera frame) |
| `keypoint__tear_duct` | x, y, z | mm | Medial (inner) corner of the eye socket in camera space |
| `keypoint__outer_eye` | x, y, z | mm | Lateral (outer) corner of the eye socket in camera space |
| `keypoint__pupil_center` | x, y, z | mm | Tracked center of the pupil in camera space |
| `keypoint__p1`–`keypoint__p8` | x, y, z | mm | Eight points around the pupil boundary in camera space |
| `keypoint__gaze_target` | x, y, z | mm | Direction the eye points at rest (unit vector, camera frame) |
| `pupil_axis` | major, minor | mm | Major and minor axes of the fitted pupil ellipse |
| `eye_in_head` | adduction, elevation | rad | Anatomical gaze angles: adduction (positive = toward nose/medial, negative = away from nose/lateral), elevation (positive = up) |

`adduction` is nose-relative and its sign is **flipped between the two eyes** to stay anatomically consistent: it's derived from azimuth (where +azimuth = looking toward +X = subject's left), and +X is medial (toward the nose) for the **right** eye but lateral (away from the nose) for the **left** eye. So `adduction = +azimuth` for the right eye and `adduction = -azimuth` for the left eye. `elevation` has no such flip — +Y = up is the same for both eyes.

---

## `gaze_kinematics/left_gaze_kinematics.csv` and `gaze_kinematics/right_gaze_kinematics.csv`

World-space gaze — the combination of skull motion and eye rotation. Use these files (not the eye kinematics files) when you want to know where the ferret is actually looking in the room.

All trajectories are in **world coordinates** (see the world-frame caveat above — these axes are calibration-defined, not anatomically fixed). The `_local` trajectories use the same eye-frame convention as the eye kinematics files above (+Z = gaze direction, +Y = up, +X = subject's left), now expressed as a local frame riding on the world-space gaze orientation.

| trajectory | components | units | meaning |
|---|---|---|---|
| `position` | x, y, z | mm | Eye center position in world space (moves with the skull) |
| `orientation` | w, x, y, z | quaternion | World-space rotation representing the eye pointing direction |
| `linear_velocity` | x, y, z | mm_s | Velocity of the eye center in world space |
| `linear_acceleration` | x, y, z | mm_s2 | Acceleration of the eye center in world space |
| `angular_velocity_global` | roll, pitch, yaw | rad_s | Gaze angular velocity in world frame |
| `angular_velocity_local` | roll, pitch, yaw | rad_s | Gaze angular velocity in the eye's own frame |
| `angular_acceleration_global` | roll, pitch, yaw | rad_s2 | Gaze angular acceleration in world frame |
| `angular_acceleration_local` | roll, pitch, yaw | rad_s2 | Gaze angular acceleration in eye frame |
| `keypoint__eyeball_center` | x, y, z | mm | Eyeball center position in world space |
| `keypoint__gaze_target` | x, y, z | mm | World-space point the eye is directed toward |
| `keypoint__pupil_center` | x, y, z | mm | Canonical/idealized pupil center (fixed rest-frame point rotated by orientation), projected into world space |
| `gaze_angle` | horizontal, vertical | degrees | Gaze direction as spherical angles: horizontal (positive = right), vertical (positive = up) |
| `tracked_pupil__pupil_center` | x, y, z | mm | Actual tracked pupil center (real per-frame detection, not idealized geometry), projected into world space. **Parquet only, not in the CSV.** |
| `tracked_pupil__p1` ... `tracked_pupil__p8` | x, y, z | mm | Actual tracked pupil boundary points (8 points), projected into world space. **Parquet only, not in the CSV.** |

`tracked_pupil__*` rows come from `TrackedPupil.pupil_center_mm`/`pupil_points_mm` (real detections, distinct from the idealized `keypoint__pupil_center`). These values already embed the eyeball's own per-frame rotation (they're the data `orientation` was itself derived from), so projecting them to world only applies the eye-to-skull mounting rotation and the skull's world rotation — not the eye's own quaternion again, which would double-rotate them. See `project_tracked_pupil_to_world()` in `calculate_gaze/calculate_ferret_gaze.py`. These rows are written only to `{side}_gaze_kinematics.parquet`, not to the `.csv`.

---

## `gaze_kinematics/eye_basis_vectors/{side}_eye_basis_vectors_world.csv`

**Different format from the other files.** One row per (frame, basis axis).

| column | type | description |
|---|---|---|
| `frame` | int | Frame index |
| `timestamp_s` | float | Time in seconds |
| `basis_axis` | str | Which axis of the eye frame: `x`, `y`, or `z` |
| `world_x` | float | X component of that axis direction in world space |
| `world_y` | float | Y component |
| `world_z` | float | Z component |

Each row gives the world-space direction of one coordinate axis of the eye frame, for one frame. `basis_axis` follows the eye-frame convention above: `z` = the gaze/optical axis, `y` = up, `x` = subject's left. This is useful for verifying that the eye-to-skull-to-world coordinate transform is correct.

---

## `skull_and_spine_trajectories_resampled.csv`

Raw 3D positions of the skull and spine markers, in **world coordinates**, resampled to the common pipeline timestamps. Unlike `skull_kinematics.csv`, this contains **only positions** — no orientation, velocity, or acceleration.

Columns: `frame`, `timestamp`, `trajectory`, `component` (x/y/z), `value`, `units` (mm)

One `trajectory` per tracked marker (nose, ears, eyes, spine vertebrae, etc.).

---

## `left_eye_kinematics/left_eye_trajectories_resampled.csv` and `right_eye_kinematics/right_eye_trajectories_resampled.csv`

Raw 3D positions of the eye landmarks, resampled to common timestamps. **Coordinates are in the eye camera frame.** Only positions — no orientation or angles.

Columns: `frame`, `timestamp`, `trajectory`, `component` (x/y/z), `value`, `units` (mm)

| trajectory | meaning |
|---|---|
| `tear_duct` | Medial corner of the eye socket |
| `outer_eye` | Lateral corner of the eye socket |
| `pupil_center` | Tracked pupil center |
| `p1`–`p8` | Eight points around the pupil boundary |

---

## `toy_trajectories_resampled.csv`

World-space positions of the three tracked toy markers, resampled to common timestamps.

Columns: `frame`, `timestamp`, `trajectory`, `component` (x/y/z), `value`, `units` (mm)

| trajectory | meaning |
|---|---|
| `toy_face` | Front/face side of the toy |
| `toy_top` | Top of the toy |
| `toy_tail` | Tail/rear of the toy |

---

## `eye_data_quality.csv`

Per-eye tracking quality flags, resampled to common timestamps. Uses the common tidy format (`frame`, `timestamp_s`, `trajectory`, `component`, `value`, `units`).

| trajectory | components | units | meaning |
|---|---|---|---|
| `left_eye_data_quality` | low_threshold, medium_threshold, high_threshold | boolean | Whether left eye DLC confidence passed each threshold for that frame |
| `right_eye_data_quality` | low_threshold, medium_threshold, high_threshold | boolean | Whether right eye DLC confidence passed each threshold for that frame |

Only produced when a corresponding eye-confidence CSV exists for the recording.

---

## `reprojection_errors/reprojection_errors.csv`

Per-camera and aggregate skull-keypoint reprojection error, resampled to common timestamps. Uses the common tidy format, with `units` in `px`.

| column | meaning |
|---|---|
| `trajectory` | Camera ID, or `mean` for an aggregate row |
| `component` | Skull keypoint name, or `mean` for the error averaged across all keypoints |
| `value` | Reprojection error in pixels |

Row types per frame: one row per (camera, keypoint), one `(mean, keypoint)` row averaging across cameras, and one `(mean, mean)` row averaging across cameras and keypoints.

Only produced when a `post_solver_reprojection_errors.csv` exists for the recording — not guaranteed to be present.
