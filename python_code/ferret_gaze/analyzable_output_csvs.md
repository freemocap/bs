# Analyzable Output CSVs


## Common Format

Most files use a **tidy format**: one row per observation, with these columns:

| column | type | description |
|---|---|---|
| `frame` | int | 0-based frame index |
| `timestamp_s` | float | Time in seconds, zeroed to the start of the recording |
| `trajectory` | str | What is being measured (see tables below) |
| `component` | str | Vector or quaternion component (e.g. `x`, `y`, `z`, `w`) |
| `value` | float | The measurement value |
| `units` | str | Unit string (e.g. `mm`, `rad_s`, `quaternion`) |

> The resampled trajectory CSVs (listed at the end) use `timestamp` instead of `timestamp_s`.

Each `(frame, trajectory, component)` combination produces exactly one row.

---

## `skull_kinematics/skull_kinematics.csv`

Skull rigid-body pose over time. All trajectories are in **world coordinates**.
Angular velocity and acceleration are in the skull's local (body) frame — i.e., what would be measured by an IMU on the skull. A local yaw is a pure yaw when the head is level, but contains pitch and yaw components in the world frame when the head is already rotated.

| trajectory | components | units | meaning |
|---|---|---|---|
| `position` | x, y, z | mm | Position of the skull center in world space |
| `orientation` | w, x, y, z | quaternion | Skull rotation as a quaternion (world frame) |
| `linear_velocity` | x, y, z | mm_s | Rate of change of skull center position |
| `linear_acceleration` | x, y, z | mm_s2 | Second derivative of skull center position |
| `angular_velocity_local` | roll, pitch, yaw | rad_s | Angular velocity in the skull's own body frame |
| `angular_acceleration_local` | roll, pitch, yaw | rad_s2 | Angular acceleration in the skull body frame |

> Keypoint positions are not saved here — use `skull_and_spine_trajectories_resampled.csv` instead.

---

## `*_eye_kinematics/*_eye_kinematics.csv`

Eye orientation and pupil tracking data. **Coordinates are in the eye camera frame, not world space.** The eye is modeled as a sphere rotating in place, so position is always [0, 0, 0] and is not saved.

| trajectory | components | units | meaning |
|---|---|---|---|
| `orientation` | w, x, y, z | quaternion | Eyeball rotation quaternion (camera frame) |
| `angular_velocity_local` | x, y, z | rad_s | Angular velocity in the eyeball's own frame (camera frame) |
| `angular_acceleration_local` | x, y, z | rad_s2 | Angular acceleration in the eyeball's own frame (camera frame) |
| `pupil_axis` | major, minor | mm | Major and minor axes of the fitted pupil ellipse |
| `eye_in_head` | adduction, elevation | rad | Anatomical gaze angles: adduction (positive = toward nose), elevation (positive = up) |

> Keypoint positions (tear_duct, outer_eye, pupil_center, p1–p8) are not saved here — use `*_eye_trajectories_resampled.csv` instead.

---

## `gaze_kinematics/left_gaze_kinematics.csv` and `gaze_kinematics/right_gaze_kinematics.csv`

World-space gaze — the combination of skull motion and eye rotation. Use these files (not the eye kinematics files) when you want to know where the ferret is actually looking in the room.

All trajectories are in **world coordinates**.

| trajectory | components | units | meaning |
|---|---|---|---|
| `orientation` | w, x, y, z | quaternion | World-space rotation representing the eye pointing direction |
| `angular_velocity_local` | roll, pitch, yaw | rad_s | Gaze angular velocity in the eye's own frame |
| `angular_acceleration_local` | roll, pitch, yaw | rad_s2 | Gaze angular acceleration in eye frame |
| `keypoint__gaze_target` | x, y, z | mm | World-space point the eye is directed toward |
| `gaze_angle` | horizontal, vertical | degrees | Gaze direction as spherical angles: horizontal (positive = right), vertical (positive = up) |

> Not saved: `position`, `linear_velocity`, `linear_acceleration`, `angular_velocity_global`, `angular_acceleration_global`, `keypoint__eyeball_center`, `keypoint__pupil_center`. For 3D arrow origins in Rerun, use `skull_kinematics.keypoint_trajectories["left_eye"]` / `["right_eye"]` — lazily recomputed from skull position and orientation on load.

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

Each row gives the world-space direction of one coordinate axis of the eye frame, for one frame. This is useful for verifying that the eye-to-skull-to-world coordinate transform is correct.

---

## `skull_and_spine_trajectories_resampled.csv`

Raw 3D positions of the skull and spine markers, resampled to the common pipeline timestamps. Unlike `skull_kinematics.csv`, this contains **only positions** — no orientation, velocity, or acceleration.

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
