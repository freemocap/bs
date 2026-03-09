# Gaze Pipeline

## Skellyclicker

- DLC tracking
## Eye Analysis
- Butterworth Filter
- Alignment (Pupil to outer eye)
## Eye Kinematics (Process Ferret Eye Data)

### get camera centered positions
- Estimate camera/eye distance
  - (cam tip → eye point)
- Estimate eye width (px)
  - norm (mean_outer_eye - mean_tear_duct)
- Scale to mm (assume 7mm diameter eye)
- Calculate pupil centers (mean(P1–P8))
- Project eye points onto sphere
- Shift origin to eye center
  - ⚠ eye_center = mean(outer_eye, tear duct)
	- Not sure if this is being calculated correctly, looks wrong in videos
	- May need to use median, or remove outliers before mean
  - origin shifted with `points – eye_center`
- Convert to mm
- Compute Euclidean Distance from center
  - D² = x² + y²
- Enforce distance (eyeball radius)
- Calculate Z offset
  - x² + y² + z² = R², so Z² = √(R² – D²)
- Calculate Z height
  - z = camera_distance – z_offset
- Get points 3d
- Project non-eyeball points
	- Shift origin to center, convert to mm
	- Set Z to plane at front of eye
	  - Z = camera_distance – eyeball radius
- Eye center can be (0, 0, Z)
  - (center of eye sphere)
- Gaze direction calculated
	- Vector from center of eye to pupil center
	- Vector is normalized to length 1
	  - (x, y, –z)
	  - (0, 0, –1) → pointing at camera
### Back to Process Ferret Eye Kinematics
- Rest gaze is median of gaze trajectory
- Eye direction is outer eye – tear duct (X = negative for right eye)
- Y direction is cross product of rest gaze and open eye direction
- Compute camera to eye rotation
  - uses rest gaze as Z, treats this as ground truth vector
	  - *this be CR instead?* - No, but will need a correction for CR later
  -Approximate Y is normalized to orthogonal w/ Gram-Schmidt
  - X (L/R) is computed with cross product (X points left)
  - R = [X, Y, Z] *(vertically stacked matrix)*
- Convert gaze directions to eye frame:
  - `(R @ gaze_directions.T).T`
  - Use rotation R quaternions to transform all points
  - `(R_camera @ points.T).T`
  - Rotates after translating to origin (mean eye center)
## Calculate Ferret Gaze
- Get eye to skull rotation matrix
  - Hard coded geometry
  - Just swaps axes 

|     | right      | left       |
| --- | ---------- | ---------- |
| X   | [1, 0, 0]  | [-1, 0, 0] |
| Y   | [0, 0, 1]  | [0, 0, 1]  |
| Z   | [0, -1, 0] | [0, 1, 0]  |

- +X: subject left → skull +/-X (looking left)
- +Z up → skull +Z (up)
- Z: gaze → skull +/- Y (out from eye / out from Y axis on skull orthogonal to nose and up)

- Compute world position of eye center (gaze position) for each frame
  - `world_pos = skull_pos + q_skull.rotate(eye_pos_skull)`
- Compute world orientation of eye
  - `q_gaze_world = q_skull * q_eye_to_skull * q_eye`
- Norm everything

## Potential Changes
- Maybe add rotation correction to ensure eye center → CR vector points at camera?
	- **current eye in head rotation**
	  - Uses camera to eye rotation
	  - Z (rest gaze) is ground truth (up)
	  - Y is computed, X (L/R) is orthogonal to either
	  - Issue is eye axes are aligned totally to head axes, but we expect rotation here
- Change how eye center/rest gaze is calculated
	- There appears to be an offset
	- Not sure which (eye center or rest gaze) is the issue
	- There are sessions where mean/median eye position is **not** a plausible rest gaze
	- Also possible outliers are messing up mean eye center calculation


## Potential sources of error
- Gap between eye center in relation to tracked eyeball position?
- 