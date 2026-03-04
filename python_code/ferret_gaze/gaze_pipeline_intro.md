# Gaze Pipeline

## Skellyclicker
- DLC tracking
## Eye Analysis
- Butterworth Filter
- Alignment (Tear duct to outer eye on the horizontal)
## Eye Kinematics (Process Ferret Eye Data)

### get camera centered positions
- Estimate camera/eye distance
  - (cam tip → eye point)
- Estimate eye width (px)
  - norm (mean_outer_eye - mean_tear_duct)
	  - maybe makes more sense to compute medians?
	  - Could also compute mean only within percentile range 
  - Maybe just compute this from small # of (good) frames
	  - Do zscore/mean subtract, only compute distance for frames where points are close to 0,0 mean position
- Scale to mm (assume 7mm diameter eye) - could be close or a slight underestimation - assumes tear duct to outer eye is approximately full diameter of eye ball - we could get an estimate of that very soon
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
	- Enforce distance (eyeball radius) - another spot where eyeball center is important
		- plot 2d distribution of distance to center, look at size of deviations in each direction and compare elevation and adduction - it should look like an oval (adduction > elevation)
		- In human babies natural gaze changes over development, so we care mostly about adult animals
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
- ⚠ Rest gaze is median of gaze trajectory
	- Consider trying do a percentile filter and then computing mean or median
	- This could remove saccades from the data before getting rest gaze
	- Also consider calculating at the beginning of the recording (or during exploration phase but not chase phase)
- Eye direction is outer eye – tear duct (X = negative for right eye)
- Y direction is cross product of rest gaze and open eye direction
	- we may be interested in a separate geometric "resting" position
	- rest gaze should correspond to neutral eye position (eye muscles are relaxed)
	- If we're not as interested in comparing gaze between sessions it matters less
	- Main concern is if its correct for geometric calculations (is y axis getting messed up?)
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

- This needs another rotation to match where neutral eye position is in relation to the skull

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
	  - The CR points will be brightest when eye is more aligned with camera, and where pupil is more circular
		  - Check if CR locations are consistently pointing to the same space, or if they are moving
		  - What is the distribution?
	  - Would need an offset for where LEDs are in relation to the camera sensor, and where leds are in relation to tracked camera point
	  - could use two CRs to get separate measurements 
	  - may need to clean up the CR points with a centroid position
- If the ferrets make directed eye saccades to known small points on monitor, we'll know where they're looking (similar to human eye tracker calibration)
	- This will be offset by the kappa angle, so the error will depend on our estimate of the kappa angle
- Could do a ct scan to determine this empirically
	- Or do photogrammetry to measure it exactly
	- We could then hard code this rotation
- Could add visible locations in area (small leds spaced along bottom of monitors?) to calibrate
	- estimate reflection vectors in eye vs 3d locations of eyes pulled from behavioral cameras
- Could align neutral gaze to plane of orbital bones in 3d model of skull


- Change how eye center/rest gaze is calculated
	- There appears to be an offset
	- Not sure which (eye center or rest gaze) is the issue
	- There are sessions where mean/median eye position is **not** a plausible rest gaze
	- Also possible outliers are messing up mean eye center calculation


## Potential sources of error
- Gap between eye center in relation to tracked eyeball position?
- Measurement error in distance between camera and eye (changes with age?)
- Potential error in tear duct to outer eye distance
- Potential eyeball radius size error 
	- Can run uncertainty differences through the pipeline to estimate error
- Also error in rotating eye onto skull
	- If we use the CR method, we should be able to figure out some expected error based on distance to camera - there will also be a lot of room for error in distance between tracked points and geometric points


- For some of this, we can put guesstimates into pipeline and see how much they affect gaze angle



First clean up pipeline, make sure everything makes sense (mostly does)
Next, figure out sources of error and get an estimate
- Get variance of rotation from CR vector to expected CR vector (eye center to corrected cam tip point)
Figure out how Kerr 2025 is solving this problem (rotate eye onto skull correctly) - likely goes back to supplementary methods from Kerr 2013