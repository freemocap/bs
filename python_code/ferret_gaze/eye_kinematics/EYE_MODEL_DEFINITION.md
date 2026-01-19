# Eye Kinematics Architecture

## Overview

The eye kinematics model separates two physically distinct components:

1. **Eyeball (RigidBodyKinematics)**: Rotates within the socket
   - Markers: `eyeball_center`, `pupil_center`, `p1-p8`
   - These markers **rotate** with the eye orientation

2. **Socket Landmarks (SocketLandmarks)**: Fixed to skull
   - Markers: `tear_duct`, `outer_eye`
   - These markers do **not** rotate with the eye

## Coordinate System

Both eyes use **right-handed coordinate systems** with identical world orientation:

| Axis | Direction |
|------|-----------|
| +X | Anterior (gaze at rest) |
| +Y | Subject's left |
| +Z | Superior (up) |

Because +Y always points to **subject's left**, anatomical interpretation differs by eye:

| Eye | +Y is... | -Y is... |
|-----|----------|----------|
| Right | Medial (toward nose) | Lateral |
| Left | Lateral (away from nose) | Medial |

## Anatomical Angle Accessors

The raw Euler angles (`roll`, `pitch`, `yaw`) have different anatomical meanings for each eye. The **anatomical accessors** apply sign corrections:

| Accessor | Positive Direction | Negative Direction |
|----------|-------------------|-------------------|
| `adduction_angle` | Gaze toward nose | Gaze away from nose (abduction) |
| `elevation_angle` | Gaze upward | Gaze downward (depression) |
| `torsion_angle` | Extorsion (top rotates laterally) | Intorsion (top rotates medially) |

### Sign Correction Logic

```python
# Horizontal (adduction/abduction)
adduction = yaw * (1.0 if eye_side == "right" else -1.0)

# Vertical (elevation/depression) — no flip needed
elevation = pitch

# Torsion (extorsion/intorsion)
torsion = roll * (1.0 if eye_side == "right" else -1.0)
```

## Socket Landmark Positions

| Eye | Tear Duct Y | Outer Eye Y |
|-----|-------------|-------------|
| Right | Positive (+Y = medial) | Negative |
| Left | Negative (-Y = medial) | Positive |

## Usage

### Anatomical Analysis

```python
# Consistent meaning for both eyes:
kinematics.adduction_angle      # + = gaze toward nose
kinematics.elevation_angle      # + = gaze up
kinematics.torsion_angle        # + = top of eye tilts laterally

kinematics.adduction_velocity   # + = adducting
kinematics.elevation_velocity   # + = elevating  
kinematics.torsion_velocity     # + = extorting
```

### Raw Geometric Values

```python
# Different meanings for left vs right:
kinematics.yaw      # + = toward +Y (subject's left)
kinematics.roll     # + = top tilts toward -Y (subject's right)
kinematics.pitch    # + = upward (same for both)

kinematics.angular_velocity_global  # [ωx, ωy, ωz] in eye frame
```