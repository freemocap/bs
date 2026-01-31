"""
Ferret Kinematics Data Primer
=============================

This script demonstrates how to load ferret kinematics data from disk
into Pydantic model classes and access various properties for analysis.

Run cell-by-cell in VS Code Interactive mode (Ctrl+Enter on each cell)
or run the whole script.


Architecture Overview
---------------------

The ferret gaze pipeline produces kinematics data stored in two file types:

    1. `*_reference_geometry.json` - Static geometry of keypoints on a rigid body
       (e.g., where the nose, eyes, and ears are positioned relative to skull center)

    2. `*_kinematics.csv` - Time-varying pose data (position, orientation, velocities)

These files are loaded into Pydantic model classes that provide:
    - Type validation and data integrity checks
    - Lazy computation of derived quantities (velocity, acceleration, Euler angles)
    - Convenient property accessors for common analysis tasks
    - Numpy arrays under the hood for numerical performance


Key Data Classes
----------------

RigidBodyKinematics:
    General rigid body motion over time. Stores position (N, 3) and orientation
    as quaternions (N, 4). Lazily computes velocity, acceleration, angular velocity,
    angular acceleration, and Euler angles (roll, pitch, yaw) on first access.

    Used for: skull, any tracked rigid object

    Core properties:
        .timestamps         - (N,) array of timestamps in seconds
        .position_xyz       - (N, 3) position in mm
        .quaternions_wxyz   - (N, 4) orientation as [w, x, y, z] quaternions
        .velocity_xyz       - (N, 3) linear velocity in mm/s
        .acceleration_xyz   - (N, 3) linear acceleration in mm/s²
        .angular_velocity_global   - (N, 3) angular velocity in world frame (rad/s)
        .angular_velocity_local    - (N, 3) angular velocity in body frame (rad/s)
        .roll, .pitch, .yaw        - Timeseries objects of Euler angles (rad)
        .keypoint_trajectories     - All keypoints transformed to world coordinates

FerretEyeKinematics:
    Eye-specific kinematics. Contains a RigidBodyKinematics for the eyeball rotation,
    plus socket landmarks (tear duct, outer eye) and tracked pupil positions.
    Provides anatomically consistent angle conventions (adduction, elevation, torsion)
    that have the same meaning for both left and right eyes.

    Core properties:
        .eyeball            - The underlying RigidBodyKinematics for eyeball rotation
        .eye_side           - 'left' or 'right'
        .gaze_directions    - (N, 3) unit vectors pointing where the eye is looking
        .azimuth_radians    - Horizontal gaze angle (rotation in XZ plane)
        .elevation_radians  - Vertical gaze angle
        .adduction_angle    - Timeseries: positive = toward nose, negative = away
        .elevation_angle    - Timeseries: positive = up, negative = down
        .torsion_angle      - Timeseries: positive = extorsion, negative = intorsion
        .socket_landmarks   - SocketLandmarks with tear_duct_mm, outer_eye_mm
        .tracked_pupil      - TrackedPupil with actual detected pupil positions

Timeseries:
    A 1D time-varying scalar bundled with timestamps.

    Properties:
        .timestamps  - (N,) array of timestamps
        .values      - (N,) array of values
        .name        - Identifier string

ReferenceGeometry:
    Static geometry defining keypoint positions in the body-fixed frame and
    the coordinate frame definition (origin, axis directions).


Directory Structure (Pipeline Output)
-------------------------------------

    {clip_folder}/analyzable_output/
    ├── common_timestamps.npy           # Shared timestamps after resampling
    ├── skull_kinematics/
    │   ├── skull_kinematics.csv        # → loads into RigidBodyKinematics
    │   └── skull_reference_geometry.json
    ├── left_eye_kinematics/
    │   ├── left_eye_kinematics.csv     # → loads into FerretEyeKinematics
    │   └── left_eye_reference_geometry.json
    ├── right_eye_kinematics/
    │   ├── right_eye_kinematics.csv    # → loads into FerretEyeKinematics
    │   └── right_eye_reference_geometry.json
    └── gaze_kinematics/                # Eye rotations in world coordinates
        ├── left_gaze_kinematics.csv
        └── right_gaze_kinematics.csv

All data in analyzable_output/ has been resampled to common timestamps,
so skull and eye data can be directly compared frame-by-frame.
"""

# %% [markdown]
# # Setup and Configuration

# %%
# =============================================================================
# CONFIGURATION - Edit this path for your setup
# =============================================================================

from pathlib import Path

# Path to the analyzable_output directory from the gaze pipeline
ANALYZABLE_OUTPUT_DIR = Path(
    r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\analyzable_output"
)

# Verify the directory exists
if not ANALYZABLE_OUTPUT_DIR.exists():
    raise FileNotFoundError(
        f"Analyzable output directory not found: {ANALYZABLE_OUTPUT_DIR}\n"
        f"Make sure you've run the gaze pipeline first, or update the path above."
    )

# %%
# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Core kinematics loader
from python_code.kinematics_core.kinematics_serialization import load_kinematics

# Ferret eye kinematics loader
from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import (
    load_ferret_eye_kinematics_from_directory,
)

print("Imports successful!")

# %% [markdown]
# # Part 1: Loading RigidBodyKinematics (Skull)
#
# `RigidBodyKinematics` represents the motion of a rigid body over time.
# It stores position and orientation, and lazily computes derived quantities
# like velocity, acceleration, and angular velocity on first access.

# %%
# =============================================================================
# LOADING SKULL KINEMATICS
# =============================================================================

skull_dir = ANALYZABLE_OUTPUT_DIR / "skull_kinematics"

skull = load_kinematics(
    reference_geometry_path=skull_dir / "skull_reference_geometry.json",
    kinematics_csv_path=skull_dir / "skull_kinematics.csv",
)

print(f"Loaded: {skull.name}")
print(f"Type: {type(skull).__name__}")

# %%
# =============================================================================
# BASIC PROPERTIES
# =============================================================================

print("=" * 60)
print("BASIC PROPERTIES")
print("=" * 60)

print(f"Name: {skull.name}")
print(f"Number of frames: {skull.n_frames}")
print(f"Duration: {skull.duration:.2f} seconds")
print(f"Framerate: {skull.framerate_hz:.2f} Hz")
print(f"Timestamp range: {skull.timestamps[0]:.3f} - {skull.timestamps[-1]:.3f} s")

# %%
# =============================================================================
# POSITION AND ORIENTATION (Raw Arrays)
# =============================================================================

print("=" * 60)
print("POSITION AND ORIENTATION")
print("=" * 60)

# Position: (N, 3) array in mm
print(f"Position shape: {skull.position_xyz.shape}")
print(f"Position at frame 0: {skull.position_xyz[0]} mm")
print(f"Position at frame 100: {skull.position_xyz[100]} mm")

# Orientation as quaternions: (N, 4) array [w, x, y, z]
print(f"\nQuaternion shape: {skull.quaternions_wxyz.shape}")
print(f"Quaternion at frame 0: {skull.quaternions_wxyz[0]}")

# %%
# =============================================================================
# DERIVED QUANTITIES: VELOCITY AND ACCELERATION
# =============================================================================

print("=" * 60)
print("VELOCITY AND ACCELERATION (Lazily Computed)")
print("=" * 60)

# These are computed on first access from position data
# Linear velocity: (N, 3) array in mm/s
print(f"Velocity shape: {skull.velocity_xyz.shape}")
print(f"Velocity at frame 100: {skull.velocity_xyz[100]} mm/s")

# Linear acceleration: (N, 3) array in mm/s²
print(f"\nAcceleration shape: {skull.acceleration_xyz.shape}")
print(f"Acceleration at frame 100: {skull.acceleration_xyz[100]} mm/s²")

# Speed (scalar magnitude of velocity) - returns a Timeseries
print(f"\nSpeed at frame 100: {skull.speed.values[100]:.2f} mm/s")
print(f"Max speed: {np.max(skull.speed.values):.2f} mm/s")

# %%
# =============================================================================
# ANGULAR VELOCITY AND ACCELERATION
# =============================================================================

print("=" * 60)
print("ANGULAR VELOCITY AND ACCELERATION")
print("=" * 60)

# Global angular velocity: (N, 3) in rad/s (world frame)
# This describes rotation as seen by an external observer
print(f"Angular velocity (global) shape: {skull.angular_velocity_global.shape}")
print(f"Angular velocity (global) at frame 100: {skull.angular_velocity_global[100]} rad/s")

# Local angular velocity: (N, 3) in rad/s (body frame)
# This describes rotation in the body's own coordinate system
print(f"\nAngular velocity (local) shape: {skull.angular_velocity_local.shape}")
print(f"Angular velocity (local) at frame 100: {skull.angular_velocity_local[100]} rad/s")

# Angular speed (scalar magnitude) - returns a Timeseries
print(f"\nAngular speed at frame 100: {np.rad2deg(skull.angular_speed.values[100]):.2f} deg/s")

# %%
# =============================================================================
# EULER ANGLES (Roll, Pitch, Yaw) as Timeseries
# =============================================================================

print("=" * 60)
print("EULER ANGLES")
print("=" * 60)

# Roll, pitch, yaw are Timeseries objects (timestamps + values bundled together)
roll = skull.roll
print(f"Roll type: {type(roll).__name__}")
print(f"Roll has: .timestamps ({roll.timestamps.shape}), .values ({roll.values.shape}), .name ('{roll.name}')")

# Access values in radians
print(f"\nRoll at frame 100: {roll.values[100]:.4f} rad = {np.rad2deg(roll.values[100]):.2f} deg")
print(f"Pitch at frame 100: {skull.pitch.values[100]:.4f} rad = {np.rad2deg(skull.pitch.values[100]):.2f} deg")
print(f"Yaw at frame 100: {skull.yaw.values[100]:.4f} rad = {np.rad2deg(skull.yaw.values[100]):.2f} deg")

# %%
# =============================================================================
# KEYPOINT TRAJECTORIES
# =============================================================================

print("=" * 60)
print("KEYPOINT TRAJECTORIES")
print("=" * 60)

# The reference geometry defines keypoints in body-fixed coordinates
# These are transformed to world coordinates using the pose at each frame
print(f"Available keypoints: {skull.keypoint_names}")

keypoint_trajectories = skull.keypoint_trajectories
print(f"\nKeypoint trajectories type: {type(keypoint_trajectories).__name__}")
print(f"Full array shape: {keypoint_trajectories.trajectories_fr_id_xyz.shape}")  # (N_frames, N_keypoints, 3)

# Get trajectory for a specific keypoint by name
nose_trajectory = keypoint_trajectories["nose"]
print(f"\n'nose' trajectory shape: {nose_trajectory.shape}")
print(f"'nose' position at frame 0: {nose_trajectory[0]} mm")
print(f"'nose' position at frame 100: {nose_trajectory[100]} mm")

# %% [markdown]
# # Part 2: Loading FerretEyeKinematics
#
# `FerretEyeKinematics` extends `RigidBodyKinematics` with eye-specific data:
# - Socket landmarks (tear duct, outer eye) - fixed relative to skull
# - Tracked pupil positions from video
# - Gaze direction calculations
# - Anatomical angle conventions (adduction, elevation, torsion)

# %%
# =============================================================================
# LOADING EYE KINEMATICS
# =============================================================================

left_eye = load_ferret_eye_kinematics_from_directory(
    input_directory=ANALYZABLE_OUTPUT_DIR / "left_eye_kinematics",
    eye_name="left_eye",
)

right_eye = load_ferret_eye_kinematics_from_directory(
    input_directory=ANALYZABLE_OUTPUT_DIR / "right_eye_kinematics",
    eye_name="right_eye",
)

print(f"Loaded left eye: {left_eye.name}, side: {left_eye.eye_side}")
print(f"Loaded right eye: {right_eye.name}, side: {right_eye.eye_side}")

# %%
# =============================================================================
# EYE BASIC PROPERTIES
# =============================================================================

print("=" * 60)
print("EYE BASIC PROPERTIES")
print("=" * 60)

print(f"Eye name: {left_eye.name}")
print(f"Eye side: {left_eye.eye_side}")  # 'left' or 'right'
print(f"Number of frames: {left_eye.n_frames}")
print(f"Duration: {left_eye.duration_seconds:.2f} seconds")
print(f"Framerate: {left_eye.framerate_hz:.2f} Hz")

# %%
# =============================================================================
# ACCESSING THE UNDERLYING EYEBALL (RigidBodyKinematics)
# =============================================================================

print("=" * 60)
print("EYEBALL (RigidBodyKinematics)")
print("=" * 60)

# The eyeball is a RigidBodyKinematics representing the eyeball's rotation
# All RigidBodyKinematics properties are available through it
eyeball = left_eye.eyeball
print(f"Eyeball type: {type(eyeball).__name__}")
print(f"Eyeball name: {eyeball.name}")
print(f"\nQuaternions shape: {eyeball.quaternions_wxyz.shape}")
print(f"Angular velocity (global) shape: {eyeball.angular_velocity_global.shape}")

# %%
# =============================================================================
# GAZE DIRECTION
# =============================================================================

print("=" * 60)
print("GAZE DIRECTION")
print("=" * 60)

# Gaze direction: (N, 3) unit vectors
# At rest, gaze = [0, 0, 1] (looking along +Z axis)
gaze = left_eye.gaze_directions
print(f"Gaze directions shape: {gaze.shape}")
print(f"Gaze at frame 0: {gaze[0]}")
print(f"Gaze at frame 100: {gaze[100]}")

# Verify it's a unit vector
print(f"\nGaze magnitude at frame 100: {np.linalg.norm(gaze[100]):.6f} (should be 1.0)")

# %%
# =============================================================================
# GAZE ANGLES: AZIMUTH AND ELEVATION
# =============================================================================

print("=" * 60)
print("AZIMUTH AND ELEVATION")
print("=" * 60)

# Azimuth: horizontal gaze angle (rotation in XZ plane)
# Positive = looking toward +X (subject's left)
print(f"Azimuth at frame 100: {left_eye.azimuth_radians[100]:.4f} rad = {left_eye.azimuth_degrees[100]:.2f} deg")

# Elevation: vertical gaze angle
# Positive = looking up (+Y)
print(f"Elevation at frame 100: {left_eye.elevation_radians[100]:.4f} rad = {left_eye.elevation_degrees[100]:.2f} deg")

# %%
# =============================================================================
# ANATOMICAL ANGLES (Consistent across eyes)
# =============================================================================

print("=" * 60)
print("ANATOMICAL ANGLES")
print("=" * 60)

# These return Timeseries objects with anatomically consistent meanings
# The signs are adjusted so the same motion has the same sign for both eyes

# Adduction angle (toward nose)
# Positive = adduction (looking toward nose)
# Negative = abduction (looking away from nose)
adduction = left_eye.adduction_angle
print(f"Adduction type: {type(adduction).__name__}")
print(f"Adduction at frame 100: {np.rad2deg(adduction.values[100]):.2f} deg")

# Elevation angle
# Positive = looking up
# Negative = looking down
elevation = left_eye.elevation_angle
print(f"Elevation at frame 100: {np.rad2deg(elevation.values[100]):.2f} deg")

# Torsion angle (rotation around gaze axis)
# Positive = extorsion (top of eye rotates away from nose)
# Negative = intorsion (top of eye rotates toward nose)
torsion = left_eye.torsion_angle
print(f"Torsion at frame 100: {np.rad2deg(torsion.values[100]):.2f} deg")

# %%
# =============================================================================
# ANATOMICAL ANGULAR VELOCITIES
# =============================================================================

print("=" * 60)
print("ANATOMICAL ANGULAR VELOCITIES")
print("=" * 60)

# These return Timeseries objects for angular velocity with consistent anatomical meaning
print(f"Adduction velocity at frame 100: {np.rad2deg(left_eye.adduction_velocity.values[100]):.2f} deg/s")
print(f"Elevation velocity at frame 100: {np.rad2deg(left_eye.elevation_velocity.values[100]):.2f} deg/s")
print(f"Torsion velocity at frame 100: {np.rad2deg(left_eye.torsion_velocity.values[100]):.2f} deg/s")

# %%
# =============================================================================
# SOCKET LANDMARKS (Fixed relative to skull)
# =============================================================================

print("=" * 60)
print("SOCKET LANDMARKS")
print("=" * 60)

socket = left_eye.socket_landmarks
print(f"Socket landmarks type: {type(socket).__name__}")

# Tear duct position: (N, 3) in mm - medial corner of eye
print(f"\nTear duct shape: {socket.tear_duct_mm.shape}")
print(f"Tear duct at frame 0: {socket.tear_duct_mm[0]} mm")

# Outer eye position: (N, 3) in mm - lateral corner of eye
print(f"\nOuter eye shape: {socket.outer_eye_mm.shape}")
print(f"Outer eye at frame 0: {socket.outer_eye_mm[0]} mm")

# Eye opening width over time - returns a Timeseries
eye_width = socket.eye_opening_width_mm
print(f"\nEye opening width at frame 0: {eye_width.values[0]:.2f} mm")

# Shorthand accessors directly on FerretEyeKinematics
print(f"\nDirect access - tear_duct_mm shape: {left_eye.tear_duct_mm.shape}")
print(f"Direct access - outer_eye_mm shape: {left_eye.outer_eye_mm.shape}")

# %%
# =============================================================================
# TRACKED PUPIL (Actual detected positions from video)
# =============================================================================

print("=" * 60)
print("TRACKED PUPIL")
print("=" * 60)

tracked_pupil = left_eye.tracked_pupil
print(f"Tracked pupil type: {type(tracked_pupil).__name__}")

# Pupil center: (N, 3) in mm (eye-centered coordinates)
print(f"\nPupil center shape: {tracked_pupil.pupil_center_mm.shape}")
print(f"Pupil center at frame 0: {tracked_pupil.pupil_center_mm[0]} mm")

# Pupil boundary points: (N, 8, 3) - 8 points around pupil edge
print(f"\nPupil points shape: {tracked_pupil.pupil_points_mm.shape}")
print(f"Pupil point 1 at frame 0: {tracked_pupil.pupil_points_mm[0, 0, :]} mm")

# Shorthand accessors
print(f"\nDirect access - tracked_pupil_center shape: {left_eye.tracked_pupil_center.shape}")
print(f"Direct access - tracked_pupil_points shape: {left_eye.tracked_pupil_points.shape}")

# %% [markdown]
# # Part 3: Visualization
#
# Single figure with two columns:
# - Left column: Skull kinematics (position + orientation)
# - Right column: Eye kinematics (gaze angles for both eyes)
#
# Main plots show a 10-second window; insets show the full timeseries.
# All timeseries are centered on zero.
#
# Color coding by rotation axis (complementary pairs for left/right eye):
# - X axis: Red (left) / Cyan (right) — Roll, Elevation
# - Y axis: Green (left) / Magenta (right) — Pitch, Azimuth, Adduction
# - Z axis: Blue (left) / Orange (right) — Yaw, Torsion

# %%
# =============================================================================
# PLOTTING SETUP
# =============================================================================

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Plotting style: lines with small dots
MARKER_SIZE = 2
LINE_STYLE = '-o'
ALPHA = 0.8

# Colors by rotation axis (RGB = XYZ)
# Left eye: primary colors
C_X = '#e41a1c'          # Red: X axis (Position X, Roll, Elevation)
C_Y = '#4daf4a'          # Green: Y axis (Position Y, Pitch, Azimuth/Adduction)
C_Z = '#377eb8'          # Blue: Z axis (Position Z, Yaw, Torsion)

# Right eye: complementary colors (maximally distinguishable)
C_X_R = '#00bfc4'        # Cyan (complement of red) for right eye elevation
C_Y_R = '#c77cff'        # Magenta/purple (complement of green) for right eye azimuth/adduction
C_Z_R = '#f8766d'        # Orange/coral (complement of blue) for right eye torsion

# Left/Right eye distinction: also use line style
LEFT_STYLE = '-o'        # Solid line with circle markers
RIGHT_STYLE = '-s'       # Solid line with square markers (color does the work now)

C_SKULL = '#555555'      # Gray for skull

# 10-second main window (±5 seconds from center)
window_half = 5.0
window_center = skull.timestamps[0] + skull.duration / 2
window_start = window_center - window_half
window_end = window_center + window_half


def add_full_timeseries_inset(
    ax: plt.Axes,
    time_data: np.ndarray,
    y_data_list: list[np.ndarray],
    colors: list[str],
    linestyles: list[str] | None = None,
) -> None:
    """Add a small inset showing the full timeseries."""
    axins = inset_axes(ax, width="25%", height="35%", loc='upper right', borderpad=1.5)

    if linestyles is None:
        linestyles = ['-'] * len(y_data_list)

    for y_data, color, ls in zip(y_data_list, colors, linestyles):
        axins.plot(time_data, y_data, ls, linewidth=0.5, color=color, alpha=0.7)

    # Zero line and highlight region
    axins.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    axins.axvspan(window_start, window_end, alpha=0.2, color='yellow')

    axins.set_xlim(time_data[0], time_data[-1])
    axins.tick_params(labelsize=6)
    axins.grid(True, alpha=0.3)
    for spine in axins.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1.0)


def get_window_mask(time_data: np.ndarray) -> np.ndarray:
    """Get boolean mask for the 10-second window."""
    return (time_data >= window_start) & (time_data <= window_end)


def center_on_zero(data: np.ndarray) -> np.ndarray:
    """Center data by subtracting the mean."""
    return data - np.nanmean(data)


def set_symmetric_ylim(ax: plt.Axes, data_list: list[np.ndarray]) -> None:
    """Set y-axis limits symmetric around zero based on data."""
    max_abs = max(np.abs(d).max() for d in data_list) * 1.1
    ax.set_ylim(-max_abs, max_abs)


# %%
# =============================================================================
# COMBINED FIGURE: SKULL (left) + EYES (right)
# =============================================================================

fig, axes = plt.subplots(3, 2,  sharex='col')

# Masks for windowing
mask_skull = get_window_mask(skull.timestamps)
t_skull = skull.timestamps[mask_skull]

mask_eye = get_window_mask(left_eye.timestamps)
t_eye = left_eye.timestamps[mask_eye]

# -----------------------------------------------------------------------------
# Prepare skull data (centered on zero)
# -----------------------------------------------------------------------------
skull_x = center_on_zero(skull.x.values)
skull_y = center_on_zero(skull.y.values)
skull_z = center_on_zero(skull.z.values)
skull_roll = center_on_zero(np.rad2deg(skull.roll.values))
skull_pitch = center_on_zero(np.rad2deg(skull.pitch.values))
skull_yaw = center_on_zero(np.rad2deg(skull.yaw.values))

# -----------------------------------------------------------------------------
# Prepare eye data (centered on zero)
# Color coding by rotation axis:
#   Elevation = rotation around X axis → Red
#   Azimuth/Adduction = rotation around Y axis → Green
#   Torsion = rotation around Z axis → Blue
# -----------------------------------------------------------------------------
left_az = center_on_zero(left_eye.azimuth_degrees)
right_az = center_on_zero(right_eye.azimuth_degrees)
left_el = center_on_zero(left_eye.elevation_degrees)
right_el = center_on_zero(right_eye.elevation_degrees)
left_add = center_on_zero(np.rad2deg(left_eye.adduction_angle.values))
right_add = center_on_zero(np.rad2deg(right_eye.adduction_angle.values))
left_el_anat = center_on_zero(np.rad2deg(left_eye.elevation_angle.values))
right_el_anat = center_on_zero(np.rad2deg(right_eye.elevation_angle.values))
left_tor = center_on_zero(np.rad2deg(left_eye.torsion_angle.values))
right_tor = center_on_zero(np.rad2deg(right_eye.torsion_angle.values))

# -----------------------------------------------------------------------------
# LEFT COLUMN: SKULL KINEMATICS
# -----------------------------------------------------------------------------

# Row 0: Roll (rotation around X → Red)
axes[0, 0].plot(t_skull, skull_roll[mask_skull], LINE_STYLE, markersize=MARKER_SIZE,
                color=C_X, alpha=ALPHA, label='Roll')
axes[0, 0].axhline(y=0, color='k', linewidth=1, alpha=0.5)
axes[0, 0].set_ylabel('Roll (deg)')
set_symmetric_ylim(axes[0, 0], [skull_roll[mask_skull]])
axes[0, 0].legend(loc='upper left', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)
add_full_timeseries_inset(axes[0, 0], skull.timestamps, [skull_roll], [C_X])

# Row 1: Pitch (rotation around Y → Green)
axes[1, 0].plot(t_skull, skull_pitch[mask_skull], LINE_STYLE, markersize=MARKER_SIZE,
                color=C_Y, alpha=ALPHA, label='Pitch')
axes[1, 0].axhline(y=0, color='k', linewidth=1, alpha=0.5)
axes[1, 0].set_ylabel('Pitch (deg)')
set_symmetric_ylim(axes[1, 0], [skull_pitch[mask_skull]])
axes[1, 0].legend(loc='upper left', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)
add_full_timeseries_inset(axes[0, 0], skull.timestamps, [skull_pitch], [C_Y])

# Row 2: Yaw (rotation around Z → Blue)
axes[2, 0].plot(t_skull, skull_yaw[mask_skull], LINE_STYLE, markersize=MARKER_SIZE,
                color=C_Z, alpha=ALPHA, label='Yaw')
axes[2, 0].axhline(y=0, color='k', linewidth=1, alpha=0.5)
axes[2, 0].set_ylabel('Yaw (deg)')
set_symmetric_ylim(axes[2, 0], [skull_yaw[mask_skull]])
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].legend(loc='upper left', fontsize=9)
axes[2, 0].grid(True, alpha=0.3)
add_full_timeseries_inset(axes[2, 0], skull.timestamps, [skull_yaw], [C_Z])

# -----------------------------------------------------------------------------
# RIGHT COLUMN: EYE KINEMATICS
# Color coding by rotation axis:
#   Elevation = rotation around X axis → Red (left) / Cyan (right)
#   Azimuth/Adduction = rotation around Y axis → Green (left) / Magenta (right)
#   Torsion = rotation around Z axis → Blue (left) / Orange (right)
# -----------------------------------------------------------------------------

# Row 0: Azimuth (horizontal gaze = rotation around Y axis)
axes[0, 1].plot(t_eye, left_az[mask_eye], LEFT_STYLE, markersize=MARKER_SIZE,
                color=C_Y, alpha=ALPHA, label='Left')
axes[0, 1].plot(t_eye, right_az[mask_eye], RIGHT_STYLE, markersize=MARKER_SIZE,
                color=C_Y_R, alpha=ALPHA, label='Right')
axes[0, 1].axhline(y=0, color='k', linewidth=1, alpha=0.5)
axes[0, 1].set_ylabel('Azimuth (deg)')
set_symmetric_ylim(axes[0, 1], [left_az[mask_eye], right_az[mask_eye]])
axes[0, 1].legend(loc='upper left', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Eye Kinematics', fontweight='bold', fontsize=12)
add_full_timeseries_inset(axes[0, 1], left_eye.timestamps, [left_az, right_az],
                          [C_Y, C_Y_R], ['-', '-'])

# Row 1: Elevation (vertical gaze = rotation around X axis)
axes[1, 1].plot(t_eye, left_el[mask_eye], LEFT_STYLE, markersize=MARKER_SIZE,
                color=C_X, alpha=ALPHA, label='Left')
axes[1, 1].plot(t_eye, right_el[mask_eye], RIGHT_STYLE, markersize=MARKER_SIZE,
                color=C_X_R, alpha=ALPHA, label='Right')
axes[1, 1].axhline(y=0, color='k', linewidth=1, alpha=0.5)
axes[1, 1].set_ylabel('Elevation (deg)')
set_symmetric_ylim(axes[1, 1], [left_el[mask_eye], right_el[mask_eye]])
axes[1, 1].legend(loc='upper left', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)
add_full_timeseries_inset(axes[1, 1], left_eye.timestamps, [left_el, right_el],
                          [C_X, C_X_R], ['-', '-'])

# Row 2: Adduction (horizontal anatomical = rotation around Y axis)
axes[2, 1].plot(t_eye, left_add[mask_eye], LEFT_STYLE, markersize=MARKER_SIZE,
                color=C_Y, alpha=ALPHA, label='Left')
axes[2, 1].plot(t_eye, right_add[mask_eye], RIGHT_STYLE, markersize=MARKER_SIZE,
                color=C_Y_R, alpha=ALPHA, label='Right')
axes[2, 1].axhline(y=0, color='k', linewidth=1, alpha=0.5)
axes[2, 1].set_ylabel('Adduction (deg)')
set_symmetric_ylim(axes[2, 1], [left_add[mask_eye], right_add[mask_eye]])
axes[2, 1].legend(loc='upper left', fontsize=9)
axes[2, 1].grid(True, alpha=0.3)
add_full_timeseries_inset(axes[2, 1], left_eye.timestamps, [left_add, right_add],
                          [C_Y, C_Y_R], ['-', '-'])

#
# # Row 3: Torsion (rotation around Z axis)
# axes[3, 1].plot(t_eye, left_tor[mask_eye], LEFT_STYLE, markersize=MARKER_SIZE,
#                 color=C_Z, alpha=ALPHA, label='Left')
# axes[3, 1].plot(t_eye, right_tor[mask_eye], RIGHT_STYLE, markersize=MARKER_SIZE,
#                 color=C_Z_R, alpha=ALPHA, label='Right')
# axes[3, 1].axhline(y=0, color='k', linewidth=1, alpha=0.5)
# axes[3, 1].set_ylabel('Torsion (deg)')
# axes[3, 1].set_xlabel('Time (s)')
# set_symmetric_ylim(axes[3, 1], [left_tor[mask_eye], right_tor[mask_eye]])
# axes[3, 1].legend(loc='upper left', fontsize=9)
# axes[3, 1].grid(True, alpha=0.3)
# add_full_timeseries_inset(axes[3, 1], left_eye.timestamps, [left_tor, right_tor],
#                           [C_Z, C_Z_R], ['-', '-'])


fig.suptitle(f'Ferret Kinematics (10s window centered at t={window_center:.1f}s)\n'
             f'Left eye: Red/Green/Blue | Right eye: Cyan/Magenta/Orange (complementary pairs)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# %%
# =============================================================================
# FIGURE 2: Skull vs Eye Angular Velocity Correlations (3x3 grid)
# =============================================================================

fig2, axes2 = plt.subplots(3, 3, figsize=(14, 12))

# Skull angular velocities (local frame: roll, pitch, yaw)
skull_roll_vel = np.rad2deg(skull.angular_velocity_local[:, 0])
skull_pitch_vel = np.rad2deg(skull.angular_velocity_local[:, 1])
skull_yaw_vel = np.rad2deg(skull.angular_velocity_local[:, 2])

# Eye angular velocities (anatomical)
left_add_vel = np.rad2deg(left_eye.adduction_velocity.values)
right_add_vel = np.rad2deg(right_eye.adduction_velocity.values)
left_el_vel = np.rad2deg(left_eye.elevation_velocity.values)
right_el_vel = np.rad2deg(right_eye.elevation_velocity.values)
left_tor_vel = np.rad2deg(left_eye.torsion_velocity.values)
right_tor_vel = np.rad2deg(right_eye.torsion_velocity.values)

# Organize data for iteration
# Skull columns: Roll (X), Pitch (Y), Yaw (Z)
skull_vels = [skull_roll_vel, skull_pitch_vel, skull_yaw_vel]
skull_labels = ['Skull Roll Velocity (deg/s)', 'Skull Pitch Velocity (deg/s)', 'Skull Yaw Velocity (deg/s)']
skull_colors = [C_X, C_Y, C_Z]

# Eye rows: Adduction (Y), Elevation (X), Torsion (Z) - ordered by rotation axis
eye_vels_left = [left_add_vel, left_el_vel, left_tor_vel]
eye_vels_right = [right_add_vel, right_el_vel, right_tor_vel]
eye_labels = ['Eye Adduction Vel (deg/s)', 'Eye Elevation Vel (deg/s)', 'Eye Torsion Vel (deg/s)']
eye_colors_left = [C_Y, C_X, C_Z]  # Adduction=Y(green), Elevation=X(red), Torsion=Z(blue)
eye_colors_right = [C_Y_R, C_X_R, C_Z_R]  # Complementary colors for right eye

# Plot 3x3 grid
# Rows = eye components, Columns = skull components
for row, (left_vel, right_vel, eye_label, eye_color_l, eye_color_r) in enumerate(
        zip(eye_vels_left, eye_vels_right, eye_labels, eye_colors_left, eye_colors_right)):
    for col, (skull_vel, skull_label, skull_color) in enumerate(zip(skull_vels, skull_labels, skull_colors)):
        ax = axes2[row, col]

        # Scatter with regression for both eyes
        # Left = primary color + circles; Right = complementary color + squares
        sns.regplot(x=skull_vel, y=left_vel, ax=ax,
                    scatter_kws={'alpha': 0.3, 's': 8, 'color': eye_color_l, 'marker': 'o'},
                    line_kws={'color': eye_color_l, 'linewidth': 2},
                    label='Left' if row == 0 and col == 0 else None)
        sns.regplot(x=skull_vel, y=right_vel, ax=ax,
                    scatter_kws={'alpha': 0.3, 's': 8, 'color': eye_color_r, 'marker': 's'},
                    line_kws={'color': eye_color_r, 'linewidth': 2},
                    label='Right' if row == 0 and col == 0 else None)

        # Zero lines
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        # Labels
        if row == 2:
            ax.set_xlabel(skull_label, fontsize=9)
        else:
            ax.set_xlabel('')

        if col == 0:
            ax.set_ylabel(eye_label, fontsize=9)
        else:
            ax.set_ylabel('')

        # Color the column header with skull axis color
        if row == 0:
            ax.set_title(skull_label.replace(' Velocity (deg/s)', ''),
                        fontweight='bold', color=skull_color, fontsize=11)

        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

# Add legend to first plot
axes2[0, 0].legend(loc='upper left', fontsize=8)

fig2.suptitle('Skull vs Eye Angular Velocity Correlations\n(Left: circles | Right: squares)',
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# # Part 4: Working with Timeseries Objects
#
# Many properties return `Timeseries` objects which bundle timestamps with values.

# %%
# =============================================================================
# TIMESERIES PROPERTIES AND METHODS
# =============================================================================

# Get a Timeseries
roll = skull.roll

print("=" * 60)
print("TIMESERIES OBJECT")
print("=" * 60)

print(f"Name: {roll.name}")
print(f"Timestamps shape: {roll.timestamps.shape}")
print(f"Values shape: {roll.values.shape}")

# Access specific values
print(f"\nValue at index 100: {roll.values[100]}")
print(f"Timestamp at index 100: {roll.timestamps[100]}")

# Basic statistics
print(f"\nMin value: {np.min(roll.values):.4f}")
print(f"Max value: {np.max(roll.values):.4f}")
print(f"Mean value: {np.mean(roll.values):.4f}")
print(f"Std value: {np.std(roll.values):.4f}")

# %% [markdown]
# # Part 5: Resampling and Time Manipulation

# %%
# =============================================================================
# RESAMPLING KINEMATICS
# =============================================================================

print("=" * 60)
print("RESAMPLING")
print("=" * 60)

# Create new timestamps at 30 Hz
original_duration = skull.duration
new_framerate = 30.0
new_timestamps = np.arange(0, original_duration, 1.0 / new_framerate)

print(f"Original framerate: {skull.framerate_hz:.2f} Hz")
print(f"Original frames: {skull.n_frames}")
print(f"New framerate: {new_framerate} Hz")
print(f"New frames: {len(new_timestamps)}")

# Resample skull kinematics
# Note: Quaternions are interpolated using SLERP, positions linearly
skull_resampled = skull.resample(new_timestamps)

print(f"\nResampled skull frames: {skull_resampled.n_frames}")
print(f"Resampled framerate: {skull_resampled.framerate_hz:.2f} Hz")


# %% [markdown]
# # Summary
#
# ## Key Classes:
# - `RigidBodyKinematics`: General rigid body motion (skull, any tracked object)
# - `FerretEyeKinematics`: Eye-specific kinematics with gaze, landmarks, pupil
# - `Timeseries`: 1D scalar time series with timestamps
# - `ReferenceGeometry`: Static keypoint positions and coordinate frame definition
#
# ## Loading Data:
# ```python
# # Skull (RigidBodyKinematics)
# skull = load_kinematics(reference_geometry_path, kinematics_csv_path)
#
# # Eye (FerretEyeKinematics)
# eye = load_ferret_eye_kinematics_from_directory(input_directory, eye_name)
# ```
#
# ## Common Properties:
# - `.timestamps`, `.n_frames`, `.duration`, `.framerate_hz`
# - `.position_xyz`, `.velocity_xyz`, `.acceleration_xyz`
# - `.quaternions_wxyz`, `.orientations`
# - `.angular_velocity_global`, `.angular_velocity_local`
# - `.roll`, `.pitch`, `.yaw` (Timeseries)
# - `.keypoint_trajectories`
#
# ## Eye-Specific Properties:
# - `.gaze_directions`, `.azimuth_radians`, `.elevation_radians`
# - `.adduction_angle`, `.elevation_angle`, `.torsion_angle`
# - `.adduction_velocity`, `.elevation_velocity`, `.torsion_velocity`
# - `.socket_landmarks`, `.tracked_pupil`
# - `.eye_side` ('left' or 'right')

# %%
print("\n" + "=" * 60)
print("PRIMER COMPLETE!")
print("=" * 60)
print("\nYou've learned how to:")
print("  ✓ Load RigidBodyKinematics (skull)")
print("  ✓ Load FerretEyeKinematics (eyes)")
print("  ✓ Access position, velocity, acceleration")
print("  ✓ Access angular velocity and Euler angles")
print("  ✓ Work with keypoint trajectories")
print("  ✓ Access eye-specific gaze and anatomical angles")
print("  ✓ Work with socket landmarks and tracked pupil")
print("  ✓ Create basic plots")
