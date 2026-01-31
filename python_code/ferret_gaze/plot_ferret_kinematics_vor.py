"""
Ferret Kinematics VOR Plotter
=============================

Standalone plotting script extracted from ferret_kinematics_primer.py
with fixes for VOR (vestibulo-ocular reflex) analysis.

KEY FIXES FROM AXIS ALIGNMENT ANALYSIS:
1. Uses RAW angular velocity components (x, y, z) instead of anatomical velocities
   for VOR correlation - the anatomical sign correction makes left/right eyes
   have OPPOSITE signs during conjugate movement, destroying VOR correlation
2. Shows all axis combinations in correlation grid to identify correct pairings
3. Includes conjugate movement check (both eyes should move together)
4. Properly labels coordinate frames

COORDINATE SYSTEMS:
- Skull body frame: +X=forward(nose), +Y=left, +Z=down (X×Y)
- Eye frame: +Z=gaze, +Y=up, +X=subject's left
- These frames are NOT aligned! Use correlation grid to find matching axes.

Usage:
    python plot_ferret_kinematics_vor.py [analyzable_output_dir] [--save output_dir]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import NDArray
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_ANALYZABLE_OUTPUT_DIR = Path(
    r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\analyzable_output"
)


# =============================================================================
# PLOTTING STYLE CONSTANTS
# =============================================================================

MARKER_SIZE = 2
LINE_STYLE = '-o'
ALPHA = 0.8

# Colors by rotation axis (RGB = XYZ)
C_X = '#e41a1c'          # Red: X axis
C_Y = '#4daf4a'          # Green: Y axis
C_Z = '#377eb8'          # Blue: Z axis

# Right eye: complementary colors
C_X_R = '#00bfc4'        # Cyan
C_Y_R = '#c77cff'        # Magenta
C_Z_R = '#f8766d'        # Orange

LEFT_STYLE = '-o'
RIGHT_STYLE = '-s'
C_SKULL = '#555555'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_kinematics_data(analyzable_output_dir: Path) -> tuple:
    """Load skull and eye kinematics from directory."""
    # Import here to avoid issues if modules not available
    from python_code.kinematics_core.kinematics_serialization import load_kinematics
    from python_code.ferret_gaze.eye_kinematics.ferret_eye_kinematics_serialization import (
        load_ferret_eye_kinematics_from_directory,
    )

    skull_dir = analyzable_output_dir / "skull_kinematics"
    skull = load_kinematics(
        reference_geometry_path=skull_dir / "skull_reference_geometry.json",
        kinematics_csv_path=skull_dir / "skull_kinematics.csv",
    )

    left_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=analyzable_output_dir / "left_eye_kinematics",
        eye_name="left_eye",
    )

    right_eye = load_ferret_eye_kinematics_from_directory(
        input_directory=analyzable_output_dir / "right_eye_kinematics",
        eye_name="right_eye",
    )

    return skull, left_eye, right_eye


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def center_on_zero(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center data by subtracting the mean."""
    return data - np.nanmean(data)


def set_symmetric_ylim(ax: plt.Axes, data_list: list[NDArray[np.float64]]) -> None:
    """Set y-axis limits symmetric around zero."""
    max_abs = max(np.nanmax(np.abs(d)) for d in data_list) * 1.1
    ax.set_ylim(-max_abs, max_abs)


def compute_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Compute Pearson correlation and p-value."""
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 10:
        return 0.0, 1.0
    result = stats.pearsonr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue)


def add_full_timeseries_inset(
    ax: plt.Axes,
    time_data: NDArray[np.float64],
    y_data_list: list[NDArray[np.float64]],
    colors: list[str],
    window_start: float,
    window_end: float,
    linestyles: list[str] | None = None,
) -> None:
    """Add a small inset showing the full timeseries."""
    axins = inset_axes(ax, width="25%", height="35%", loc='upper right', borderpad=1.5)

    if linestyles is None:
        linestyles = ['-'] * len(y_data_list)

    for y_data, color, ls in zip(y_data_list, colors, linestyles):
        axins.plot(time_data, y_data, ls, linewidth=0.5, color=color, alpha=0.7)

    axins.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    axins.axvspan(window_start, window_end, alpha=0.2, color='yellow')
    axins.set_xlim(time_data[0], time_data[-1])
    axins.tick_params(labelsize=6)
    axins.grid(True, alpha=0.3)
    for spine in axins.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1.0)


# =============================================================================
# LEFT EYE SIGN CORRECTION
# =============================================================================
# The left eye has X and Y flipped in pixels_to_camera_3d, which propagates
# to the angular velocity. To make left and right eyes comparable, we need
# to flip the left eye's ωy (and possibly ωx) signs.
#
# Based on the conjugate movement check showing r=-0.77 for ωy (should be positive),
# we flip the LEFT eye's horizontal (ωy) component.

LEFT_EYE_OMEGA_Y_SIGN = -1.0  # Flip left eye ωy to match right eye convention
LEFT_EYE_OMEGA_X_SIGN = 1.0   # May also need flipping - check conjugate plot
LEFT_EYE_OMEGA_Z_SIGN = -1.0  # Flip left eye ωz based on r=-0.67


def get_corrected_eye_angular_velocity(
    eye,
    component: int,
    apply_left_eye_correction: bool = True,
) -> NDArray[np.float64]:
    """
    Get angular velocity with optional left eye sign correction.

    Args:
        eye: FerretEyeKinematics object
        component: 0=x, 1=y, 2=z
        apply_left_eye_correction: Whether to apply sign correction for left eye

    Returns:
        Angular velocity in deg/s
    """
    omega = np.rad2deg(eye.angular_velocity_global[:, component])

    if apply_left_eye_correction and 'left' in eye.name.lower():
        signs = [LEFT_EYE_OMEGA_X_SIGN, LEFT_EYE_OMEGA_Y_SIGN, LEFT_EYE_OMEGA_Z_SIGN]
        omega = signs[component] * omega

    return omega


# =============================================================================
# FIGURE 1: TIME SERIES OVERVIEW
# =============================================================================

def plot_timeseries_overview(
    skull,
    left_eye,
    right_eye,
    window_duration: float = 10.0,
    output_path: Path | None = None,
) -> None:
    """
    Plot time series of skull and eye kinematics.

    Shows a windowed view with full-timeseries insets.
    Now includes torsion with caveat about unvalidated calculation.
    """
    # Window setup
    window_half = window_duration / 2
    window_center = skull.timestamps[0] + skull.duration / 2
    window_start = window_center - window_half
    window_end = window_center + window_half

    def get_window_mask(time_data: NDArray[np.float64]) -> NDArray[np.bool_]:
        return (time_data >= window_start) & (time_data <= window_end)

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col')

    mask_skull = get_window_mask(skull.timestamps)
    t_skull = skull.timestamps[mask_skull]
    mask_eye = get_window_mask(left_eye.timestamps)
    t_eye = left_eye.timestamps[mask_eye]

    # Prepare skull Euler angles (centered)
    skull_roll = center_on_zero(np.rad2deg(skull.roll.values))
    skull_pitch = center_on_zero(np.rad2deg(skull.pitch.values))
    skull_yaw = center_on_zero(np.rad2deg(skull.yaw.values))

    # Prepare eye azimuth/elevation (centered) - raw, no anatomical correction
    left_az = center_on_zero(left_eye.azimuth_degrees)
    right_az = center_on_zero(right_eye.azimuth_degrees)
    left_el = center_on_zero(left_eye.elevation_degrees)
    right_el = center_on_zero(right_eye.elevation_degrees)

    # Torsion - using raw ωz component (rotation around gaze axis)
    # NOTE: Torsion calculation is NOT validated and may be unreliable!
    left_torsion = center_on_zero(np.rad2deg(left_eye.angular_velocity_global[:, 2]))
    right_torsion = center_on_zero(np.rad2deg(right_eye.angular_velocity_global[:, 2]))

    # LEFT COLUMN: SKULL
    # Row 0: Roll
    axes[0, 0].plot(t_skull, skull_roll[mask_skull], LINE_STYLE, markersize=MARKER_SIZE,
                    color=C_X, alpha=ALPHA, label='Roll (around X)')
    axes[0, 0].axhline(y=0, color='k', linewidth=1, alpha=0.5)
    axes[0, 0].set_ylabel('Roll (deg)')
    axes[0, 0].set_title('Skull Euler Angles (Body Frame)', fontweight='bold')
    set_symmetric_ylim(axes[0, 0], [skull_roll[mask_skull]])
    axes[0, 0].legend(loc='upper left', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    add_full_timeseries_inset(axes[0, 0], skull.timestamps, [skull_roll], [C_X],
                              window_start, window_end)

    # Row 1: Pitch
    axes[1, 0].plot(t_skull, skull_pitch[mask_skull], LINE_STYLE, markersize=MARKER_SIZE,
                    color=C_Y, alpha=ALPHA, label='Pitch (around Y)')
    axes[1, 0].axhline(y=0, color='k', linewidth=1, alpha=0.5)
    axes[1, 0].set_ylabel('Pitch (deg)')
    set_symmetric_ylim(axes[1, 0], [skull_pitch[mask_skull]])
    axes[1, 0].legend(loc='upper left', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    add_full_timeseries_inset(axes[1, 0], skull.timestamps, [skull_pitch], [C_Y],
                              window_start, window_end)

    # Row 2: Yaw
    axes[2, 0].plot(t_skull, skull_yaw[mask_skull], LINE_STYLE, markersize=MARKER_SIZE,
                    color=C_Z, alpha=ALPHA, label='Yaw (around Z)')
    axes[2, 0].axhline(y=0, color='k', linewidth=1, alpha=0.5)
    axes[2, 0].set_ylabel('Yaw (deg)')
    set_symmetric_ylim(axes[2, 0], [skull_yaw[mask_skull]])
    axes[2, 0].legend(loc='upper left', fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)
    add_full_timeseries_inset(axes[2, 0], skull.timestamps, [skull_yaw], [C_Z],
                              window_start, window_end)

    # Row 3: Coordinate frame info
    axes[3, 0].text(0.5, 0.5,
                    "COORDINATE FRAMES:\n\n"
                    "Skull body frame:\n"
                    "  +X = forward (nose)\n"
                    "  +Y = left\n"
                    "  +Z = down (X×Y)\n\n"
                    "Eye frame:\n"
                    "  +Z = gaze direction\n"
                    "  +Y = up\n"
                    "  +X = subject's left\n\n"
                    "⚠️ LEFT EYE has flipped axes!\n"
                    "See VOR correlation grid.",
                    transform=axes[3, 0].transAxes,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].axis('off')

    # RIGHT COLUMN: EYES
    # Row 0: Azimuth (horizontal gaze)
    axes[0, 1].plot(t_eye, left_az[mask_eye], LEFT_STYLE, markersize=MARKER_SIZE,
                    color=C_Y, alpha=ALPHA, label='Left')
    axes[0, 1].plot(t_eye, right_az[mask_eye], RIGHT_STYLE, markersize=MARKER_SIZE,
                    color=C_Y_R, alpha=ALPHA, label='Right')
    axes[0, 1].axhline(y=0, color='k', linewidth=1, alpha=0.5)
    axes[0, 1].set_ylabel('Azimuth (deg)')
    axes[0, 1].set_title('Eye Gaze Angles (Eye Frame)', fontweight='bold')
    set_symmetric_ylim(axes[0, 1], [left_az[mask_eye], right_az[mask_eye]])
    axes[0, 1].legend(loc='upper left', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    add_full_timeseries_inset(axes[0, 1], left_eye.timestamps, [left_az, right_az],
                              [C_Y, C_Y_R], window_start, window_end)

    # Row 1: Elevation (vertical gaze)
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
                              [C_X, C_X_R], window_start, window_end)

    # Row 2: Torsion (rotation around gaze axis)
    axes[2, 1].plot(t_eye, left_torsion[mask_eye], LEFT_STYLE, markersize=MARKER_SIZE,
                    color=C_Z, alpha=ALPHA, label='Left')
    axes[2, 1].plot(t_eye, right_torsion[mask_eye], RIGHT_STYLE, markersize=MARKER_SIZE,
                    color=C_Z_R, alpha=ALPHA, label='Right')
    axes[2, 1].axhline(y=0, color='k', linewidth=1, alpha=0.5)
    axes[2, 1].set_ylabel('Torsion ωz (deg/s)')
    set_symmetric_ylim(axes[2, 1], [left_torsion[mask_eye], right_torsion[mask_eye]])
    axes[2, 1].legend(loc='upper left', fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)
    add_full_timeseries_inset(axes[2, 1], left_eye.timestamps, [left_torsion, right_torsion],
                              [C_Z, C_Z_R], window_start, window_end)

    # Row 3: Torsion warning
    axes[3, 1].text(0.5, 0.5,
                    "⚠️ TORSION WARNING ⚠️\n\n"
                    "Torsion (ωz) calculation is\n"
                    "NOT VALIDATED and may be\n"
                    "unreliable!\n\n"
                    "The pupil tracking does not\n"
                    "reliably capture rotation\n"
                    "around the gaze axis.\n\n"
                    "Use with extreme caution.",
                    transform=axes[3, 1].transAxes,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].axis('off')

    fig.suptitle(f'Ferret Kinematics Overview ({window_duration:.0f}s window at t={window_center:.1f}s)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


# =============================================================================
# FIGURE 2: VOR CORRELATION GRID (FIXED - uses raw angular velocities)
# =============================================================================

def plot_vor_correlation_grid(
    skull,
    left_eye,
    right_eye,
    output_path: Path | None = None,
) -> None:
    """
    Plot 3x3 grid of skull vs eye angular velocity correlations.

    Uses CORRECTED angular velocity (left eye sign flipped) to properly
    show VOR correlations.
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Skull angular velocities - LOCAL frame (body-fixed)
    skull_roll_vel = np.rad2deg(skull.angular_velocity_local[:, 0])
    skull_pitch_vel = np.rad2deg(skull.angular_velocity_local[:, 1])
    skull_yaw_vel = np.rad2deg(skull.angular_velocity_local[:, 2])

    # Eye angular velocities - CORRECTED (left eye sign flipped)
    left_omega_x = LEFT_EYE_OMEGA_X_SIGN * np.rad2deg(left_eye.angular_velocity_global[:, 0])
    left_omega_y = LEFT_EYE_OMEGA_Y_SIGN * np.rad2deg(left_eye.angular_velocity_global[:, 1])
    left_omega_z = LEFT_EYE_OMEGA_Z_SIGN * np.rad2deg(left_eye.angular_velocity_global[:, 2])

    right_omega_x = np.rad2deg(right_eye.angular_velocity_global[:, 0])
    right_omega_y = np.rad2deg(right_eye.angular_velocity_global[:, 1])
    right_omega_z = np.rad2deg(right_eye.angular_velocity_global[:, 2])

    # Skull columns
    skull_vels = [skull_roll_vel, skull_pitch_vel, skull_yaw_vel]
    skull_labels = ['Skull ωroll (X-fwd)', 'Skull ωpitch (Y-left)', 'Skull ωyaw (Z-down)']
    skull_colors = [C_X, C_Y, C_Z]

    # Eye rows - RAW components, same for both eyes (no anatomical sign flip!)
    eye_vels_left = [left_omega_x, left_omega_y, left_omega_z]
    eye_vels_right = [right_omega_x, right_omega_y, right_omega_z]
    eye_labels = ['Eye ωx (X-left)', 'Eye ωy (Y-up) [HORIZONTAL]', 'Eye ωz (Z-gaze) [TORSION]']
    eye_colors_left = [C_X, C_Y, C_Z]
    eye_colors_right = [C_X_R, C_Y_R, C_Z_R]

    # Plot grid
    for row, (left_vel, right_vel, eye_label, eye_color_l, eye_color_r) in enumerate(
            zip(eye_vels_left, eye_vels_right, eye_labels, eye_colors_left, eye_colors_right)):
        for col, (skull_vel, skull_label, skull_color) in enumerate(
                zip(skull_vels, skull_labels, skull_colors)):
            ax = axes[row, col]

            # Compute correlations
            corr_left, pval_left = compute_correlation(skull_vel, left_vel)
            corr_right, pval_right = compute_correlation(skull_vel, right_vel)

            # Scatter with regression
            sns.regplot(x=skull_vel, y=left_vel, ax=ax,
                        scatter_kws={'alpha': 0.2, 's': 5, 'color': eye_color_l},
                        line_kws={'color': eye_color_l, 'linewidth': 2},
                        label=f'L: r={corr_left:.2f}')
            sns.regplot(x=skull_vel, y=right_vel, ax=ax,
                        scatter_kws={'alpha': 0.2, 's': 5, 'color': eye_color_r, 'marker': 's'},
                        line_kws={'color': eye_color_r, 'linewidth': 2},
                        label=f'R: r={corr_right:.2f}')

            # Zero lines
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

            # Labels
            if row == 2:
                ax.set_xlabel(skull_label, fontsize=9)
            if col == 0:
                ax.set_ylabel(eye_label, fontsize=9)

            # Highlight strong VOR correlations (negative)
            avg_corr = (corr_left + corr_right) / 2
            if avg_corr < -0.3:
                ax.set_facecolor('#ffe6e6')  # Light red for negative correlation
            elif avg_corr > 0.3:
                ax.set_facecolor('#e6ffe6')  # Light green for positive correlation

            ax.legend(loc='upper left', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    fig.suptitle(
        'VOR Correlation Grid: Skull vs Eye Angular Velocity\n'
        '(Using RAW ω components - NO anatomical sign correction)\n'
        'Red background = negative correlation (expected for VOR)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


# =============================================================================
# FIGURE 3: CONJUGATE MOVEMENT CHECK
# =============================================================================

def plot_conjugate_check(
    left_eye,
    right_eye,
    output_path: Path | None = None,
) -> None:
    """
    Check if both eyes are moving together (conjugate movement).

    Shows BOTH uncorrected and corrected (with left eye sign flip) versions
    to demonstrate the coordinate frame issue.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    components = ['x', 'y', 'z']
    labels = ['ωx (elevation axis)', 'ωy (horizontal axis)', 'ωz (torsion axis)']
    colors = [C_X, C_Y, C_Z]

    # Row 0: UNCORRECTED (raw data)
    for idx, (comp, label, color) in enumerate(zip(components, labels, colors)):
        left_vel = np.rad2deg(left_eye.angular_velocity_global[:, idx])
        right_vel = np.rad2deg(right_eye.angular_velocity_global[:, idx])

        corr, pval = compute_correlation(left_vel, right_vel)

        ax = axes[0, idx]
        ax.scatter(left_vel, right_vel, alpha=0.3, s=5, color=color)

        # Add regression line
        mask = np.isfinite(left_vel) & np.isfinite(right_vel)
        if np.sum(mask) > 10:
            z = np.polyfit(left_vel[mask], right_vel[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.min(left_vel[mask]), np.max(left_vel[mask]), 100)
            ax.plot(x_line, p(x_line), 'k-', linewidth=2)

        # Identity line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Identity')

        ax.set_xlabel(f'Left Eye {label} (deg/s)')
        ax.set_ylabel(f'Right Eye {label} (deg/s)')
        ax.set_title(f'UNCORRECTED\n{label}\nr = {corr:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Row 1: CORRECTED (left eye sign flipped for ωy and ωz)
    sign_corrections = [LEFT_EYE_OMEGA_X_SIGN, LEFT_EYE_OMEGA_Y_SIGN, LEFT_EYE_OMEGA_Z_SIGN]

    for idx, (comp, label, color, sign) in enumerate(zip(components, labels, colors, sign_corrections)):
        left_vel = sign * np.rad2deg(left_eye.angular_velocity_global[:, idx])
        right_vel = np.rad2deg(right_eye.angular_velocity_global[:, idx])

        corr, pval = compute_correlation(left_vel, right_vel)

        ax = axes[1, idx]
        ax.scatter(left_vel, right_vel, alpha=0.3, s=5, color=color)

        # Add regression line
        mask = np.isfinite(left_vel) & np.isfinite(right_vel)
        if np.sum(mask) > 10:
            z = np.polyfit(left_vel[mask], right_vel[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.min(left_vel[mask]), np.max(left_vel[mask]), 100)
            ax.plot(x_line, p(x_line), 'k-', linewidth=2)

        # Identity line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Identity')

        correction_str = f'(L×{sign:+.0f})' if sign != 1.0 else '(no change)'
        ax.set_xlabel(f'Left Eye {label} {correction_str} (deg/s)')
        ax.set_ylabel(f'Right Eye {label} (deg/s)')

        status = "✓" if corr > 0.5 else "⚠" if corr > 0 else "✗"
        ax.set_title(f'CORRECTED\n{label}\nr = {corr:.3f} {status}', fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        'Conjugate Movement Check: Left vs Right Eye Angular Velocity\n'
        'Top row: Raw data (negative r = coordinate frame issue)\n'
        'Bottom row: With left eye sign correction (positive r = conjugate movement confirmed)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


# =============================================================================
# FIGURE 4: FULL CORRELATION MATRIX
# =============================================================================

def plot_full_correlation_matrix(
    skull,
    left_eye,
    right_eye,
    output_path: Path | None = None,
) -> None:
    """
    Plot complete correlation matrix between all skull and eye angular velocity components.
    """
    # Build data dictionary
    data: dict[str, NDArray[np.float64]] = {}

    # Skull - both global and local
    data['skull_global_roll'] = np.rad2deg(skull.angular_velocity_global[:, 0])
    data['skull_global_pitch'] = np.rad2deg(skull.angular_velocity_global[:, 1])
    data['skull_global_yaw'] = np.rad2deg(skull.angular_velocity_global[:, 2])
    data['skull_local_roll'] = np.rad2deg(skull.angular_velocity_local[:, 0])
    data['skull_local_pitch'] = np.rad2deg(skull.angular_velocity_local[:, 1])
    data['skull_local_yaw'] = np.rad2deg(skull.angular_velocity_local[:, 2])

    # Eyes - RAW components only
    for eye, prefix in [(left_eye, 'L'), (right_eye, 'R')]:
        data[f'{prefix}_global_x'] = np.rad2deg(eye.angular_velocity_global[:, 0])
        data[f'{prefix}_global_y'] = np.rad2deg(eye.angular_velocity_global[:, 1])
        data[f'{prefix}_global_z'] = np.rad2deg(eye.angular_velocity_global[:, 2])
        data[f'{prefix}_local_x'] = np.rad2deg(eye.angular_velocity_local[:, 0])
        data[f'{prefix}_local_y'] = np.rad2deg(eye.angular_velocity_local[:, 1])
        data[f'{prefix}_local_z'] = np.rad2deg(eye.angular_velocity_local[:, 2])

    # Skull components (rows)
    skull_keys = [k for k in data.keys() if k.startswith('skull')]
    # Eye components (columns)
    eye_keys = [k for k in data.keys() if k.startswith('L') or k.startswith('R')]

    n_skull = len(skull_keys)
    n_eye = len(eye_keys)

    # Compute correlation matrix
    corr_matrix = np.zeros((n_skull, n_eye))
    for i, sk in enumerate(skull_keys):
        for j, ek in enumerate(eye_keys):
            corr, _ = compute_correlation(data[sk], data[ek])
            corr_matrix[i, j] = corr

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', norm=norm, aspect='auto')

    cbar = plt.colorbar(im, ax=ax, label='Pearson Correlation')

    ax.set_xticks(np.arange(n_eye))
    ax.set_yticks(np.arange(n_skull))
    ax.set_xticklabels([k.replace('_', '\n') for k in eye_keys], fontsize=8)
    ax.set_yticklabels([k.replace('_', '\n') for k in skull_keys], fontsize=8)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add correlation values
    for i in range(n_skull):
        for j in range(n_eye):
            corr = corr_matrix[i, j]
            color = 'white' if abs(corr) > 0.5 else 'black'
            ax.text(j, i, f'{corr:.2f}', ha='center', va='center', color=color, fontsize=7)

    # Add separators
    ax.axvline(x=5.5, color='black', linewidth=2)  # Between L and R
    ax.axhline(y=2.5, color='black', linewidth=2)  # Between global and local

    ax.set_xlabel('Eye Angular Velocity Component', fontsize=10)
    ax.set_ylabel('Skull Angular Velocity Component', fontsize=10)

    ax.set_title(
        'Full Correlation Matrix: Skull vs Eye Angular Velocity\n'
        '(VOR should show negative correlations in horizontal components)',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_correlation_summary(skull, left_eye, right_eye) -> None:
    """Print summary of strongest correlations with and without correction."""

    skull_data = {
        'skull_local_roll': np.rad2deg(skull.angular_velocity_local[:, 0]),
        'skull_local_pitch': np.rad2deg(skull.angular_velocity_local[:, 1]),
        'skull_local_yaw': np.rad2deg(skull.angular_velocity_local[:, 2]),
    }

    # Uncorrected conjugate check first
    print("\n" + "=" * 70)
    print("CONJUGATE MOVEMENT CHECK (Left vs Right eye)")
    print("=" * 70)
    print("UNCORRECTED (raw data):")
    for comp, label in [(0, 'ωx'), (1, 'ωy (horizontal)'), (2, 'ωz (torsion)')]:
        left_vals = np.rad2deg(left_eye.angular_velocity_global[:, comp])
        right_vals = np.rad2deg(right_eye.angular_velocity_global[:, comp])
        corr, _ = compute_correlation(left_vals, right_vals)
        status = "✓" if corr > 0.5 else "⚠" if corr > 0 else "✗ SIGN FLIP!"
        print(f"  {label:20s}: r = {corr:+.3f}  {status}")

    print("\nCORRECTED (left eye ωy and ωz signs flipped):")
    signs = [LEFT_EYE_OMEGA_X_SIGN, LEFT_EYE_OMEGA_Y_SIGN, LEFT_EYE_OMEGA_Z_SIGN]
    for comp, label, sign in [(0, 'ωx', signs[0]), (1, 'ωy (horizontal)', signs[1]), (2, 'ωz (torsion)', signs[2])]:
        left_vals = sign * np.rad2deg(left_eye.angular_velocity_global[:, comp])
        right_vals = np.rad2deg(right_eye.angular_velocity_global[:, comp])
        corr, _ = compute_correlation(left_vals, right_vals)
        status = "✓ CONJUGATE" if corr > 0.5 else "⚠" if corr > 0 else "✗"
        correction = f"(L×{sign:+.0f})" if sign != 1 else ""
        print(f"  {label:20s}: r = {corr:+.3f}  {status} {correction}")

    # VOR correlations with corrected values
    print("\n" + "=" * 70)
    print("VOR CORRELATION SUMMARY (with left eye correction)")
    print("=" * 70)

    pairs: list[tuple[float, str, str]] = []
    eye_data_corrected = {
        'L_ωx': LEFT_EYE_OMEGA_X_SIGN * np.rad2deg(left_eye.angular_velocity_global[:, 0]),
        'L_ωy': LEFT_EYE_OMEGA_Y_SIGN * np.rad2deg(left_eye.angular_velocity_global[:, 1]),
        'L_ωz': LEFT_EYE_OMEGA_Z_SIGN * np.rad2deg(left_eye.angular_velocity_global[:, 2]),
        'R_ωx': np.rad2deg(right_eye.angular_velocity_global[:, 0]),
        'R_ωy': np.rad2deg(right_eye.angular_velocity_global[:, 1]),
        'R_ωz': np.rad2deg(right_eye.angular_velocity_global[:, 2]),
    }

    for skull_name, skull_vals in skull_data.items():
        for eye_name, eye_vals in eye_data_corrected.items():
            corr, _ = compute_correlation(skull_vals, eye_vals)
            pairs.append((corr, skull_name, eye_name))

    pairs.sort(key=lambda x: x[0])

    print("Most negative correlations (expected for VOR):")
    print("-" * 70)
    for corr, skull_name, eye_name in pairs[:6]:
        vor_marker = "← VOR?" if corr < -0.3 else ""
        print(f"  {skull_name:20s} vs {eye_name:10s}: r = {corr:+.3f}  {vor_marker}")

    print("-" * 70)
    print("Most positive correlations:")
    for corr, skull_name, eye_name in pairs[-3:]:
        print(f"  {skull_name:20s} vs {eye_name:10s}: r = {corr:+.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  - VOR: Head yaw → negative correlation with eye ωy (horizontal)")
    print("  - VOR: Head pitch → negative correlation with eye ωx (elevation)")
    print("  - If correlations are weak, check skull/eye frame alignment")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def run_all_plots(
    analyzable_output_dir: Path,
    output_dir: Path | None = None,
) -> None:
    """Run all VOR analysis plots."""
    print(f"Loading data from: {analyzable_output_dir}")
    skull, left_eye, right_eye = load_kinematics_data(analyzable_output_dir)

    print(f"Loaded skull: {skull.n_frames} frames, {skull.duration:.1f}s")
    print(f"Loaded left eye: {left_eye.n_frames} frames")
    print(f"Loaded right eye: {right_eye.n_frames} frames")

    # Print summary
    print_correlation_summary(skull, left_eye, right_eye)

    # Generate plots
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            'overview': output_dir / 'timeseries_overview.png',
            'vor_grid': output_dir / 'vor_correlation_grid.png',
            'conjugate': output_dir / 'conjugate_check.png',
            'full_matrix': output_dir / 'full_correlation_matrix.png',
        }
    else:
        paths = {'overview': None, 'vor_grid': None, 'conjugate': None, 'full_matrix': None}

    print("\nGenerating plots...")
    plot_timeseries_overview(skull, left_eye, right_eye, output_path=paths['overview'])
    plot_vor_correlation_grid(skull, left_eye, right_eye, output_path=paths['vor_grid'])
    plot_conjugate_check(left_eye, right_eye, output_path=paths['conjugate'])
    plot_full_correlation_matrix(skull, left_eye, right_eye, output_path=paths['full_matrix'])

    print("\nDone!")


def main() -> None:
    """Command-line entry point."""
    if len(sys.argv) >= 2:
        analyzable_output_dir = Path(sys.argv[1])
    else:
        analyzable_output_dir = DEFAULT_ANALYZABLE_OUTPUT_DIR
        print(f"Using default directory: {analyzable_output_dir}")

    if not analyzable_output_dir.exists():
        print(f"Error: Directory not found: {analyzable_output_dir}")
        print("\nUsage: python plot_ferret_kinematics_vor.py [analyzable_output_dir] [--save output_dir]")
        sys.exit(1)

    output_dir: Path | None = None
    if '--save' in sys.argv:
        save_idx = sys.argv.index('--save')
        if save_idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[save_idx + 1])
        else:
            output_dir = analyzable_output_dir / 'vor_analysis_plots'

    run_all_plots(analyzable_output_dir, output_dir)


if __name__ == "__main__":
    main()