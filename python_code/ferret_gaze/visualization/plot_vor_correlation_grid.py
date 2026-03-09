"""
VOR Correlation Grid Analysis

Creates a comprehensive grid showing correlations between ALL combinations of
head and eye angular velocity components to identify which axes are aligned
and show proper VOR (vestibulo-ocular reflex) anticorrelation.

This helps diagnose axis alignment issues between different coordinate frames.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import TwoSlopeNorm
from numpy.typing import NDArray
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

SKULL_KINEMATICS_SUBDIR = "skull_kinematics"
SKULL_KINEMATICS_CSV = "skull_kinematics.csv"

LEFT_EYE_KINEMATICS_SUBDIR = "left_eye_kinematics"
LEFT_EYE_KINEMATICS_CSV = "left_eye_kinematics.csv"

RIGHT_EYE_KINEMATICS_SUBDIR = "right_eye_kinematics"
RIGHT_EYE_KINEMATICS_CSV = "right_eye_kinematics.csv"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AngularVelocityData:
    """Container for angular velocity data with all components."""

    def __init__(
        self,
        name: str,
        timestamps: NDArray[np.float64],
        global_components: dict[str, NDArray[np.float64]],
        local_components: dict[str, NDArray[np.float64]],
    ) -> None:
        self.name = name
        self.timestamps = timestamps
        self.global_components = global_components
        self.local_components = local_components

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_skull_angular_velocity(kinematics_dir: Path) -> AngularVelocityData:
    """Load skull angular velocity with all components (global and local)."""
    kinematics_csv = kinematics_dir / SKULL_KINEMATICS_CSV
    if not kinematics_csv.exists():
        raise FileNotFoundError(f"Skull kinematics CSV not found: {kinematics_csv}")

    df = pl.read_csv(kinematics_csv)

    # Extract timestamps
    timestamps = (
        df.select(["frame", "timestamp_s"])
        .unique()
        .sort("frame")
        ["timestamp_s"]
        .to_numpy()
        .astype(np.float64)
    )

    # Skull uses component names: roll, pitch, yaw (indices 0, 1, 2)
    skull_component_names = ["roll", "pitch", "yaw"]

    global_components = _extract_angular_velocity_components(
        df=df,
        trajectory_name="angular_velocity_global",
        component_names=skull_component_names,
    )

    local_components = _extract_angular_velocity_components(
        df=df,
        trajectory_name="angular_velocity_local",
        component_names=skull_component_names,
    )

    return AngularVelocityData(
        name="skull",
        timestamps=timestamps,
        global_components=global_components,
        local_components=local_components,
    )


def load_eye_angular_velocity(kinematics_dir: Path, eye_name: str) -> AngularVelocityData:
    """Load eye angular velocity with all components (global and local)."""
    if eye_name == "left_eye":
        kinematics_csv_name = LEFT_EYE_KINEMATICS_CSV
    elif eye_name == "right_eye":
        kinematics_csv_name = RIGHT_EYE_KINEMATICS_CSV
    else:
        raise ValueError(f"eye_name must be 'left_eye' or 'right_eye', got '{eye_name}'")

    kinematics_csv = kinematics_dir / kinematics_csv_name
    if not kinematics_csv.exists():
        raise FileNotFoundError(f"Eye kinematics CSV not found: {kinematics_csv}")

    df = pl.read_csv(kinematics_csv)

    # Extract timestamps
    timestamps = (
        df.select(["frame", "timestamp_s"])
        .unique()
        .sort("frame")
        ["timestamp_s"]
        .to_numpy()
        .astype(np.float64)
    )

    # Eye uses component names: x, y, z (indices 0, 1, 2)
    eye_component_names = ["x", "y", "z"]

    global_components = _extract_angular_velocity_components(
        df=df,
        trajectory_name="angular_velocity_global",
        component_names=eye_component_names,
    )

    local_components = _extract_angular_velocity_components(
        df=df,
        trajectory_name="angular_velocity_local",
        component_names=eye_component_names,
    )

    return AngularVelocityData(
        name=eye_name,
        timestamps=timestamps,
        global_components=global_components,
        local_components=local_components,
    )


def _extract_angular_velocity_components(
    df: pl.DataFrame,
    trajectory_name: str,
    component_names: list[str],
) -> dict[str, NDArray[np.float64]]:
    """Extract angular velocity components from tidy dataframe."""
    trajectory_df = df.filter(pl.col("trajectory") == trajectory_name)

    components: dict[str, NDArray[np.float64]] = {}
    for component_name in component_names:
        component_df = (
            trajectory_df
            .filter(pl.col("component") == component_name)
            .sort("frame")
        )
        values_rad_s = component_df["value"].to_numpy().astype(np.float64)
        # Convert to degrees for easier interpretation
        components[component_name] = np.rad2deg(values_rad_s)

    return components


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Compute Pearson correlation coefficient and p-value."""
    # Handle NaN/inf values
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 10:
        return 0.0, 1.0

    result = stats.pearsonr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_vor_correlation_grid(
    skull: AngularVelocityData,
    left_eye: AngularVelocityData,
    right_eye: AngularVelocityData,
    output_path: Path | None = None,
) -> None:
    """
    Create a comprehensive grid showing correlations between all head/eye
    angular velocity component combinations.
    """
    # Define head components (rows)
    head_components: list[tuple[str, str, NDArray[np.float64]]] = []
    for frame_type, components in [("global", skull.global_components), ("local", skull.local_components)]:
        for comp_name, values in components.items():
            head_components.append((f"skull_{frame_type}_{comp_name}", f"Head {frame_type}\n{comp_name}", values))

    # Define eye components (columns) - both eyes
    eye_components: list[tuple[str, str, NDArray[np.float64]]] = []

    for eye_data, eye_label in [(left_eye, "L"), (right_eye, "R")]:
        for frame_type, components in [("global", eye_data.global_components), ("local", eye_data.local_components)]:
            for comp_name, values in components.items():
                eye_components.append((
                    f"{eye_data.name}_{frame_type}_{comp_name}",
                    f"{eye_label} eye {frame_type}\n{comp_name}",
                    values
                ))

    n_head = len(head_components)
    n_eye = len(eye_components)

    # Compute correlation matrix
    corr_matrix = np.zeros((n_head, n_eye))
    pval_matrix = np.zeros((n_head, n_eye))

    for i, (_, _, head_vals) in enumerate(head_components):
        for j, (_, _, eye_vals) in enumerate(eye_components):
            corr, pval = compute_correlation(head_vals, eye_vals)
            corr_matrix[i, j] = corr
            pval_matrix[i, j] = pval

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Use diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Pearson Correlation")

    # Set tick labels
    head_labels = [label for _, label, _ in head_components]
    eye_labels = [label for _, label, _ in eye_components]

    ax.set_xticks(np.arange(n_eye))
    ax.set_yticks(np.arange(n_head))
    ax.set_xticklabels(eye_labels, fontsize=8)
    ax.set_yticklabels(head_labels, fontsize=8)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values as text
    for i in range(n_head):
        for j in range(n_eye):
            corr = corr_matrix[i, j]
            pval = pval_matrix[i, j]

            # Choose text color based on background
            text_color = "white" if abs(corr) > 0.5 else "black"

            # Add significance marker
            sig_marker = ""
            if pval < 0.001:
                sig_marker = "***"
            elif pval < 0.01:
                sig_marker = "**"
            elif pval < 0.05:
                sig_marker = "*"

            ax.text(j, i, f"{corr:.2f}{sig_marker}",
                    ha="center", va="center", color=text_color, fontsize=7)

    # Add grid lines to separate groups
    # Vertical lines between eyes
    ax.axvline(x=5.5, color="black", linewidth=2)
    ax.axvline(x=11.5, color="black", linewidth=2)

    # Horizontal line between global and local
    ax.axhline(y=2.5, color="black", linewidth=2)

    ax.set_title(
        "VOR Correlation Analysis: Head vs Eye Angular Velocity Components\n"
        "(* p<0.05, ** p<0.01, *** p<0.001)\n"
        "VOR should show strong NEGATIVE correlation between head and compensatory eye movement",
        fontsize=12
    )

    ax.set_xlabel("Eye Angular Velocity Component (deg/s)", fontsize=10)
    ax.set_ylabel("Head Angular Velocity Component (deg/s)", fontsize=10)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved correlation grid to {output_path}")
    else:
        plt.show()


def plot_top_correlations_scatter(
    skull: AngularVelocityData,
    left_eye: AngularVelocityData,
    right_eye: AngularVelocityData,
    n_top: int = 6,
    output_path: Path | None = None,
) -> None:
    """
    Create scatter plots for the top N strongest (most negative) correlations.
    These are the most likely VOR-related pairs.
    """
    # Collect all pairs with their correlations
    pairs: list[tuple[float, str, str, NDArray[np.float64], NDArray[np.float64]]] = []

    head_data = [
        ("skull_global_roll", skull.global_components["roll"]),
        ("skull_global_pitch", skull.global_components["pitch"]),
        ("skull_global_yaw", skull.global_components["yaw"]),
        ("skull_local_roll", skull.local_components["roll"]),
        ("skull_local_pitch", skull.local_components["pitch"]),
        ("skull_local_yaw", skull.local_components["yaw"]),
    ]

    eye_data = [
        ("L_global_x", left_eye.global_components["x"]),
        ("L_global_y", left_eye.global_components["y"]),
        ("L_global_z", left_eye.global_components["z"]),
        ("L_local_x", left_eye.local_components["x"]),
        ("L_local_y", left_eye.local_components["y"]),
        ("L_local_z", left_eye.local_components["z"]),
        ("R_global_x", right_eye.global_components["x"]),
        ("R_global_y", right_eye.global_components["y"]),
        ("R_global_z", right_eye.global_components["z"]),
        ("R_local_x", right_eye.local_components["x"]),
        ("R_local_y", right_eye.local_components["y"]),
        ("R_local_z", right_eye.local_components["z"]),
    ]

    for head_name, head_vals in head_data:
        for eye_name, eye_vals in eye_data:
            corr, _ = compute_correlation(head_vals, eye_vals)
            pairs.append((corr, head_name, eye_name, head_vals, eye_vals))

    # Sort by correlation (most negative first for VOR)
    pairs.sort(key=lambda x: x[0])

    # Take top N most negative and top N most positive
    top_negative = pairs[:n_top]
    top_positive = pairs[-n_top:][::-1]

    # Create figure with subplots
    fig, axes = plt.subplots(2, n_top, figsize=(4 * n_top, 8))

    # Plot most negative correlations (top row)
    for idx, (corr, head_name, eye_name, head_vals, eye_vals) in enumerate(top_negative):
        ax = axes[0, idx]
        ax.scatter(head_vals, eye_vals, alpha=0.3, s=5)

        # Add regression line
        mask = np.isfinite(head_vals) & np.isfinite(eye_vals)
        if np.sum(mask) > 10:
            z = np.polyfit(head_vals[mask], eye_vals[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.min(head_vals[mask]), np.max(head_vals[mask]), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=2)

        ax.set_xlabel(f"{head_name} (deg/s)", fontsize=8)
        ax.set_ylabel(f"{eye_name} (deg/s)", fontsize=8)
        ax.set_title(f"r = {corr:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Plot most positive correlations (bottom row)
    for idx, (corr, head_name, eye_name, head_vals, eye_vals) in enumerate(top_positive):
        ax = axes[1, idx]
        ax.scatter(head_vals, eye_vals, alpha=0.3, s=5, color="orange")

        # Add regression line
        mask = np.isfinite(head_vals) & np.isfinite(eye_vals)
        if np.sum(mask) > 10:
            z = np.polyfit(head_vals[mask], eye_vals[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.min(head_vals[mask]), np.max(head_vals[mask]), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=2)

        ax.set_xlabel(f"{head_name} (deg/s)", fontsize=8)
        ax.set_ylabel(f"{eye_name} (deg/s)", fontsize=8)
        ax.set_title(f"r = {corr:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel(f"Most NEGATIVE correlations\n{axes[0, 0].get_ylabel()}")
    axes[1, 0].set_ylabel(f"Most POSITIVE correlations\n{axes[1, 0].get_ylabel()}")

    fig.suptitle(
        "Top Correlations: Head vs Eye Angular Velocity\n"
        "(Top row: Most negative - potential VOR | Bottom row: Most positive)",
        fontsize=14
    )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved scatter plots to {output_path}")
    else:
        plt.show()


def print_correlation_summary(
    skull: AngularVelocityData,
    left_eye: AngularVelocityData,
    right_eye: AngularVelocityData,
) -> None:
    """Print a summary of the strongest correlations."""
    pairs: list[tuple[float, float, str, str]] = []

    head_data = [
        ("skull_global_roll", skull.global_components["roll"]),
        ("skull_global_pitch", skull.global_components["pitch"]),
        ("skull_global_yaw", skull.global_components["yaw"]),
        ("skull_local_roll", skull.local_components["roll"]),
        ("skull_local_pitch", skull.local_components["pitch"]),
        ("skull_local_yaw", skull.local_components["yaw"]),
    ]

    # Raw eye components (no anatomical sign correction - for conjugate movement)
    eye_data = [
        ("L_global_x", left_eye.global_components["x"]),
        ("L_global_y", left_eye.global_components["y"]),
        ("L_global_z", left_eye.global_components["z"]),
        ("L_local_x", left_eye.local_components["x"]),
        ("L_local_y", left_eye.local_components["y"]),
        ("L_local_z", left_eye.local_components["z"]),
        ("R_global_x", right_eye.global_components["x"]),
        ("R_global_y", right_eye.global_components["y"]),
        ("R_global_z", right_eye.global_components["z"]),
        ("R_local_x", right_eye.local_components["x"]),
        ("R_local_y", right_eye.local_components["y"]),
        ("R_local_z", right_eye.local_components["z"]),
    ]

    for head_name, head_vals in head_data:
        for eye_name, eye_vals in eye_data:
            corr, pval = compute_correlation(head_vals, eye_vals)
            pairs.append((corr, pval, head_name, eye_name))

    # Sort by absolute correlation
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)

    print("\n" + "=" * 80)
    print("CORRELATION SUMMARY (sorted by |r|)")
    print("=" * 80)
    print(f"{'Head Component':<25} {'Eye Component':<20} {'r':>8} {'p-value':>12}")
    print("-" * 80)

    for corr, pval, head_name, eye_name in pairs[:20]:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"{head_name:<25} {eye_name:<20} {corr:>8.3f} {pval:>10.2e} {sig}")

    # Also check if left and right eyes are correlated (conjugate movement check)
    print("\n" + "=" * 80)
    print("CONJUGATE MOVEMENT CHECK (L vs R eye same component)")
    print("=" * 80)
    for comp in ["x", "y", "z"]:
        for frame in ["global", "local"]:
            l_vals = left_eye.global_components[comp] if frame == "global" else left_eye.local_components[comp]
            r_vals = right_eye.global_components[comp] if frame == "global" else right_eye.local_components[comp]
            corr, pval = compute_correlation(l_vals, r_vals)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"L_{frame}_{comp} vs R_{frame}_{comp}: r = {corr:>8.3f} {sig}")
    
    print("\n(High positive correlation = conjugate movement)")
    print("(If VOR is working, both eyes should move together)")

    print("\n" + "=" * 80)
    print("INTERPRETATION NOTES:")
    print("=" * 80)
    print("- VOR should produce NEGATIVE correlations between head and eye angular velocity")
    print("- Strong negative correlation indicates compensatory eye movement")
    print("- The 'correct' pair depends on coordinate frame alignment:")
    print("  * If world frame is Y-up: head pitch ↔ eye X, head yaw ↔ eye Y")
    print("  * If world frame is Z-up: head roll ↔ eye X, head pitch ↔ eye Y, head yaw ↔ eye Z")
    print("- Eye frame: +Z=gaze, +Y=up, +X=subject's left")
    print("- Skull frame: +X=forward(nose), +Y=left, +Z=computed (likely down)")
    print("\nIMPORTANT: This analysis uses RAW eye ωy values without anatomical sign")
    print("correction. Both eyes should show similar correlation patterns.")


def run_analysis(
    analyzable_output_dir: Path,
    output_dir: Path | None = None,
) -> None:
    """Run the complete VOR correlation analysis."""
    skull_dir = analyzable_output_dir / SKULL_KINEMATICS_SUBDIR
    left_eye_dir = analyzable_output_dir / LEFT_EYE_KINEMATICS_SUBDIR
    right_eye_dir = analyzable_output_dir / RIGHT_EYE_KINEMATICS_SUBDIR

    for dir_path, name in [
        (skull_dir, SKULL_KINEMATICS_SUBDIR),
        (left_eye_dir, LEFT_EYE_KINEMATICS_SUBDIR),
        (right_eye_dir, RIGHT_EYE_KINEMATICS_SUBDIR),
    ]:
        if not dir_path.exists():
            raise FileNotFoundError(
                f"{name} directory not found: {dir_path}\n"
                f"Make sure you've run the gaze pipeline first."
            )

    print("Loading skull kinematics...")
    skull = load_skull_angular_velocity(skull_dir)
    print(f"  Loaded {skull.n_frames} frames")

    print("Loading left eye kinematics...")
    left_eye = load_eye_angular_velocity(left_eye_dir, "left_eye")
    print(f"  Loaded {left_eye.n_frames} frames")

    print("Loading right eye kinematics...")
    right_eye = load_eye_angular_velocity(right_eye_dir, "right_eye")
    print(f"  Loaded {right_eye.n_frames} frames")

    # Verify timestamps match
    if not np.allclose(skull.timestamps, left_eye.timestamps, rtol=1e-9):
        raise ValueError("Skull and left eye timestamps do not match!")
    if not np.allclose(skull.timestamps, right_eye.timestamps, rtol=1e-9):
        raise ValueError("Skull and right eye timestamps do not match!")

    # Print summary
    print_correlation_summary(skull, left_eye, right_eye)

    # Determine output paths
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = output_dir / "vor_correlation_grid.png"
        scatter_path = output_dir / "vor_top_correlations.png"
    else:
        grid_path = None
        scatter_path = None

    # Create plots
    print("\nCreating correlation grid...")
    plot_vor_correlation_grid(skull, left_eye, right_eye, output_path=grid_path)

    print("\nCreating scatter plots...")
    plot_top_correlations_scatter(skull, left_eye, right_eye, output_path=scatter_path)


if __name__ == "__main__":
    """Main entry point."""

    recording = Path("")

    if len(sys.argv) >= 2:
        recording_folder = Path(sys.argv[1])
    else:
        recording_folder = recording
        print(f"Using default directory: {recording_folder}")

    if not recording_folder.exists():
        print(f"Error: Directory does not exist: {recording_folder}")
        print("\nUsage: python plot_vor_correlation_grid.py [recording_folder] [output_dir]")
        sys.exit(1)

    output_dir: Path | None = None
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])

    analyzable_output_dir = recording_folder / "analyzable_output"

    run_analysis(analyzable_output_dir, output_dir)
