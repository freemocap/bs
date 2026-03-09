"""Visualization utilities for rigid body solver results."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_origin_trajectories(df: pd.DataFrame) -> None:
    """Plot original vs optimized local_origin trajectory as stacked x/y/z timeseries."""

    keypoints_to_plot = ["local_origin", "nose"]

    styles = {
        "original": {"color": "C0", "linestyle": "-", "keypoint": "o", "keypointfacecolor": "none", "keypointsize": 4},
        "optimized": {"color": "C1", "linestyle": "-", "keypoint": ".", "keypointsize": 6},
    }

    # Extract data for each keypoint/data_type combination
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for keypoint in keypoints_to_plot:
        data[keypoint] = {}
        for data_type in ["original", "optimized"]:
            subset = df[(df["keypoint"] == keypoint) & (df["data_type"] == data_type)].sort_values("frame")
            if len(subset) == 0:
                raise ValueError(f"No '{keypoint}' keypoint with data_type='{data_type}' found in DataFrame")
            data[keypoint][data_type] = subset

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)

    for idx, coord in enumerate(["x", "y", "z"]):
        ax = axes[idx]

        for keypoint in keypoints_to_plot:
            for data_type, style in styles.items():
                subset = data[keypoint][data_type]
                label = f"{keypoint} ({data_type})"
                ax.plot(
                    subset["timestamp"].values,
                    subset[coord].values,
                    label=label,
                    **style,
                )

        ax.set_ylabel(coord)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("timestamp")
    fig.suptitle("Local Origin Trajectory: Original vs Optimized")
    plt.tight_layout()
    plt.show()


def load_trajectory_csv(filepath: Path) -> pd.DataFrame:
    """Load a tidy trajectory CSV saved by the solver."""
    if not filepath.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = {"frame", "timestamp", "keypoint", "data_type", "x", "y", "z"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    logger.info(f"Loaded {len(df)} rows from {filepath.name}")
    return df


def run_visualization(trajectory_csv: Path) -> None:
    """Run all visualization plots from a saved trajectory CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logger.info(f"Loading trajectory data from: {trajectory_csv}")
    df = load_trajectory_csv(filepath=trajectory_csv)

    unique_keypoints = df["keypoint"].unique()
    logger.info(f"Found keypoints: {list(unique_keypoints)}")

    plot_origin_trajectories(df=df)


if __name__ == "__main__":
    # Example: point to your solver output
    output_dir = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\mocap_data\output_data\solver_output"
    )
    trajectory_csv = output_dir / "skull_and_spine_trajectory_data.csv"

    run_visualization(trajectory_csv=trajectory_csv)