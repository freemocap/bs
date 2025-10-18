"""Main analyzer orchestrating all eye tracking analysis components."""

import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from python_code.eye_data_cleanup.eye_analysis.plots.plot_dashboard import plot_integrated_dashboard
from python_code.eye_data_cleanup.eye_viewer import EyeVideoDataset


class EyeTrackingAnalyzer:
    """Analysis and visualization tools for eye tracking data."""

    def __init__(self, *, dataset: EyeVideoDataset) -> None:
        """Initialize analyzer with eye tracking dataset.

        Args:
            dataset: Eye tracking dataset to analyze
        """
        self.dataset: EyeVideoDataset = dataset
        self.pupil_centers: np.ndarray = dataset.get_pupil_centers()
        self.tear_ducts: np.ndarray = dataset.get_tear_duct_positions()
        self.eye_outers: np.ndarray = dataset.get_eye_outer_positions()
        self.frames: np.ndarray = dataset.pixel_trajectories.frame_indices


    def plot_integrated_dashboard(
        self,
        *,
        cutoff: float = 5.0,
        fs: float = 30.0,
        order: int = 4,
        nbins: int = 50,
        show: bool = True,
        output_path: Path | None = None
    ) -> go.Figure:
        """Create integrated dashboard with all analysis views.

        Args:
            cutoff: Butterworth filter cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Butterworth filter order
            nbins: Number of bins for histograms
            show: Whether to display the plot
            output_path: Optional path to save HTML figure

        Returns:
            Plotly figure object with all analysis views
        """
        return plot_integrated_dashboard(
            pupil_x_positions=self.pupil_centers[:, 0],
            pupil_y_positions=self.pupil_centers[:, 1],
            tear_duct_x_positions=self.tear_ducts[:, 0],
            tear_duct_y_positions=self.tear_ducts[:, 1],
            eye_outer_x_positions=self.eye_outers[:, 0],
            eye_outer_y_positions=self.eye_outers[:, 1],
            frames=self.frames,
            data_name=self.dataset.data_name,
            cutoff=cutoff,
            fs=fs,
            order=order,
            nbins=nbins,
            show=show,
            output_path=output_path
        )

    def create_analysis_report(
        self,
        *,
        output_dir: Path,
        cutoff: float = 5.0,
        fs: float = 30.0,
        order: int = 4,
        nbins: int = 50
    ) -> None:
        """Generate complete analysis report with all visualizations.

        Args:
            output_dir: Directory to save all plots
            cutoff: Butterworth filter cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Butterworth filter order
            nbins: Number of bins for histograms
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating analysis report for {self.dataset.data_name}...")

        # Integrated dashboard
        print("  Creating integrated dashboard...")
        self.plot_integrated_dashboard(
            cutoff=cutoff,
            fs=fs,
            order=order,
            nbins=nbins,
            show=False,
            output_path=output_dir / "dashboard.html"
        )


        print(f"Analysis report saved to: {output_dir}")
