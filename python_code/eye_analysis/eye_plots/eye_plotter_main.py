"""Main analyzer orchestrating all eye tracking analysis components - simplified."""

import cv2
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from python_code.eye_analysis.eye_plots.plot_dashboard import plot_integrated_dashboard
from python_code.eye_analysis.eye_plots.plot_timeseries import plot_pupil_timeseries
from python_code.eye_analysis.eye_plots.plot_heatmap import plot_2d_trajectory_heatmap
from python_code.eye_analysis.eye_plots.plot_surface_3d import plot_3d_gaze_surface
from python_code.eye_analysis.eye_plots.plot_histogram import plot_position_histograms
from python_code.eye_analysis.data_models.eye_video_dataset import EyeVideoData


class EyeDataPlotter:
    """Analysis and visualization tools for eye tracking data."""

    def __init__(self, *, dataset: EyeVideoData) -> None:
        """Initialize analyzer with eye tracking dataset.

        Args:
            dataset: Eye tracking dataset to analyze
        """
        self.dataset = dataset

    def get_pupil_center_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get pupil center trajectories (raw and cleaned).

        Returns:
            Tuple of (raw_centers, cleaned_centers) as (n_frames, 2) arrays
        """
        pupil_pairs = [self.dataset.dataset.pairs[f'p{i}'] for i in range(1, 9)]

        raw_x = np.mean([pair.raw.x for pair in pupil_pairs], axis=0)
        raw_y = np.mean([pair.raw.y for pair in pupil_pairs], axis=0)
        raw_centers = np.column_stack([raw_x, raw_y])

        cleaned_x = np.mean([pair.cleaned.x for pair in pupil_pairs], axis=0)
        cleaned_y = np.mean([pair.cleaned.y for pair in pupil_pairs], axis=0)
        cleaned_centers = np.column_stack([cleaned_x, cleaned_y])

        return raw_centers, cleaned_centers

    def plot_integrated_dashboard(
        self,
        *,
        nbins: int = 50,
        show: bool = True,
        output_path: Path | None = None,
        frame_index: int = 0
    ) -> go.Figure:
        """Create integrated dashboard with all analysis views.

        Args:
            nbins: Number of bins for histograms
            show: Whether to display the plot
            output_path: Optional path to save HTML figure
            frame_index: Frame index to extract from video for display

        Returns:
            Plotly figure object with all analysis views
        """
        # Extract frame from video
        video_frame = None
        if self.dataset.videos.video_capture is not None:
            self.dataset.videos.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.dataset.videos.video_capture.read()
            if ret:
                video_frame = frame

        return plot_integrated_dashboard(
            dataset=self.dataset.dataset,
            data_name=self.dataset.data_name,
            nbins=nbins,
            show=show,
            output_path=output_path,
            video_frame=video_frame
        )

    def plot_timeseries(
        self,
        *,
        show: bool = True,
        output_path: Path | None = None
    ) -> go.Figure:
        """Create timeseries plots of pupil position.

        Args:
            show: Whether to display the plot
            output_path: Optional path to save HTML figure

        Returns:
            Plotly figure object
        """
        return plot_pupil_timeseries(
            dataset=self.dataset.dataset,
            frames=self.dataset.dataset.frame_indices,
            data_name=self.dataset.data_name,
            show=show,
            output_path=output_path
        )

    def plot_heatmap(
        self,
        *,
        nbins: int = 50,
        colorscale: str = 'Hot',
        show: bool = True,
        output_path: Path | None = None
    ) -> go.Figure:
        """Create 2D heatmap of pupil position distribution.

        Args:
            nbins: Number of bins for histogram
            colorscale: Plotly colorscale name
            show: Whether to display the plot
            output_path: Optional path to save HTML figure

        Returns:
            Plotly figure object
        """
        # Get average pupil positions (raw)
        pupil_pairs = [self.dataset.dataset.pairs[f'p{i}'] for i in range(1, 9)]
        x_positions = np.mean([pair.raw.x for pair in pupil_pairs], axis=0)
        y_positions = np.mean([pair.raw.y for pair in pupil_pairs], axis=0)

        return plot_2d_trajectory_heatmap(
            x_positions=x_positions,
            y_positions=y_positions,
            data_name=self.dataset.data_name,
            nbins=nbins,
            colorscale=colorscale,
            show=show,
            output_path=output_path
        )

    def plot_3d_surface(
        self,
        *,
        nbins: int = 50,
        colorscale: str = 'Hot',
        show: bool = True,
        output_path: Path | None = None
    ) -> go.Figure:
        """Create 3D surface plot of pupil position probability distribution.

        Args:
            nbins: Number of bins for histogram
            colorscale: Plotly colorscale name
            show: Whether to display the plot
            output_path: Optional path to save HTML figure

        Returns:
            Plotly figure object
        """
        # Get average pupil positions (raw)
        pupil_pairs = [self.dataset.dataset.pairs[f'p{i}'] for i in range(1, 9)]
        x_positions = np.mean([pair.raw.x for pair in pupil_pairs], axis=0)
        y_positions = np.mean([pair.raw.y for pair in pupil_pairs], axis=0)

        return plot_3d_gaze_surface(
            x_positions=x_positions,
            y_positions=y_positions,
            data_name=self.dataset.data_name,
            nbins=nbins,
            colorscale=colorscale,
            show=show,
            output_path=output_path
        )

    def plot_histograms(
        self,
        *,
        nbins: int = 50,
        show: bool = True,
        output_path: Path | None = None
    ) -> go.Figure:
        """Create histograms of X and Y position distributions.

        Args:
            nbins: Number of histogram bins
            show: Whether to display the plot
            output_path: Optional path to save HTML figure

        Returns:
            Plotly figure object
        """
        # Get average pupil positions (raw)
        pupil_pairs = [self.dataset.dataset.pairs[f'p{i}'] for i in range(1, 9)]
        x_positions = np.mean([pair.raw.x for pair in pupil_pairs], axis=0)
        y_positions = np.mean([pair.raw.y for pair in pupil_pairs], axis=0)

        return plot_position_histograms(
            x_positions=x_positions,
            y_positions=y_positions,
            data_name=self.dataset.data_name,
            nbins=nbins,
            show=show,
            output_path=output_path
        )

    def create_analysis_report(
        self,
        *,
        output_dir: Path,
        nbins: int = 50,
        frame_index: int = 0
    ) -> None:
        """Generate complete analysis report with all visualizations.

        Args:
            output_dir: Directory to save all plots
            nbins: Number of bins for histograms
            frame_index: Frame index to extract from video for dashboard
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating analysis report for {self.dataset.data_name}...")

        # Integrated dashboard
        print("  Creating integrated dashboard...")
        self.plot_integrated_dashboard(
            nbins=nbins,
            show=False,
            output_path=output_dir / "dashboard.html",
            frame_index=frame_index
        )

        # Timeseries
        print("  Creating timeseries plots...")
        self.plot_timeseries(
            show=False,
            output_path=output_dir / "timeseries.html"
        )

        # Heatmap
        print("  Creating 2D heatmap...")
        self.plot_heatmap(
            nbins=nbins,
            show=False,
            output_path=output_dir / "heatmap.html"
        )

        # 3D Surface
        print("  Creating 3D surface plot...")
        self.plot_3d_surface(
            nbins=nbins,
            show=False,
            output_path=output_dir / "surface_3d.html"
        )

        # Histograms
        print("  Creating histograms...")
        self.plot_histograms(
            nbins=nbins,
            show=False,
            output_path=output_dir / "histograms.html"
        )

        print(f"Analysis report saved to: {output_dir}")


if __name__ == "__main__":
    # Setup paths
    _base_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37"
    )
    video_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4"
    )
    timestamps_npy_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy"
    )
    csv_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )

    # Create dataset - raw and cleaned data are automatically loaded and filtered
    print("Loading eye tracking dataset...")
    eye_dataset = EyeVideoData.create(
        data_name="ferret_757_eye_tracking",
        recording_path=_base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
        butterworth_cutoff=6.0,  # Hz - applied to create cleaned data
    )

    # Create analyzer
    print("Initializing analyzer...")
    analyzer = EyeDataPlotter(dataset=eye_dataset)

    # Show integrated dashboard with video frame
    # You can specify which frame to display (default is frame 0)
    print("\nDisplaying integrated dashboard with video frame...")
    analyzer.plot_integrated_dashboard(
        nbins=100,
        show=True,
        frame_index=50  # Show frame 50 from the video
    )

    # # Generate complete analysis report with frame
    # print("\nGenerating complete analysis report...")
    # output_dir = _base_path / "analysis_output"
    #
    # analyzer.create_analysis_report(
    #     output_dir=output_dir,
    #     nbins=50,
    #     frame_index=50  # Include frame 50 in the dashboard
    # )

