"""Plotly-based analysis tools for eye tracking data.

Provides visualization and filtering tools for pupil position analysis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
from pathlib import Path

from eye_viewer import EyeVideoDataset


class EyeTrackingAnalyzer:
    """Analysis and visualization tools for eye tracking data."""

    def __init__(self, *, dataset: EyeVideoDataset) -> None:
        """Initialize analyzer with eye tracking dataset.

        Args:
            dataset: Eye tracking dataset to analyze
        """
        self.dataset: EyeVideoDataset = dataset
        self.pupil_centers: np.ndarray = dataset.get_pupil_centers()

    def apply_butterworth_filter(
        self,
        *,
        data: np.ndarray,
        cutoff: float = 5.0,
        fs: float = 30.0,
        order: int = 4
    ) -> np.ndarray:
        """Apply Butterworth lowpass filter to data.

        Args:
            data: Input data array (n_samples,)
            cutoff: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order

        Returns:
            Filtered data array
        """
        # Remove NaN values for filtering
        valid_mask: np.ndarray = ~np.isnan(data)
        if not np.any(valid_mask):
            return data

        # Design filter
        nyquist: float = 0.5 * fs
        normal_cutoff: float = cutoff / nyquist
        b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False)

        # Apply filter only to valid data
        filtered_data: np.ndarray = data.copy()
        if np.sum(valid_mask) > order * 3:  # Need enough points for filtering
            filtered_data[valid_mask] = filtfilt(b=b, a=a, x=data[valid_mask])

        return filtered_data

    def plot_pupil_timeseries(
        self,
        *,
        cutoff: float = 5.0,
        fs: float = 30.0,
        order: int = 4,
        show: bool = True,
        output_path: Path | None = None
    ) -> go.Figure:
        """Create timeseries plots of pupil center X and Y positions.

        Shows both raw and butterworth-filtered data.

        Args:
            cutoff: Butterworth filter cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Butterworth filter order
            show: Whether to display the plot
            output_path: Optional path to save HTML figure

        Returns:
            Plotly figure object
        """
        x_positions: np.ndarray = self.pupil_centers[:, 0]
        y_positions: np.ndarray = self.pupil_centers[:, 1]

        # Apply filters
        x_filtered: np.ndarray = self.apply_butterworth_filter(
            data=x_positions,
            cutoff=cutoff,
            fs=fs,
            order=order
        )
        y_filtered: np.ndarray = self.apply_butterworth_filter(
            data=y_positions,
            cutoff=cutoff,
            fs=fs,
            order=order
        )

        # Create frame indices
        frames: np.ndarray = self.dataset.pixel_trajectories.frame_indices

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Pupil X Position", "Pupil Y Position"),
            vertical_spacing=0.12
        )

        # X position plot
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=x_positions,
                mode='lines',
                name='X Raw',
                line=dict(color='lightblue', width=1),
                opacity=0.6
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=x_filtered,
                mode='lines',
                name='X Filtered',
                line=dict(color='blue', width=2)
            ),
            row=1,
            col=1
        )

        # Y position plot
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=y_positions,
                mode='lines',
                name='Y Raw',
                line=dict(color='lightcoral', width=1),
                opacity=0.6
            ),
            row=2,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=y_filtered,
                mode='lines',
                name='Y Filtered',
                line=dict(color='red', width=2)
            ),
            row=2,
            col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Frame", row=2, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_yaxes(title_text="X Position (pixels)", row=1, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_yaxes(title_text="Y Position (pixels)", row=2, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')

        fig.update_layout(
            title=dict(
                text=f"Pupil Center Timeseries - {self.dataset.data_name}",
                font=dict(color='lightgray')
            ),
            height=700,
            showlegend=True,
            hovermode='x unified',
            paper_bgcolor="rgb(20, 20, 20)",
            plot_bgcolor="rgb(30, 30, 30)",
            font=dict(color='lightgray'),
            legend=dict(
                font=dict(color='lightgray'),
                bgcolor="rgba(40, 40, 40, 0.8)",
                bordercolor="rgb(80, 80, 80)",
                borderwidth=1
            )
        )

        # Update subplot title colors
        for annotation in fig.layout.annotations:
            annotation.font.color = 'lightgray'

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(file=str(output_path))

        if show:
            fig.show()

        return fig

    def plot_2d_trajectory_heatmap(
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
        x_positions: np.ndarray = self.pupil_centers[:, 0]
        y_positions: np.ndarray = self.pupil_centers[:, 1]

        # Remove NaN values
        valid_mask: np.ndarray = ~(np.isnan(x_positions) | np.isnan(y_positions))
        x_valid: np.ndarray = x_positions[valid_mask]
        y_valid: np.ndarray = y_positions[valid_mask]

        # Create figure
        fig = go.Figure()

        # Add 2D histogram heatmap
        fig.add_trace(
            go.Histogram2d(
                x=x_valid,
                y=y_valid,
                nbinsx=nbins,
                nbinsy=nbins,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(
                        text="Count",
                        font=dict(color='lightgray')
                    ),
                    tickfont=dict(color='lightgray')
                ),
                hovertemplate='X: %{x}<br>Y: %{y}<br>Count: %{z}<extra></extra>'
            )
        )

        # Add trajectory trace
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='lines',
                line=dict(color='cyan', width=1),
                opacity=0.3,
                name='Trajectory',
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Pupil Position 2D Distribution - {self.dataset.data_name}",
                font=dict(color='lightgray')
            ),
            xaxis_title="X Position (pixels)",
            yaxis_title="Y Position (pixels)",
            width=800,
            height=800,
            yaxis=dict(scaleanchor="x", scaleratio=1, color='lightgray', gridcolor='rgb(60, 60, 60)'),
            paper_bgcolor="rgb(20, 20, 20)",
            plot_bgcolor="rgb(30, 30, 30)",
            font=dict(color='lightgray'),
            xaxis=dict(color='lightgray', gridcolor='rgb(60, 60, 60)'),
            legend=dict(
                font=dict(color='lightgray'),
                bgcolor="rgba(40, 40, 40, 0.8)",
                bordercolor="rgb(80, 80, 80)",
                borderwidth=1
            )
        )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(file=str(output_path))

        if show:
            fig.show()

        return fig

    def plot_position_histograms(
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
        x_positions: np.ndarray = self.pupil_centers[:, 0]
        y_positions: np.ndarray = self.pupil_centers[:, 1]

        # Remove NaN values
        x_valid: np.ndarray = x_positions[~np.isnan(x_positions)]
        y_valid: np.ndarray = y_positions[~np.isnan(y_positions)]

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("X Position Distribution", "Y Position Distribution")
        )

        # X histogram
        fig.add_trace(
            go.Histogram(
                x=x_valid,
                nbinsx=nbins,
                name='X Position',
                marker=dict(color='blue', line=dict(color='darkblue', width=1)),
                hovertemplate='Position: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1,
            col=1
        )

        # Y histogram
        fig.add_trace(
            go.Histogram(
                x=y_valid,
                nbinsx=nbins,
                name='Y Position',
                marker=dict(color='red', line=dict(color='darkred', width=1)),
                hovertemplate='Position: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1,
            col=2
        )

        # Add statistics annotations
        x_mean: float = float(np.mean(x_valid))
        x_std: float = float(np.std(x_valid))
        y_mean: float = float(np.mean(y_valid))
        y_std: float = float(np.std(y_valid))

        fig.add_annotation(
            text=f"Mean: {x_mean:.1f}<br>Std: {x_std:.1f}",
            xref="x1",
            yref="paper",
            x=x_mean,
            y=0.95,
            showarrow=False,
            bgcolor="rgba(100, 100, 255, 0.3)",
            bordercolor="blue",
            borderwidth=2,
            font=dict(color='lightgray'),
            row=1,
            col=1
        )

        fig.add_annotation(
            text=f"Mean: {y_mean:.1f}<br>Std: {y_std:.1f}",
            xref="x2",
            yref="paper",
            x=y_mean,
            y=0.95,
            showarrow=False,
            bgcolor="rgba(255, 100, 100, 0.3)",
            bordercolor="red",
            borderwidth=2,
            font=dict(color='lightgray'),
            row=1,
            col=2
        )

        fig.update_xaxes(title_text="X Position (pixels)", row=1, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_xaxes(title_text="Y Position (pixels)", row=1, col=2, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_yaxes(title_text="Count", row=1, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_yaxes(title_text="Count", row=1, col=2, color='lightgray', gridcolor='rgb(60, 60, 60)')

        fig.update_layout(
            title=dict(
                text=f"Position Distributions - {self.dataset.data_name}",
                font=dict(color='lightgray')
            ),
            height=500,
            width=1000,
            showlegend=False,
            paper_bgcolor="rgb(20, 20, 20)",
            plot_bgcolor="rgb(30, 30, 30)",
            font=dict(color='lightgray')
        )

        # Update subplot title colors
        for annotation in fig.layout.annotations[:2]:  # Only the first two are subplot titles
            annotation.font.color = 'lightgray'

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(file=str(output_path))

        if show:
            fig.show()

        return fig

    def plot_3d_gaze_surface(
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
        x_positions: np.ndarray = self.pupil_centers[:, 0]
        y_positions: np.ndarray = self.pupil_centers[:, 1]

        # Remove NaN values
        valid_mask: np.ndarray = ~(np.isnan(x_positions) | np.isnan(y_positions))
        x_valid: np.ndarray = x_positions[valid_mask]
        y_valid: np.ndarray = y_positions[valid_mask]

        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            x=x_valid,
            y=y_valid,
            bins=nbins
        )

        # Normalize to probabilities
        hist_normalized: np.ndarray = hist / np.sum(hist)

        # Get bin centers
        x_centers: np.ndarray = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers: np.ndarray = (y_edges[:-1] + y_edges[1:]) / 2

        # Create figure
        fig = go.Figure(
            data=[
                go.Surface(
                    x=x_centers,
                    y=y_centers,
                    z=hist_normalized.T,
                    colorscale=colorscale,
                    colorbar=dict(
                        title=dict(
                            text="Probability",
                            font=dict(color='lightgray')
                        ),
                        tickfont=dict(color='lightgray')
                    ),
                    hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Probability: %{z:.6f}<extra></extra>'
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text=f"3D Gaze Probability Surface - {self.dataset.data_name}",
                font=dict(color='lightgray')
            ),
            scene=dict(
                xaxis=dict(
                    title="X Position (pixels)",
                    backgroundcolor="rgb(30, 30, 30)",
                    gridcolor="rgb(60, 60, 60)",
                    titlefont=dict(color='lightgray'),
                    tickfont=dict(color='lightgray')
                ),
                yaxis=dict(
                    title="Y Position (pixels)",
                    backgroundcolor="rgb(30, 30, 30)",
                    gridcolor="rgb(60, 60, 60)",
                    titlefont=dict(color='lightgray'),
                    tickfont=dict(color='lightgray')
                ),
                zaxis=dict(
                    title="Probability",
                    backgroundcolor="rgb(30, 30, 30)",
                    gridcolor="rgb(60, 60, 60)",
                    titlefont=dict(color='lightgray'),
                    tickfont=dict(color='lightgray')
                ),
                bgcolor="rgb(20, 20, 20)"
            ),
            paper_bgcolor="rgb(20, 20, 20)",
            plot_bgcolor="rgb(20, 20, 20)",
            width=900,
            height=800
        )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(file=str(output_path))

        if show:
            fig.show()

        return fig

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
        x_positions: np.ndarray = self.pupil_centers[:, 0]
        y_positions: np.ndarray = self.pupil_centers[:, 1]

        # Apply filters
        x_filtered: np.ndarray = self.apply_butterworth_filter(
            data=x_positions,
            cutoff=cutoff,
            fs=fs,
            order=order
        )
        y_filtered: np.ndarray = self.apply_butterworth_filter(
            data=y_positions,
            cutoff=cutoff,
            fs=fs,
            order=order
        )

        # Remove NaN values for heatmaps
        valid_mask: np.ndarray = ~(np.isnan(x_positions) | np.isnan(y_positions))
        x_valid: np.ndarray = x_positions[valid_mask]
        y_valid: np.ndarray = y_positions[valid_mask]

        # Create 2D histogram for surface plot
        hist, x_edges, y_edges = np.histogram2d(
            x=x_valid,
            y=y_valid,
            bins=nbins
        )
        hist_normalized: np.ndarray = hist / np.sum(hist)
        x_centers: np.ndarray = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers: np.ndarray = (y_edges[:-1] + y_edges[1:]) / 2

        # Create frame indices
        frames: np.ndarray = self.dataset.pixel_trajectories.frame_indices

        # Create subplots with custom specs
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Pupil X Position Timeseries",
                "Pupil Y Position Timeseries",
                "2D Trajectory Heatmap",
                "3D Gaze Probability Surface",
                "X Position Distribution",
                "Y Position Distribution"
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "surface"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            row_heights=[0.33, 0.33, 0.33]
        )

        # Row 1: Timeseries plots
        # X position
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=x_positions,
                mode='lines',
                name='X Raw',
                line=dict(color='#4A9EFF', width=1),
                opacity=0.5,
                legendgroup='x'
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=x_filtered,
                mode='lines',
                name='X Filtered',
                line=dict(color='#00D9FF', width=2),
                legendgroup='x'
            ),
            row=1,
            col=1
        )

        # Y position
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=y_positions,
                mode='lines',
                name='Y Raw',
                line=dict(color='#FF6B6B', width=1),
                opacity=0.5,
                legendgroup='y'
            ),
            row=1,
            col=2
        )
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=y_filtered,
                mode='lines',
                name='Y Filtered',
                line=dict(color='#FF3366', width=2),
                legendgroup='y'
            ),
            row=1,
            col=2
        )

        # Row 2: 2D Heatmap and 3D Surface
        # 2D Heatmap
        fig.add_trace(
            go.Histogram2d(
                x=x_valid,
                y=y_valid,
                nbinsx=nbins,
                nbinsy=nbins,
                colorscale='Hot',
                colorbar=dict(
                    title=dict(
                        text="Count",
                        font=dict(color='lightgray')
                    ),
                    x=0.46,
                    len=0.3,
                    y=0.5,
                    tickfont=dict(color='lightgray')
                ),
                hovertemplate='X: %{x}<br>Y: %{y}<br>Count: %{z}<extra></extra>',
                showscale=True
            ),
            row=2,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='lines',
                line=dict(color='#00FFFF', width=1),
                opacity=0.3,
                name='Trajectory',
                showlegend=False,
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
            ),
            row=2,
            col=1
        )

        # 3D Surface
        fig.add_trace(
            go.Surface(
                x=x_centers,
                y=y_centers,
                z=hist_normalized.T,
                colorscale='Viridis',
                colorbar=dict(
                    title=dict(
                        text="Probability",
                        font=dict(color='lightgray')
                    ),
                    x=1.0,
                    len=0.3,
                    y=0.5,
                    tickfont=dict(color='lightgray')
                ),
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>P: %{z:.6f}<extra></extra>',
                showscale=True
            ),
            row=2,
            col=2
        )

        # Row 3: Histograms
        x_valid_hist: np.ndarray = x_positions[~np.isnan(x_positions)]
        y_valid_hist: np.ndarray = y_positions[~np.isnan(y_positions)]

        fig.add_trace(
            go.Histogram(
                x=x_valid_hist,
                nbinsx=nbins,
                name='X Distribution',
                marker=dict(color='#00D9FF', line=dict(color='#0099CC', width=1)),
                hovertemplate='Position: %{x}<br>Count: %{y}<extra></extra>',
                showlegend=False
            ),
            row=3,
            col=1
        )

        fig.add_trace(
            go.Histogram(
                x=y_valid_hist,
                nbinsx=nbins,
                name='Y Distribution',
                marker=dict(color='#FF3366', line=dict(color='#CC0033', width=1)),
                hovertemplate='Position: %{x}<br>Count: %{y}<extra></extra>',
                showlegend=False
            ),
            row=3,
            col=2
        )

        # Update axes labels and styling
        # Row 1
        fig.update_xaxes(title_text="Frame", row=1, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_xaxes(title_text="Frame", row=1, col=2, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_yaxes(title_text="X Position (px)", row=1, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_yaxes(title_text="Y Position (px)", row=1, col=2, color='lightgray', gridcolor='rgb(60, 60, 60)')

        # Row 2
        fig.update_xaxes(title_text="X Position (px)", row=2, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_yaxes(title_text="Y Position (px)", row=2, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')

        # Row 3
        fig.update_xaxes(title_text="X Position (px)", row=3, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_xaxes(title_text="Y Position (px)", row=3, col=2, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_yaxes(title_text="Count", row=3, col=1, color='lightgray', gridcolor='rgb(60, 60, 60)')
        fig.update_yaxes(title_text="Count", row=3, col=2, color='lightgray', gridcolor='rgb(60, 60, 60)')
        # Update 3D scene
        fig.update_scenes(
            xaxis=dict(
                title=dict(
                    text="X (px)",
                    font=dict(color='lightgray')
                ),
                backgroundcolor="rgb(30, 30, 30)",
                gridcolor="rgb(60, 60, 60)",
                tickfont=dict(color='lightgray')
            ),
            yaxis=dict(
                title=dict(
                    text="Y (px)",
                    font=dict(color='lightgray')
                ),
                backgroundcolor="rgb(30, 30, 30)",
                gridcolor="rgb(60, 60, 60)",
                tickfont=dict(color='lightgray')
            ),
            zaxis=dict(
                title=dict(
                    text="Probability",
                    font=dict(color='lightgray')
                ),
                backgroundcolor="rgb(30, 30, 30)",
                gridcolor="rgb(60, 60, 60)",
                tickfont=dict(color='lightgray')
            ),
            bgcolor="rgb(20, 20, 20)"
        )
        # Statistics annotations
        x_mean: float = float(np.mean(x_valid_hist))
        x_std: float = float(np.std(x_valid_hist))
        y_mean: float = float(np.mean(y_valid_hist))
        y_std: float = float(np.std(y_valid_hist))

        fig.add_annotation(
            text=f"μ={x_mean:.1f}<br>σ={x_std:.1f}",
            xref="x5",
            yref="paper",
            x=x_mean,
            y=0.02,
            showarrow=False,
            bgcolor="rgba(0, 217, 255, 0.2)",
            bordercolor="#00D9FF",
            borderwidth=2,
            font=dict(color='lightgray', size=11),
            row=3,
            col=1
        )

        fig.add_annotation(
            text=f"μ={y_mean:.1f}<br>σ={y_std:.1f}",
            xref="x6",
            yref="paper",
            x=y_mean,
            y=0.02,
            showarrow=False,
            bgcolor="rgba(255, 51, 102, 0.2)",
            bordercolor="#FF3366",
            borderwidth=2,
            font=dict(color='lightgray', size=11),
            row=3,
            col=2
        )

        # Update overall layout for dark mode
        fig.update_layout(
            title=dict(
                text=f"Eye Tracking Analysis Dashboard - {self.dataset.data_name}",
                font=dict(color='lightgray', size=20)
            ),
            height=1400,
            width=1600,
            showlegend=True,
            legend=dict(
                font=dict(color='lightgray'),
                bgcolor="rgba(40, 40, 40, 0.8)",
                bordercolor="rgb(80, 80, 80)",
                borderwidth=1
            ),
            hovermode='closest',
            paper_bgcolor="rgb(20, 20, 20)",
            plot_bgcolor="rgb(30, 30, 30)",
            font=dict(color='lightgray')
        )

        # Update subplot title colors
        for annotation in fig.layout.annotations:
            annotation.font.color = 'lightgray'
            annotation.font.size = 14

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(file=str(output_path))

        if show:
            fig.show()

        return fig

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

        # Individual plots (optional, for detailed inspection)
        print("  Creating individual plots...")

        self.plot_pupil_timeseries(
            cutoff=cutoff,
            fs=fs,
            order=order,
            show=False,
            output_path=output_dir / "timeseries.html"
        )

        self.plot_2d_trajectory_heatmap(
            nbins=nbins,
            show=False,
            output_path=output_dir / "trajectory_heatmap.html"
        )

        self.plot_3d_gaze_surface(
            nbins=nbins,
            show=False,
            output_path=output_dir / "gaze_surface_3d.html"
        )

        self.plot_position_histograms(
            nbins=nbins,
            show=False,
            output_path=output_dir / "histograms.html"
        )

        print(f"Analysis report saved to: {output_dir}")



if __name__ == "__main__":
    # Setup paths (same as your original)
    base_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37"
    )
    video_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4"
    )
    timestamps_npy_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy"
    )
    csv_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )

    # Create dataset
    print("Loading eye tracking dataset...")
    eye_dataset: EyeVideoDataset = EyeVideoDataset.create(
        data_name="ferret_757_eye_tracking",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
    )

    # Create analyzer
    print("Initializing analyzer...")
    analyzer: EyeTrackingAnalyzer = EyeTrackingAnalyzer(dataset=eye_dataset)

    # Option 1: Show integrated dashboard (all plots in one view)
    print("\nDisplaying integrated dashboard...")
    analyzer.plot_integrated_dashboard(
        cutoff=5.0,
        fs=30.0,
        order=4,
        nbins=50,
        show=True
    )


    print("\nGenerating complete analysis report...")
    output_dir: Path = base_path / "analysis_output"

    analyzer.create_analysis_report(
        output_dir=output_dir,
        cutoff=5.0,
        fs=30.0,
        order=4,
        nbins=50
    )

    print("\nAnalysis complete!")