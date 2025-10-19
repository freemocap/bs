"""Integrated dashboard combining all analysis views - simplified using TrajectoryDataset."""

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image

from python_code.eye_data_cleanup.csv_io import TrajectoryDataset, Trajectory2D
from python_code.eye_data_cleanup.eye_analysis.plots.plot_config import (
    COLORS, get_dark_axis_config, get_dark_layout_config, apply_dark_theme_to_annotations
)
from python_code.eye_data_cleanup.eye_analysis.signal_processing import remove_nan_values


def plot_integrated_dashboard(
    *,
    dataset: TrajectoryDataset,
    data_name: str,
    nbins: int = 50,
    show: bool = True,
    output_path: Path | None = None,
    video_frame: np.ndarray | None = None
) -> go.Figure:
    """Create integrated dashboard with all analysis views.

    Args:
        dataset: Trajectory dataset with raw and cleaned data
        data_name: Name of dataset for title
        nbins: Number of bins for histograms
        show: Whether to display the plot
        output_path: Optional path to save HTML figure
        video_frame: Optional video frame to display (BGR format)

    Returns:
        Plotly figure object with all analysis views
    """
    # Extract pupil trajectory pairs
    pupil_pairs = [dataset.pairs[f'p{i}'] for i in range(1, 9)]
    tear_duct = dataset.pairs['tear_duct']
    eye_outer = dataset.pairs['outer_eye']

    # Get frame indices
    frames = dataset.frame_indices

    # Compute pupil centers from cleaned data
    pupil_x_cleaned = np.mean([pair.cleaned.x for pair in pupil_pairs], axis=0)
    pupil_y_cleaned = np.mean([pair.cleaned.y for pair in pupil_pairs], axis=0)

    # Get raw pupil data for heatmaps
    pupil_x_raw = np.mean([pair.raw.x for pair in pupil_pairs], axis=0)
    pupil_y_raw = np.mean([pair.raw.y for pair in pupil_pairs], axis=0)

    # Remove NaN values for heatmaps
    x_valid, y_valid = remove_nan_values(x_data=pupil_x_raw, y_data=pupil_y_raw)

    # Create 2D histogram for surface plot
    hist, x_edges, y_edges = np.histogram2d(x=x_valid, y=y_valid, bins=nbins)
    hist_normalized = hist / np.sum(hist)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Get image dimensions if video frame is provided
    img_height = None
    img_width = None
    if video_frame is not None:
        img_height, img_width = video_frame.shape[:2]

    # Create subplots
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
    _add_timeseries_traces(
        fig=fig,
        frames=frames,
        pupil_pairs=pupil_pairs,
        tear_duct=tear_duct,
        eye_outer=eye_outer
    )

    # Row 2: 2D Heatmap and 3D Surface
    _add_heatmap_traces(
        fig=fig,
        x_positions=pupil_x_raw,
        y_positions=pupil_y_raw,
        x_valid=x_valid,
        y_valid=y_valid,
        nbins=nbins,
        video_frame=video_frame
    )

    _add_surface_trace(
        fig=fig,
        x_centers=x_centers,
        y_centers=y_centers,
        hist_normalized=hist_normalized
    )

    # Row 3: Histograms
    _add_histogram_traces(
        fig=fig,
        x_positions=pupil_x_raw,
        y_positions=pupil_y_raw,
        nbins=nbins
    )

    # Update axes and styling
    _update_dashboard_axes(fig=fig, img_width=img_width, img_height=img_height)
    _update_dashboard_layout(fig=fig, data_name=data_name)
    _add_statistics_annotations(fig=fig, x_positions=pupil_x_raw, y_positions=pupil_y_raw)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(file=str(output_path))

    if show:
        fig.show()

    return fig


def _add_timeseries_traces(
    *,
    fig: go.Figure,
    frames: np.ndarray,
    pupil_pairs: list,
    tear_duct,
    eye_outer
) -> None:
    """Add timeseries traces to dashboard using trajectory pairs."""
    # Compute average pupil positions
    pupil_x_raw = np.mean([pair.raw.x for pair in pupil_pairs], axis=0)
    pupil_y_raw = np.mean([pair.raw.y for pair in pupil_pairs], axis=0)
    pupil_x_cleaned = np.mean([pair.cleaned.x for pair in pupil_pairs], axis=0)
    pupil_y_cleaned = np.mean([pair.cleaned.y for pair in pupil_pairs], axis=0)

    # X position - Pupil
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pupil_x_raw,
            mode='lines+markers',
            marker=dict(size=4),
            name='Pupil X Raw',
            line=dict(color=COLORS['x_raw'], width=1),
            opacity=0.5,
            legendgroup='x'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pupil_x_cleaned,
            mode='lines+markers',
            marker=dict(size=4),
            name='Pupil X Filtered',
            line=dict(color=COLORS['x_filtered'], width=2),
            legendgroup='x'
        ),
        row=1, col=1
    )

    # X position - Tear Duct (green)
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=tear_duct.raw.x,
            mode='lines+markers',
            marker=dict(size=3),
            name='Tear Duct X Raw',
            line=dict(color='rgb(100, 200, 100)', width=1, dash='dot'),
            opacity=0.5,
            legendgroup='tear_duct'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=tear_duct.cleaned.x,
            mode='lines+markers',
            marker=dict(size=3),
            name='Tear Duct X Filtered',
            line=dict(color='rgb(0, 255, 0)', width=2, dash='dot'),
            legendgroup='tear_duct'
        ),
        row=1, col=1
    )

    # X position - Eye Outer (purple)
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=eye_outer.raw.x,
            mode='lines+markers',
            marker=dict(size=3),
            name='Eye Outer X Raw',
            line=dict(color='rgb(200, 100, 200)', width=1, dash='dot'),
            opacity=0.5,
            legendgroup='eye_outer'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=eye_outer.cleaned.x,
            mode='lines+markers',
            marker=dict(size=3),
            name='Eye Outer X Filtered',
            line=dict(color='rgb(200, 0, 255)', width=2, dash='dot'),
            legendgroup='eye_outer'
        ),
        row=1, col=1
    )

    # Y position - Pupil
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pupil_y_raw,
            mode='lines+markers',
            marker=dict(size=4),
            name='Pupil Y Raw',
            line=dict(color=COLORS['y_raw'], width=1),
            opacity=0.5,
            legendgroup='y'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pupil_y_cleaned,
            mode='lines+markers',
            marker=dict(size=4),
            name='Pupil Y Filtered',
            line=dict(color=COLORS['y_filtered'], width=2),
            legendgroup='y'
        ),
        row=1, col=2
    )

    # Y position - Tear Duct (green)
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=tear_duct.raw.y,
            mode='lines+markers',
            marker=dict(size=3),
            name='Tear Duct Y Raw',
            line=dict(color='rgb(100, 200, 100)', width=1, dash='dot'),
            opacity=0.5,
            legendgroup='tear_duct',
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=tear_duct.cleaned.y,
            mode='lines+markers',
            marker=dict(size=3),
            name='Tear Duct Y Filtered',
            line=dict(color='rgb(0, 255, 0)', width=2, dash='dot'),
            legendgroup='tear_duct',
            showlegend=False
        ),
        row=1, col=2
    )

    # Y position - Eye Outer (purple)
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=eye_outer.raw.y,
            mode='lines+markers',
            marker=dict(size=3),
            name='Eye Outer Y Raw',
            line=dict(color='rgb(200, 100, 200)', width=1, dash='dot'),
            opacity=0.5,
            legendgroup='eye_outer',
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=eye_outer.cleaned.y,
            mode='lines+markers',
            marker=dict(size=3),
            name='Eye Outer Y Filtered',
            line=dict(color='rgb(200, 0, 255)', width=2, dash='dot'),
            legendgroup='eye_outer',
            showlegend=False
        ),
        row=1, col=2
    )


def _add_heatmap_traces(
    *,
    fig: go.Figure,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    nbins: int,
    video_frame: np.ndarray | None
) -> None:
    """Add 2D heatmap traces to dashboard with optional video frame background."""

    # Add video frame as background image if provided
    if video_frame is not None:
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(src=video_frame, code=cv2.COLOR_BGR2RGB)
        img_height, img_width = frame_rgb.shape[:2]

        # Convert numpy array to PIL Image for Plotly
        pil_image = Image.fromarray(obj=frame_rgb)

        # Add background image using layout image
        fig.add_layout_image(
            dict(
                source=pil_image,
                xref="x3",
                yref="y3",
                x=0,
                y=img_height,
                sizex=img_width,
                sizey=img_height,
                sizing="stretch",
                opacity=0.5,
                layer="below"
            )
        )

    fig.add_trace(
        go.Histogram2d(
            x=x_valid,
            y=y_valid,
            nbinsx=nbins,
            nbinsy=nbins,
            colorscale='Hot',
            colorbar=dict(
                title=dict(text="Count", font=dict(color=COLORS['text'])),
                x=0.46,
                len=0.3,
                y=0.5,
                tickfont=dict(color=COLORS['text'])
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Count: %{z}<extra></extra>',
            showscale=True
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(color=COLORS['trajectory'], width=1),
            opacity=0.3,
            name='Trajectory',
            showlegend=False,
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
        ),
        row=2, col=1
    )


def _add_surface_trace(
    *,
    fig: go.Figure,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    hist_normalized: np.ndarray
) -> None:
    """Add 3D surface trace to dashboard."""
    fig.add_trace(
        go.Surface(
            x=x_centers,
            y=y_centers,
            z=hist_normalized.T,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text="Probability", font=dict(color=COLORS['text'])),
                x=1.0,
                len=0.3,
                y=0.5,
                tickfont=dict(color=COLORS['text'])
            ),
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>P: %{z:.6f}<extra></extra>',
            showscale=True
        ),
        row=2, col=2
    )


def _add_histogram_traces(
    *,
    fig: go.Figure,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    nbins: int
) -> None:
    """Add histogram traces to dashboard."""
    x_valid_hist = x_positions[~np.isnan(x_positions)]
    y_valid_hist = y_positions[~np.isnan(y_positions)]

    fig.add_trace(
        go.Histogram(
            x=x_valid_hist,
            nbinsx=nbins,
            name='X Distribution',
            marker=dict(color=COLORS['x_filtered'], line=dict(color='#0099CC', width=1)),
            hovertemplate='Position: %{x}<br>Count: %{y}<extra></extra>',
            showlegend=False
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Histogram(
            x=y_valid_hist,
            nbinsx=nbins,
            name='Y Distribution',
            marker=dict(color=COLORS['y_filtered'], line=dict(color='#CC0033', width=1)),
            hovertemplate='Position: %{x}<br>Count: %{y}<extra></extra>',
            showlegend=False
        ),
        row=3, col=2
    )


def _update_dashboard_axes(
    *,
    fig: go.Figure,
    img_width: int | None = None,
    img_height: int | None = None
) -> None:
    """Update all axes styling for dashboard."""
    axis_config = get_dark_axis_config()

    # Row 1: Timeseries
    fig.update_xaxes(title_text="Frame", row=1, col=1, **axis_config)
    fig.update_xaxes(title_text="Frame", row=1, col=2, **axis_config)
    fig.update_yaxes(title_text="X Position (px)", row=1, col=1, **axis_config)
    fig.update_yaxes(title_text="Y Position (px)", row=1, col=2, **axis_config)

    # Row 2: Heatmap and Surface
    fig.update_xaxes(title_text="X Position (px)", row=2, col=1, **axis_config)
    fig.update_yaxes(title_text="Y Position (px)", row=2, col=1, **axis_config)

    # Set axis ranges to match image dimensions if provided
    if img_width is not None and img_height is not None:
        fig.update_xaxes(range=[0, img_width], row=2, col=1)
        fig.update_yaxes(range=[img_height, 0], row=2, col=1)  # Invert Y axis for image coordinates

    # Row 3: Histograms
    fig.update_xaxes(title_text="X Position (px)", row=3, col=1, **axis_config)
    fig.update_xaxes(title_text="Y Position (px)", row=3, col=2, **axis_config)
    fig.update_yaxes(title_text="Count", row=3, col=1, **axis_config)
    fig.update_yaxes(title_text="Count", row=3, col=2, **axis_config)

    # Update 3D scene
    fig.update_scenes(
        xaxis=dict(
            title=dict(text="X (px)", font=dict(color=COLORS['text'])),
            backgroundcolor=COLORS['background_plot'],
            gridcolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text'])
        ),
        yaxis=dict(
            title=dict(text="Y (px)", font=dict(color=COLORS['text'])),
            backgroundcolor=COLORS['background_plot'],
            gridcolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text'])
        ),
        zaxis=dict(
            title=dict(text="Probability", font=dict(color=COLORS['text'])),
            backgroundcolor=COLORS['background_plot'],
            gridcolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text'])
        ),
        bgcolor=COLORS['background_paper']
    )


def _update_dashboard_layout(*, fig: go.Figure, data_name: str) -> None:
    """Update overall dashboard layout."""
    layout_config = get_dark_layout_config()

    fig.update_layout(
        title=dict(
            text=f"Eye Tracking Analysis Dashboard - {data_name}",
            font=dict(color=COLORS['text'], size=20)
        ),
        height=1500,
        width=1600,
        showlegend=True,
        hovermode='closest',
        **layout_config
    )

    apply_dark_theme_to_annotations(fig=fig)

    # Increase subtitle size
    for annotation in fig.layout.annotations:
        annotation.font.size = 14


def _add_statistics_annotations(
    *,
    fig: go.Figure,
    x_positions: np.ndarray,
    y_positions: np.ndarray
) -> None:
    """Add statistics annotations to histograms."""
    x_valid = x_positions[~np.isnan(x_positions)]
    y_valid = y_positions[~np.isnan(y_positions)]

    x_mean = float(np.mean(x_valid))
    x_std = float(np.std(x_valid))
    y_mean = float(np.mean(y_valid))
    y_std = float(np.std(y_valid))

    fig.add_annotation(
        text=f"μ={x_mean:.1f}<br>σ={x_std:.1f}",
        xref="x5",
        yref="paper",
        x=x_mean,
        y=0.02,
        showarrow=False,
        bgcolor="rgba(0, 217, 255, 0.2)",
        bordercolor=COLORS['x_filtered'],
        borderwidth=2,
        font=dict(color=COLORS['text'], size=11),
        row=3, col=1
    )

    fig.add_annotation(
        text=f"μ={y_mean:.1f}<br>σ={y_std:.1f}",
        xref="x6",
        yref="paper",
        x=y_mean,
        y=0.02,
        showarrow=False,
        bgcolor="rgba(255, 51, 102, 0.2)",
        bordercolor=COLORS['y_filtered'],
        borderwidth=2,
        font=dict(color=COLORS['text'], size=11),
        row=3, col=2
    )