"""Integrated dashboard combining all analysis views."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from python_code.eye_data_cleanup.eye_analysis.plots.plot_config import COLORS, get_dark_axis_config, \
    get_dark_layout_config, apply_dark_theme_to_annotations
from python_code.eye_data_cleanup.eye_analysis.signal_processing import apply_butterworth_filter, remove_nan_values


def plot_integrated_dashboard(
    *,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    frames: np.ndarray,
    data_name: str,
    cutoff: float = 5.0,
    fs: float = 30.0,
    order: int = 4,
    nbins: int = 50,
    show: bool = True,
    output_path: Path | None = None
) -> go.Figure:
    """Create integrated dashboard with all analysis views.

    Args:
        x_positions: X coordinate array
        y_positions: Y coordinate array
        frames: Frame indices array
        data_name: Name of dataset for title
        cutoff: Butterworth filter cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Butterworth filter order
        nbins: Number of bins for histograms
        show: Whether to display the plot
        output_path: Optional path to save HTML figure

    Returns:
        Plotly figure object with all analysis views
    """
    # Apply filters
    x_filtered: np.ndarray = apply_butterworth_filter(
        data=x_positions,
        cutoff=cutoff,
        fs=fs,
        order=order
    )
    y_filtered: np.ndarray = apply_butterworth_filter(
        data=y_positions,
        cutoff=cutoff,
        fs=fs,
        order=order
    )

    # Remove NaN values for heatmaps
    x_valid, y_valid = remove_nan_values(x_data=x_positions, y_data=y_positions)

    # Create 2D histogram for surface plot
    hist, x_edges, y_edges = np.histogram2d(
        x=x_valid,
        y=y_valid,
        bins=nbins
    )
    hist_normalized: np.ndarray = hist / np.sum(hist)
    x_centers: np.ndarray = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers: np.ndarray = (y_edges[:-1] + y_edges[1:]) / 2

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
    _add_timeseries_traces(
        fig=fig,
        frames=frames,
        x_positions=x_positions,
        y_positions=y_positions,
        x_filtered=x_filtered,
        y_filtered=y_filtered
    )

    # Row 2: 2D Heatmap and 3D Surface
    _add_heatmap_traces(
        fig=fig,
        x_positions=x_positions,
        y_positions=y_positions,
        x_valid=x_valid,
        y_valid=y_valid,
        nbins=nbins
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
        x_positions=x_positions,
        y_positions=y_positions,
        nbins=nbins
    )

    # Update axes and styling
    _update_dashboard_axes(fig=fig)
    _update_dashboard_layout(fig=fig, data_name=data_name)
    _add_statistics_annotations(fig=fig, x_positions=x_positions, y_positions=y_positions)

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
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    x_filtered: np.ndarray,
    y_filtered: np.ndarray
) -> None:
    """Add timeseries traces to dashboard."""
    # X position
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=x_positions,
            mode='lines+markers',
            marker=dict(size=4),
            name='X Raw',
            line=dict(color=COLORS['x_raw'], width=1),
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
            mode='lines+markers',
            marker=dict(size=4),
            name='X Filtered',
            line=dict(color=COLORS['x_filtered'], width=2),
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
            mode='lines+markers',
            marker=dict(size=4),
            name='Y Raw',
            line=dict(color=COLORS['y_raw'], width=1),
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
            mode='lines+markers',
            marker=dict(size=4),
            name='Y Filtered',
            line=dict(color=COLORS['y_filtered'], width=2),
            legendgroup='y'
        ),
        row=1,
        col=2
    )


def _add_heatmap_traces(
    *,
    fig: go.Figure,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    nbins: int
) -> None:
    """Add 2D heatmap traces to dashboard."""
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
                    font=dict(color=COLORS['text'])
                ),
                x=0.46,
                len=0.3,
                y=0.5,
                tickfont=dict(color=COLORS['text'])
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
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(color=COLORS['trajectory'], width=1),
            opacity=0.3,
            name='Trajectory',
            showlegend=False,
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
        ),
        row=2,
        col=1
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
                title=dict(
                    text="Probability",
                    font=dict(color=COLORS['text'])
                ),
                x=1.0,
                len=0.3,
                y=0.5,
                tickfont=dict(color=COLORS['text'])
            ),
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>P: %{z:.6f}<extra></extra>',
            showscale=True
        ),
        row=2,
        col=2
    )


def _add_histogram_traces(
    *,
    fig: go.Figure,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    nbins: int
) -> None:
    """Add histogram traces to dashboard."""
    x_valid_hist: np.ndarray = x_positions[~np.isnan(x_positions)]
    y_valid_hist: np.ndarray = y_positions[~np.isnan(y_positions)]

    fig.add_trace(
        go.Histogram(
            x=x_valid_hist,
            nbinsx=nbins,
            name='X Distribution',
            marker=dict(color=COLORS['x_filtered'], line=dict(color='#0099CC', width=1)),
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
            marker=dict(color=COLORS['y_filtered'], line=dict(color='#CC0033', width=1)),
            hovertemplate='Position: %{x}<br>Count: %{y}<extra></extra>',
            showlegend=False
        ),
        row=3,
        col=2
    )


def _update_dashboard_axes(*, fig: go.Figure) -> None:
    """Update all axes styling for dashboard."""
    axis_config: dict[str, object] = get_dark_axis_config()
    
    # Row 1
    fig.update_xaxes(title_text="Frame", row=1, col=1, **axis_config)
    fig.update_xaxes(title_text="Frame", row=1, col=2, **axis_config)
    fig.update_yaxes(title_text="X Position (px)", row=1, col=1, **axis_config)
    fig.update_yaxes(title_text="Y Position (px)", row=1, col=2, **axis_config)

    # Row 2
    fig.update_xaxes(title_text="X Position (px)", row=2, col=1, **axis_config)
    fig.update_yaxes(title_text="Y Position (px)", row=2, col=1, **axis_config)

    # Row 3
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
    layout_config: dict[str, object] = get_dark_layout_config()
    
    fig.update_layout(
        title=dict(
            text=f"Eye Tracking Analysis Dashboard - {data_name}",
            font=dict(color=COLORS['text'], size=20)
        ),
        height=1400,
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
    x_valid: np.ndarray = x_positions[~np.isnan(x_positions)]
    y_valid: np.ndarray = y_positions[~np.isnan(y_positions)]
    
    x_mean: float = float(np.mean(x_valid))
    x_std: float = float(np.std(x_valid))
    y_mean: float = float(np.mean(y_valid))
    y_std: float = float(np.std(y_valid))

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
        bordercolor=COLORS['y_filtered'],
        borderwidth=2,
        font=dict(color=COLORS['text'], size=11),
        row=3,
        col=2
    )
