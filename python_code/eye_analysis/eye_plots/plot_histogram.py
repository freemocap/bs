"""Histogram plotting for pupil position distributions."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from python_code.eye_analysis.eye_plots.plot_config import COLORS, get_dark_layout_config, get_dark_axis_config, apply_dark_theme_to_annotations


def plot_position_histograms(
    *,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    data_name: str,
    nbins: int = 50,
    show: bool = True,
    output_path: Path | None = None
) -> go.Figure:
    """Create histograms of X and Y position distributions.

    Args:
        x_positions: X coordinate array
        y_positions: Y coordinate array
        data_name: Name of dataset for title
        nbins: Number of histogram bins
        show: Whether to display the plot
        output_path: Optional path to save HTML figure

    Returns:
        Plotly figure object
    """
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
            keypoint=dict(color='blue', line=dict(color='darkblue', width=1)),
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
            keypoint=dict(color='red', line=dict(color='darkred', width=1)),
            hovertemplate='Position: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=1,
        col=2
    )

    # Calculate statistics
    x_mean: float = float(np.mean(x_valid))
    x_std: float = float(np.std(x_valid))
    y_mean: float = float(np.mean(y_valid))
    y_std: float = float(np.std(y_valid))

    # Add statistics annotations
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
        font=dict(color=COLORS['text']),
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
        font=dict(color=COLORS['text']),
        row=1,
        col=2
    )

    # Update axes
    axis_config: dict[str, object] = get_dark_axis_config()
    fig.update_xaxes(title_text="X Position (pixels)", row=1, col=1, **axis_config)
    fig.update_xaxes(title_text="Y Position (pixels)", row=1, col=2, **axis_config)
    fig.update_yaxes(title_text="Count", row=1, col=1, **axis_config)
    fig.update_yaxes(title_text="Count", row=1, col=2, **axis_config)

    # Update layout
    layout_config: dict[str, object] = get_dark_layout_config()
    fig.update_layout(
        title=dict(
            text=f"Position Distributions - {data_name}",
            font=dict(color=COLORS['text'])
        ),
        height=500,
        width=1000,
        showlegend=False,
        **layout_config
    )

    # Apply theme to annotations
    apply_dark_theme_to_annotations(fig=fig)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(file=str(output_path))

    if show:
        fig.show()

    return fig
