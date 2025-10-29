"""2D heatmap plotting for pupil position distribution."""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from python_code.eye_analysis.eye_plots.plot_config import COLORS, get_dark_layout_config, get_dark_axis_config
from python_code.eye_analysis.data_processing.signal_processing import remove_nan_values


def plot_2d_trajectory_heatmap(
    *,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    data_name: str,
    nbins: int = 50,
    colorscale: str = 'Hot',
    show: bool = True,
    output_path: Path | None = None
) -> go.Figure:
    """Create 2D heatmap of pupil position distribution.

    Args:
        x_positions: X coordinate array
        y_positions: Y coordinate array
        data_name: Name of dataset for title
        nbins: Number of bins for histogram
        colorscale: Plotly colorscale name
        show: Whether to display the plot
        output_path: Optional path to save HTML figure

    Returns:
        Plotly figure object
    """
    # Remove NaN values
    x_valid, y_valid = remove_nan_values(x_data=x_positions, y_data=y_positions)

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
                    font=dict(color=COLORS['text'])
                ),
                tickfont=dict(color=COLORS['text'])
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
            line=dict(color=COLORS['cyan'], width=1),
            opacity=0.3,
            name='Trajectory',
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
        )
    )

    # Update layout
    axis_config: dict[str, object] = get_dark_axis_config()
    layout_config: dict[str, object] = get_dark_layout_config()
    
    fig.update_layout(
        title=dict(
            text=f"Pupil Position 2D Distribution - {data_name}",
            font=dict(color=COLORS['text'])
        ),
        xaxis_title="X Position (pixels)",
        yaxis_title="Y Position (pixels)",
        width=800,
        height=800,
        yaxis=dict(scaleanchor="x", scaleratio=1, **axis_config),
        xaxis=axis_config,
        **layout_config
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(file=str(output_path))

    if show:
        fig.show()

    return fig
