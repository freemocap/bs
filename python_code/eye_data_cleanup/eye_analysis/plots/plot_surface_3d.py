"""3D surface plotting for pupil position probability distribution."""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from python_code.eye_data_cleanup.eye_analysis.signal_processing import remove_nan_values
from python_code.eye_data_cleanup.eye_analysis.plots.plot_config import COLORS, get_dark_layout_config, get_dark_scene_config


def plot_3d_gaze_surface(
    *,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    data_name: str,
    nbins: int = 50,
    colorscale: str = 'Hot',
    show: bool = True,
    output_path: Path | None = None
) -> go.Figure:
    """Create 3D surface plot of pupil position probability distribution.

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
                        font=dict(color=COLORS['text'])
                    ),
                    tickfont=dict(color=COLORS['text'])
                ),
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Probability: %{z:.6f}<extra></extra>'
            )
        ]
    )

    # Update layout
    scene_config: dict[str, object] = get_dark_scene_config()
    scene_config['xaxis']['title'] = "X Position (pixels)"
    scene_config['yaxis']['title'] = "Y Position (pixels)"
    scene_config['zaxis']['title'] = "Probability"
    
    layout_config: dict[str, object] = get_dark_layout_config()
    
    fig.update_layout(
        title=dict(
            text=f"3D Gaze Probability Surface - {data_name}",
            font=dict(color=COLORS['text'])
        ),
        scene=scene_config,
        width=900,
        height=800,
        **layout_config
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(file=str(output_path))

    if show:
        fig.show()

    return fig
