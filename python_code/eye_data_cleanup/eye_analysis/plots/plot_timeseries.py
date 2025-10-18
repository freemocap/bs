"""Timeseries plotting for pupil position data."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from python_code.eye_data_cleanup.eye_analysis.signal_processing import apply_butterworth_filter
from python_code.eye_data_cleanup.eye_analysis.plots.plot_config import COLORS, get_dark_layout_config, get_dark_axis_config, apply_dark_theme_to_annotations


def plot_pupil_timeseries(
    *,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    frames: np.ndarray,
    data_name: str,
    cutoff: float = 5.0,
    fs: float = 30.0,
    order: int = 4,
    show: bool = True,
    output_path: Path | None = None
) -> go.Figure:
    """Create timeseries plots of pupil center X and Y positions.

    Shows both raw and butterworth-filtered data.

    Args:
        x_positions: X coordinate array
        y_positions: Y coordinate array
        frames: Frame indices array
        data_name: Name of dataset for title
        cutoff: Butterworth filter cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Butterworth filter order
        show: Whether to display the plot
        output_path: Optional path to save HTML figure

    Returns:
        Plotly figure object
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

    # Update axes
    axis_config: dict[str, object] = get_dark_axis_config()
    fig.update_xaxes(title_text="Frame", row=2, col=1, **axis_config)
    fig.update_yaxes(title_text="X Position (pixels)", row=1, col=1, **axis_config)
    fig.update_yaxes(title_text="Y Position (pixels)", row=2, col=1, **axis_config)

    # Update layout
    layout_config: dict[str, object] = get_dark_layout_config()
    fig.update_layout(
        title=dict(
            text=f"Pupil Center Timeseries - {data_name}",
            font=dict(color=COLORS['text'])
        ),
        height=700,
        showlegend=True,
        hovermode='x unified',
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
