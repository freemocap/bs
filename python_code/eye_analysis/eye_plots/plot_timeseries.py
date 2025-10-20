"""Timeseries plotting for pupil position data - simplified."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from python_code.eye_analysis.data_models.trajectory_dataset import TrajectoryDataset
from python_code.eye_analysis.eye_plots.plot_config import (
    COLORS, get_dark_layout_config, get_dark_axis_config, apply_dark_theme_to_annotations
)


def plot_pupil_timeseries(
    *,
    dataset: TrajectoryDataset,
    frames: np.ndarray,
    data_name: str,
    show: bool = True,
    output_path: Path | None = None
) -> go.Figure:
    """Create timeseries plots of pupil center X and Y positions.

    Shows both raw and cleaned (filtered) data from the dataset.

    Args:
        dataset: Trajectory dataset with raw and cleaned data
        frames: Frame indices array
        data_name: Name of dataset for title
        show: Whether to display the plot
        output_path: Optional path to save HTML figure

    Returns:
        Plotly figure object
    """
    # Get pupil landmark pairs
    pupil_pairs = [dataset.pairs[f'p{i}'] for i in range(1, 9)]
    tear_duct = dataset.pairs['tear_duct']
    eye_outer = dataset.pairs['outer_eye']

    # Compute average pupil positions
    pupil_x_raw = np.mean([pair.raw.x for pair in pupil_pairs], axis=0)
    pupil_y_raw = np.mean([pair.raw.y for pair in pupil_pairs], axis=0)
    pupil_x_cleaned = np.mean([pair.cleaned.x for pair in pupil_pairs], axis=0)
    pupil_y_cleaned = np.mean([pair.cleaned.y for pair in pupil_pairs], axis=0)

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Pupil X Position", "Pupil Y Position"),
        vertical_spacing=0.12
    )

    # X position plot - Pupil
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pupil_x_raw,
            mode='lines+markers',
            marker=dict(size=4),
            name='Pupil X Raw',
            line=dict(color='lightblue', width=1),
            opacity=0.6
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pupil_x_cleaned,
            mode='lines+markers',
            marker=dict(size=4),
            name='Pupil X Cleaned',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # X position - Tear Duct
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=tear_duct.raw.x,
            mode='lines+markers',
            marker=dict(size=3),
            name='Tear Duct X',
            line=dict(color='green', width=1, dash='dot'),
            opacity=0.7
        ),
        row=1, col=1
    )

    # X position - Eye Outer
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=eye_outer.raw.x,
            mode='lines+markers',
            marker=dict(size=3),
            name='Eye Outer X',
            line=dict(color='purple', width=1, dash='dot'),
            opacity=0.7
        ),
        row=1, col=1
    )

    # Y position plot - Pupil
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pupil_y_raw,
            mode='lines+markers',
            marker=dict(size=4),
            name='Pupil Y Raw',
            line=dict(color='lightcoral', width=1),
            opacity=0.6
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=pupil_y_cleaned,
            mode='lines+markers',
            marker=dict(size=4),
            name='Pupil Y Cleaned',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )

    # Y position - Tear Duct
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=tear_duct.raw.y,
            mode='lines+markers',
            marker=dict(size=3),
            name='Tear Duct Y',
            line=dict(color='green', width=1, dash='dot'),
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=1
    )

    # Y position - Eye Outer
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=eye_outer.raw.y,
            mode='lines+markers',
            marker=dict(size=3),
            name='Eye Outer Y',
            line=dict(color='purple', width=1, dash='dot'),
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=1
    )

    # Update axes
    axis_config = get_dark_axis_config()
    fig.update_xaxes(title_text="Frame", row=2, col=1, **axis_config)
    fig.update_yaxes(title_text="X Position (pixels)", row=1, col=1, **axis_config)
    fig.update_yaxes(title_text="Y Position (pixels)", row=2, col=1, **axis_config)

    # Update layout
    layout_config = get_dark_layout_config()
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