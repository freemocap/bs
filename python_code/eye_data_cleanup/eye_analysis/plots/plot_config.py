"""Shared plotting configuration and theme settings."""

# Dark theme color palette
COLORS: dict[str, str] = {
    'background_paper': 'rgb(20, 20, 20)',
    'background_plot': 'rgb(30, 30, 30)',
    'background_legend': 'rgba(40, 40, 40, 0.8)',
    'text': 'lightgray',
    'grid': 'rgb(60, 60, 60)',
    'border': 'rgb(80, 80, 80)',
    
    # Data colors
    'x_raw': '#4A9EFF',
    'x_filtered': '#00D9FF',
    'y_raw': '#FF6B6B',
    'y_filtered': '#FF3366',
    'trajectory': '#00FFFF',
    'cyan': 'cyan',
}


def get_dark_layout_config() -> dict[str, object]:
    """Get standard dark theme layout configuration.

    Returns:
        Dictionary of layout parameters for Plotly figures
    """
    return {
        'paper_bgcolor': COLORS['background_paper'],
        'plot_bgcolor': COLORS['background_plot'],
        'font': dict(color=COLORS['text']),
        'legend': dict(
            font=dict(color=COLORS['text']),
            bgcolor=COLORS['background_legend'],
            bordercolor=COLORS['border'],
            borderwidth=1
        )
    }


def get_dark_axis_config() -> dict[str, object]:
    """Get standard dark theme axis configuration.

    Returns:
        Dictionary of axis parameters for Plotly figures
    """
    return {
        'color': COLORS['text'],
        'gridcolor': COLORS['grid']
    }


def get_dark_scene_config() -> dict[str, object]:
    """Get standard dark theme 3D scene configuration.

    Returns:
        Dictionary of 3D scene parameters for Plotly figures
    """
    return {
        'xaxis': dict(
            backgroundcolor=COLORS['background_plot'],
            gridcolor=COLORS['grid'],
            titlefont=dict(color=COLORS['text']),
            tickfont=dict(color=COLORS['text'])
        ),
        'yaxis': dict(
            backgroundcolor=COLORS['background_plot'],
            gridcolor=COLORS['grid'],
            titlefont=dict(color=COLORS['text']),
            tickfont=dict(color=COLORS['text'])
        ),
        'zaxis': dict(
            backgroundcolor=COLORS['background_plot'],
            gridcolor=COLORS['grid'],
            titlefont=dict(color=COLORS['text']),
            tickfont=dict(color=COLORS['text'])
        ),
        'bgcolor': COLORS['background_paper']
    }


def apply_dark_theme_to_annotations(*, fig: object) -> None:
    """Apply dark theme styling to all figure annotations.

    Args:
        fig: Plotly figure object to modify in-place
    """
    for annotation in fig.layout.annotations:
        annotation.font.color = COLORS['text']
