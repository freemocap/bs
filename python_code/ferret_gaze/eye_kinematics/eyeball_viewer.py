"""
Interactive 3D viewer for EyeballReferenceGeometry using Plotly.

Directly uses the EyeballReferenceGeometry model from the pipeline
to ensure the visualization matches the actual geometry definition.
"""

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from ferret_eye_kinematics_pipeline import (
    EyeballReferenceGeometry,
    EyeGeometryParameters,
    NUM_PUPIL_POINTS,
)


def generate_sphere_mesh(
    radius: float,
    resolution: int = 30,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate sphere surface mesh for visualization."""
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution * 2)
    theta, phi = np.meshgrid(theta, phi)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    return x, y, z


def create_eyeball_figure(
    geometry: EyeballReferenceGeometry,
    show_sphere: bool = True,
    show_axes: bool = True,
    title: str = "Eyeball Reference Geometry",
) -> go.Figure:
    """
    Create an interactive 3D Plotly figure from EyeballReferenceGeometry.
    
    Args:
        geometry: The eyeball reference geometry to visualize
        show_sphere: Whether to show the translucent sphere surface
        show_axes: Whether to show coordinate axes
        title: Plot title
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    markers = geometry.get_marker_positions()
    R = geometry.eye_radius_mm
    
    # Sphere surface
    if show_sphere:
        x, y, z = generate_sphere_mesh(R, resolution=30)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.3,
            colorscale=[[0, 'rgb(220, 220, 240)'], [1, 'rgb(180, 180, 220)']],
            showscale=False,
            hoverinfo='skip',
            name='Eyeball Surface',
        ))
    
    # Coordinate axes
    if show_axes:
        axis_length = R * 1.5
        # +X axis (red) - gaze direction
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines',
            line=dict(color='red', width=6),
            name='+X (gaze)',
            hoverinfo='name',
        ))
        # +Y axis (green) - medial
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines',
            line=dict(color='green', width=6),
            name='+Y (medial)',
            hoverinfo='name',
        ))
        # +Z axis (blue) - up
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines',
            line=dict(color='blue', width=6),
            name='+Z (up)',
            hoverinfo='name',
        ))
    
    # Display edges from geometry
    for m1, m2 in geometry.display_edges:
        p1 = markers[m1]
        p2 = markers[m2]
        is_pupil_edge = m1.startswith('p') and m2.startswith('p')
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='lines',
            line=dict(
                color='black' if is_pupil_edge else 'gray',
                width=6 if is_pupil_edge else 3,
            ),
            showlegend=False,
            hoverinfo='skip',
        ))
    
    # Marker groups with different colors
    marker_groups = {
        'Eyeball Center': {
            'markers': ['eyeball_center'],
            'color': 'purple',
            'size': 10,
            'symbol': 'diamond',
        },
        'Pupil Center': {
            'markers': ['pupil_center'],
            'color': 'black',
            'size': 12,
            'symbol': 'circle',
        },
        'Pupil Boundary': {
            'markers': [f'p{i+1}' for i in range(NUM_PUPIL_POINTS)],
            'color': 'darkblue',
            'size': 8,
            'symbol': 'circle',
        },
        'Landmarks': {
            'markers': ['tear_duct', 'outer_eye'],
            'color': 'orange',
            'size': 10,
            'symbol': 'square',
        },
    }
    
    for group_name, group in marker_groups.items():
        x, y, z, text = [], [], [], []
        for name in group['markers']:
            pos = markers[name]
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
            text.append(f"{name}<br>x: {pos[0]:.3f}<br>y: {pos[1]:.3f}<br>z: {pos[2]:.3f}")
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(
                size=group['size'],
                color=group['color'],
                symbol=group['symbol'],
            ),
            text=group['markers'],
            textposition='top center',
            textfont=dict(size=10, color=group['color']),
            name=group_name,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=text,
        ))
    
    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (mm) - Gaze'),
            yaxis=dict(title='Y (mm) - Medial'),
            zaxis=dict(title='Z (mm) - Up'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    # Add annotation for coordinate system
    fig.add_annotation(
        text=(
            "<b>Coordinate System:</b> Origin at eyeball center | "
            "<span style='color:red'>+X = Gaze</span> | "
            "<span style='color:green'>+Y = Medial</span> | "
            "<span style='color:blue'>+Z = Up</span>"
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.02,
        showarrow=False,
        font=dict(size=11),
    )
    
    return fig


def create_interactive_viewer() -> go.FigureWidget:
    """
    Create an interactive viewer with sliders for geometry parameters.
    
    Returns a FigureWidget that can be displayed in Jupyter notebooks.
    """
    import ipywidgets as widgets
    from IPython.display import display
    
    # Initial geometry
    eye_params = EyeGeometryParameters(
        eye_diameter_mm=7.0,
        pupil_radius_mm=0.5,
        pupil_eccentricity=0.8,
    )
    geometry = eye_params.to_reference_geometry()
    
    # Create initial figure
    fig = go.FigureWidget(create_eyeball_figure(geometry))
    
    # Sliders
    eye_radius_slider = widgets.FloatSlider(
        value=3.5, min=2.0, max=6.0, step=0.1,
        description='Eye Radius (mm):',
        style={'description_width': '120px'},
    )
    pupil_radius_slider = widgets.FloatSlider(
        value=0.5, min=0.2, max=1.5, step=0.05,
        description='Pupil Radius (mm):',
        style={'description_width': '120px'},
    )
    eccentricity_slider = widgets.FloatSlider(
        value=0.8, min=0.3, max=1.0, step=0.05,
        description='Eccentricity:',
        style={'description_width': '120px'},
    )
    show_sphere_checkbox = widgets.Checkbox(
        value=True, description='Show Sphere',
    )
    show_axes_checkbox = widgets.Checkbox(
        value=True, description='Show Axes',
    )
    
    def update_figure(*args):
        eye_params = EyeGeometryParameters(
            eye_diameter_mm=eye_radius_slider.value * 2,
            pupil_radius_mm=pupil_radius_slider.value,
            pupil_eccentricity=eccentricity_slider.value,
        )
        geometry = eye_params.to_reference_geometry()
        new_fig = create_eyeball_figure(
            geometry,
            show_sphere=show_sphere_checkbox.value,
            show_axes=show_axes_checkbox.value,
        )
        fig.data = []
        for trace in new_fig.data:
            fig.add_trace(trace)
        fig.layout = new_fig.layout
    
    eye_radius_slider.observe(update_figure, names='value')
    pupil_radius_slider.observe(update_figure, names='value')
    eccentricity_slider.observe(update_figure, names='value')
    show_sphere_checkbox.observe(update_figure, names='value')
    show_axes_checkbox.observe(update_figure, names='value')
    
    controls = widgets.VBox([
        widgets.HBox([eye_radius_slider, pupil_radius_slider, eccentricity_slider]),
        widgets.HBox([show_sphere_checkbox, show_axes_checkbox]),
    ])
    
    display(controls)
    return fig


def print_geometry_summary(geometry: EyeballReferenceGeometry) -> None:
    """Print a summary of all marker positions."""
    print(f"Eyeball Reference Geometry")
    print(f"=" * 50)
    print(f"Eye radius: {geometry.eye_radius_mm:.2f} mm")
    print(f"Pupil radius: {geometry.pupil_radius_mm:.2f} mm")
    print(f"Pupil eccentricity: {geometry.pupil_eccentricity:.2f}")
    print()
    print(f"Coordinate Frame:")
    print(f"  Origin: {geometry.coordinate_frame.origin_markers}")
    print(f"  X-axis: {geometry.coordinate_frame.x_axis}")
    print(f"  Y-axis: {geometry.coordinate_frame.y_axis}")
    print()
    print(f"Markers:")
    print(f"{'Name':<20} {'X (mm)':>10} {'Y (mm)':>10} {'Z (mm)':>10}")
    print(f"-" * 50)
    for name, pos in geometry.markers.items():
        print(f"{name:<20} {pos.x:>10.4f} {pos.y:>10.4f} {pos.z:>10.4f}")


def main_eyeball_viewer():
    """Main function to demonstrate the viewer."""
    # Create geometry from parameters
    eye_params = EyeGeometryParameters(
        eye_diameter_mm=7.0,
        pupil_radius_mm=0.5,
        pupil_eccentricity=0.8,
    )
    geometry = eye_params.to_reference_geometry()
    
    # Print summary
    print_geometry_summary(geometry)
    print()
    
    # Create and show figure
    fig = create_eyeball_figure(geometry)
    fig.show()


if __name__ == "__main__":
    main_eyeball_viewer()
