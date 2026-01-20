"""
Enhanced Animated 3D Eyeball Viewer with Synced Timeseries
==========================================================

Shows:
- 3D animated eyeball with rotating pupil and ROTATING eye-frame basis vectors
- FIXED world/socket reference frame axes (stationary)
- Synchronized timeseries plots for adduction, elevation, torsion
- Animated vertical line tracking current frame across all plots

Key visualization elements:
- World frame axes (dotted, stationary): The eye-socket reference frame
- Eye frame axes (solid, rotating): Move with the eyeball orientation
  - +X (red) = Gaze direction
  - +Y (green) = Horizontal (medial for right eye)
  - +Z (blue) = Superior (up)

At rest position (quaternion = [1,0,0,0]):
- Pupil center on +X axis
- Pupil ellipse major axis along Y (equator)
- Pupil ellipse minor axis along Z (prime meridian)
"""

from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.typing import NDArray

from python_code.ferret_gaze.eye_kinematics.eye_kinematics_model import (
    FerretEyeKinematics,
    NUM_PUPIL_POINTS,
)


def generate_sphere_mesh(
    radius: float,
    resolution: int = 25,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate sphere surface mesh for visualization."""
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution * 2)
    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return x, y, z


def quaternion_to_rotation_matrix(q_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    The rotation matrix R rotates vectors: v_rotated = R @ v
    """
    w, x, y, z = q_wxyz

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

    return R


def compute_rotated_axes(
    q_wxyz: NDArray[np.float64],
    axis_length: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the eye-frame basis vectors rotated by the given quaternion.

    Returns:
        x_axis: (2, 3) array of [origin, tip] for +X axis (gaze)
        y_axis: (2, 3) array of [origin, tip] for +Y axis
        z_axis: (2, 3) array of [origin, tip] for +Z axis
    """
    R = quaternion_to_rotation_matrix(q_wxyz)

    origin = np.array([0.0, 0.0, 0.0])

    # Rest basis vectors
    x_rest = np.array([axis_length, 0.0, 0.0])
    y_rest = np.array([0.0, axis_length, 0.0])
    z_rest = np.array([0.0, 0.0, axis_length])

    # Rotated basis vectors
    x_rot = R @ x_rest
    y_rot = R @ y_rest
    z_rot = R @ z_rest

    return (
        np.array([origin, x_rot]),
        np.array([origin, y_rot]),
        np.array([origin, z_rot]),
    )


def create_animated_eye_figure(
    kinematics: FerretEyeKinematics,
    frame_step: int = 1,
    show_sphere: bool = True,
    show_socket_landmarks: bool = True,
    title: str | None = None,
    estimated_torsion: NDArray[np.float64] | None = None,
) -> go.Figure:
    """
    Create animated 3D eye figure with synchronized timeseries plots.

    The animation shows:
    - 3D eyeball with rotating pupil and eye-frame basis vectors
    - Fixed world-frame basis vectors (socket reference)
    - Three timeseries: Adduction, Elevation, Torsion
    - Animated vertical line tracking current frame across all plots

    Visual elements:
    - World axes (dotted): Fixed reference frame
    - Eye axes (solid): Rotate with the eyeball
      - +X (red) = Gaze direction
      - +Y (green) = Medial/lateral
      - +Z (blue) = Superior

    Args:
        kinematics: FerretEyeKinematics with eye movement data
        frame_step: Step between animation frames
        show_sphere: Whether to show translucent eyeball
        show_socket_landmarks: Whether to show tear_duct/outer_eye
        title: Plot title
        estimated_torsion: Optional separately-estimated torsion to overlay

    Returns:
        Plotly Figure with animation controls
    """
    if title is None:
        title = f"Eye Movement: {kinematics.name} ({kinematics.eye_side} eye)"

    n_frames = kinematics.n_frames
    frame_indices = list(range(0, n_frames, frame_step))

    # Get geometry parameters
    R = kinematics.eyeball.reference_geometry.markers["pupil_center"].x
    axis_length = R * 1.5

    # Get trajectories
    timestamps = kinematics.timestamps - kinematics.timestamps[0]
    pupil_center = kinematics.pupil_center_trajectory
    pupil_points = kinematics.pupil_points_trajectories
    quaternions = kinematics.quaternions_wxyz

    tear_duct = kinematics.tear_duct_mm
    outer_eye = kinematics.outer_eye_mm

    # Get anatomical angles in degrees
    adduction_deg = np.degrees(kinematics.adduction_angle.values)
    elevation_deg = np.degrees(kinematics.elevation_angle.values)
    torsion_deg = np.degrees(kinematics.torsion_angle.values)

    # Create figure with subplots
    # Layout: 3D scene on left (larger), timeseries stacked on right
    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[0.33, 0.33, 0.34],
        specs=[
            [{"type": "scene", "rowspan": 3}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=(
            "", "Adduction (+) / Abduction (-)",
            "Elevation (+) / Depression (-)",
            "Extorsion (+) / Intorsion (-)",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    # =========================================================================
    # TIMESERIES PLOTS (static traces, animated vertical line)
    # =========================================================================

    # Adduction timeseries
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=adduction_deg,
            mode='lines+markers', name='Adduction',
            line=dict(color='blue', width=1.5),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="lightgray", row=1, col=2)

    # Elevation timeseries
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=elevation_deg,
            mode='lines+markers', name='Elevation',
            line=dict(color='green', width=1.5),
            showlegend=False,
        ),
        row=2, col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="lightgray", row=2, col=2)

    # Torsion timeseries
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=torsion_deg,
            mode='lines+markers', name='Torsion (quat)',
            line=dict(color='red', width=1.5),
            showlegend=False,
        ),
        row=3, col=2,
    )

    # Optional: overlay estimated torsion from pupil ellipse
    if estimated_torsion is not None:
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=np.degrees(estimated_torsion),
                mode='lines+markers', name='Torsion (ellipse)',
                line=dict(color='orange', width=1.5, dash='dot'),
                showlegend=True,
            ),
            row=3, col=2,
        )

    fig.add_hline(y=0, line_dash="dash", line_color="lightgray", row=3, col=2)

    # Vertical time indicator lines (will be animated)
    idx0 = frame_indices[0]
    t0 = timestamps[idx0]

    # Y ranges for each subplot
    add_range = [np.min(adduction_deg) - 2, np.max(adduction_deg) + 2]
    elev_range = [np.min(elevation_deg) - 2, np.max(elevation_deg) + 2]
    tors_range = [np.min(torsion_deg) - 2, np.max(torsion_deg) + 2]

    # Time indicator traces (one for each timeseries)
    fig.add_trace(
        go.Scatter(
            x=[t0, t0], y=add_range,
            mode='lines', name='time',
            line=dict(color='black', width=2),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[t0, t0], y=elev_range,
            mode='lines', name='time',
            line=dict(color='black', width=2),
            showlegend=False,
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[t0, t0], y=tors_range,
            mode='lines', name='time',
            line=dict(color='black', width=2),
            showlegend=False,
        ),
        row=3, col=2,
    )

    # Current value markers on timeseries
    fig.add_trace(
        go.Scatter(
            x=[t0], y=[adduction_deg[idx0]],
            mode='markers', name='current',
            marker=dict(color='blue', size=10),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[t0], y=[elevation_deg[idx0]],
            mode='markers', name='current',
            marker=dict(color='green', size=10),
            showlegend=False,
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[t0], y=[torsion_deg[idx0]],
            mode='markers', name='current',
            marker=dict(color='red', size=10),
            showlegend=False,
        ),
        row=3, col=2,
    )

    # =========================================================================
    # 3D SCENE
    # =========================================================================

    # Sphere mesh (static)
    if show_sphere:
        x_sphere, y_sphere, z_sphere = generate_sphere_mesh(R, resolution=20)
        fig.add_trace(
            go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.15,
                colorscale=[[0, 'rgb(230, 230, 245)'], [1, 'rgb(210, 210, 235)']],
                showscale=False,
                hoverinfo='skip',
                name='Eyeball',
            ),
            row=1, col=1,
        )

    # --- WORLD/SOCKET FRAME AXES (FIXED, dotted) ---
    # These represent the eye-socket reference frame and do NOT rotate
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='darkred', width=3, dash='dot'),
        name='World +X', showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='darkgreen', width=3, dash='dot'),
        name='World +Y', showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines', line=dict(color='darkblue', width=3, dash='dot'),
        name='World +Z', showlegend=True,
    ), row=1, col=1)

    # --- EYE FRAME AXES (ROTATING, solid) ---
    # These rotate with the eyeball orientation
    # Initial position
    x_ax, y_ax, z_ax = compute_rotated_axes(quaternions[idx0], axis_length)

    # +X axis (Gaze direction) - red solid
    fig.add_trace(go.Scatter3d(
        x=x_ax[:, 0], y=x_ax[:, 1], z=x_ax[:, 2],
        mode='lines', line=dict(color='red', width=5),
        name='Eye +X (Gaze)', showlegend=True,
    ), row=1, col=1)

    # +Y axis - green solid
    fig.add_trace(go.Scatter3d(
        x=y_ax[:, 0], y=y_ax[:, 1], z=y_ax[:, 2],
        mode='lines', line=dict(color='limegreen', width=5),
        name='Eye +Y', showlegend=True,
    ), row=1, col=1)

    # +Z axis - blue solid
    fig.add_trace(go.Scatter3d(
        x=z_ax[:, 0], y=z_ax[:, 1], z=z_ax[:, 2],
        mode='lines', line=dict(color='dodgerblue', width=5),
        name='Eye +Z', showlegend=True,
    ), row=1, col=1)

    # Eye center (static at origin)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=6, color='purple', symbol='diamond'),
        name='Eye Center',
        showlegend=False,
    ), row=1, col=1)

    # Pupil center (animated)
    fig.add_trace(go.Scatter3d(
        x=[pupil_center[idx0, 0]],
        y=[pupil_center[idx0, 1]],
        z=[pupil_center[idx0, 2]],
        mode='markers',
        marker=dict(size=10, color='black'),
        name='Pupil Center',
        showlegend=False,
    ), row=1, col=1)

    # Pupil boundary (animated) - closed loop
    pupil_loop_x = list(pupil_points[idx0, :, 0]) + [pupil_points[idx0, 0, 0]]
    pupil_loop_y = list(pupil_points[idx0, :, 1]) + [pupil_points[idx0, 0, 1]]
    pupil_loop_z = list(pupil_points[idx0, :, 2]) + [pupil_points[idx0, 0, 2]]
    fig.add_trace(go.Scatter3d(
        x=pupil_loop_x, y=pupil_loop_y, z=pupil_loop_z,
        mode='lines',
        line=dict(color='darkblue', width=4),
        name='Pupil Boundary',
        showlegend=False,
    ), row=1, col=1)

    # Socket landmarks (animated - slight movement from noise)
    if show_socket_landmarks:
        fig.add_trace(go.Scatter3d(
            x=[tear_duct[idx0, 0]],
            y=[tear_duct[idx0, 1]],
            z=[tear_duct[idx0, 2]],
            mode='markers',
            marker=dict(size=8, color='orange', symbol='square'),
            name='Tear Duct (socket)',
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=[outer_eye[idx0, 0]],
            y=[outer_eye[idx0, 1]],
            z=[outer_eye[idx0, 2]],
            mode='markers',
            marker=dict(size=8, color='darkorange', symbol='square'),
            name='Outer Eye (socket)',
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=[tear_duct[idx0, 0], outer_eye[idx0, 0]],
            y=[tear_duct[idx0, 1], outer_eye[idx0, 1]],
            z=[tear_duct[idx0, 2], outer_eye[idx0, 2]],
            mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
            name='Eye Opening',
            showlegend=False,
        ), row=1, col=1)

    # =========================================================================
    # BUILD ANIMATION FRAMES
    # =========================================================================

    frames = []

    for frame_num, idx in enumerate(frame_indices):
        t = timestamps[idx]
        q = quaternions[idx]

        frame_data = []

        # --- Timeseries traces (static data, just re-add) ---
        frame_data.append(go.Scatter(x=timestamps, y=adduction_deg))
        frame_data.append(go.Scatter(x=timestamps, y=elevation_deg))
        frame_data.append(go.Scatter(x=timestamps, y=torsion_deg))

        if estimated_torsion is not None:
            frame_data.append(go.Scatter(x=timestamps, y=np.degrees(estimated_torsion)))

        # Time indicator lines (animated)
        frame_data.append(go.Scatter(x=[t, t], y=add_range))
        frame_data.append(go.Scatter(x=[t, t], y=elev_range))
        frame_data.append(go.Scatter(x=[t, t], y=tors_range))

        # Current value markers (animated)
        frame_data.append(go.Scatter(x=[t], y=[adduction_deg[idx]]))
        frame_data.append(go.Scatter(x=[t], y=[elevation_deg[idx]]))
        frame_data.append(go.Scatter(x=[t], y=[torsion_deg[idx]]))

        # --- 3D traces ---

        # Sphere (static)
        if show_sphere:
            frame_data.append(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere))

        # World frame axes (STATIC - don't change)
        frame_data.append(go.Scatter3d(x=[0, axis_length], y=[0, 0], z=[0, 0]))
        frame_data.append(go.Scatter3d(x=[0, 0], y=[0, axis_length], z=[0, 0]))
        frame_data.append(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_length]))

        # Eye frame axes (ANIMATED - rotate with quaternion)
        x_ax, y_ax, z_ax = compute_rotated_axes(q, axis_length)
        frame_data.append(go.Scatter3d(x=x_ax[:, 0], y=x_ax[:, 1], z=x_ax[:, 2]))
        frame_data.append(go.Scatter3d(x=y_ax[:, 0], y=y_ax[:, 1], z=y_ax[:, 2]))
        frame_data.append(go.Scatter3d(x=z_ax[:, 0], y=z_ax[:, 1], z=z_ax[:, 2]))

        # Eye center (static)
        frame_data.append(go.Scatter3d(x=[0], y=[0], z=[0]))

        # Pupil center (animated)
        frame_data.append(go.Scatter3d(
            x=[pupil_center[idx, 0]],
            y=[pupil_center[idx, 1]],
            z=[pupil_center[idx, 2]],
        ))

        # Pupil boundary (animated)
        pupil_loop_x = list(pupil_points[idx, :, 0]) + [pupil_points[idx, 0, 0]]
        pupil_loop_y = list(pupil_points[idx, :, 1]) + [pupil_points[idx, 0, 1]]
        pupil_loop_z = list(pupil_points[idx, :, 2]) + [pupil_points[idx, 0, 2]]
        frame_data.append(go.Scatter3d(
            x=pupil_loop_x, y=pupil_loop_y, z=pupil_loop_z,
        ))

        # Socket landmarks (animated - slight noise)
        if show_socket_landmarks:
            frame_data.append(go.Scatter3d(
                x=[tear_duct[idx, 0]],
                y=[tear_duct[idx, 1]],
                z=[tear_duct[idx, 2]],
            ))
            frame_data.append(go.Scatter3d(
                x=[outer_eye[idx, 0]],
                y=[outer_eye[idx, 1]],
                z=[outer_eye[idx, 2]],
            ))
            frame_data.append(go.Scatter3d(
                x=[tear_duct[idx, 0], outer_eye[idx, 0]],
                y=[tear_duct[idx, 1], outer_eye[idx, 1]],
                z=[tear_duct[idx, 2], outer_eye[idx, 2]],
            ))

        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_num),
        ))

    fig.frames = frames

    # =========================================================================
    # LAYOUT
    # =========================================================================

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=700,
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (mm)', range=[-R*2.5, R*2.5]),
            yaxis=dict(title='Y (mm)', range=[-R*2.5, R*2.5]),
            zaxis=dict(title='Z (mm)', range=[-R*2.5, R*2.5]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0, r=0, t=60, b=10),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0, y=0,
                xanchor="left", yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top", xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Frame: ",
                    visible=True,
                    xanchor="right",
                ),
                transition=dict(duration=0),
                pad=dict(b=10, t=50),
                len=0.45, x=0.0, y=0,
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                            ),
                        ],
                        label=str(frame_indices[i]),
                        method="animate",
                    )
                    for i in range(len(frame_indices))
                ],
            ),
        ],
    )

    # Update timeseries axes
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="°", row=1, col=2, range=add_range)
    fig.update_yaxes(title_text="°", row=2, col=2, range=elev_range)
    fig.update_yaxes(title_text="°", row=3, col=2, range=tors_range)

    return fig


# =============================================================================
# STATIC REFERENCE FIGURE
# =============================================================================

def create_static_reference_figure(
    eye_radius_mm: float = 3.5,
    pupil_radius_mm: float = 0.5,
    pupil_eccentricity: float = 0.8,
    eye_side: Literal["left", "right"] = "right",
    title: str = "Eyeball Reference Geometry (Rest Position)",
) -> go.Figure:
    """
    Create a static 3D figure showing the eyeball at rest position.

    At rest (quaternion = [1, 0, 0, 0]):
    - Pupil center at [+R, 0, 0] on +X axis
    - Pupil ellipse major axis along Y (equator)
    - Pupil ellipse minor axis along Z (prime meridian)
    """
    fig = go.Figure()
    R = eye_radius_mm
    axis_length = R * 1.5

    # Sphere surface
    x, y, z = generate_sphere_mesh(R, resolution=25)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        opacity=0.2,
        colorscale=[[0, 'rgb(220, 220, 240)'], [1, 'rgb(200, 200, 230)']],
        showscale=False,
        hoverinfo='skip',
        name='Eyeball',
    ))

    # World frame axes (dotted) - these are the socket reference
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='darkred', width=4, dash='dot'),
        name='World +X',
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='darkgreen', width=4, dash='dot'),
        name='World +Y',
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines', line=dict(color='darkblue', width=4, dash='dot'),
        name='World +Z',
    ))

    # Eye frame axes (solid) - at rest, same as world frame
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=6),
        name='Eye +X (Gaze)',
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='limegreen', width=6),
        name='Eye +Y',
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines', line=dict(color='dodgerblue', width=6),
        name='Eye +Z',
    ))

    # Eye center
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=6, color='purple', symbol='diamond'),
        name='Eye Center',
    ))

    # Pupil center (at +X on sphere surface)
    fig.add_trace(go.Scatter3d(
        x=[R], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='black'),
        name='Pupil Center',
    ))

    # Pupil boundary points
    # At rest: major axis along Y (equator), minor axis along Z (prime meridian)
    a = pupil_radius_mm  # Semi-major (along Y)
    b = pupil_radius_mm * pupil_eccentricity  # Semi-minor (along Z)
    pupil_x, pupil_y, pupil_z = [], [], []

    for i in range(NUM_PUPIL_POINTS + 1):  # +1 to close loop
        phi = 2 * np.pi * (i % NUM_PUPIL_POINTS) / NUM_PUPIL_POINTS
        # Ellipse in tangent plane at pupil center
        # Major axis = Y direction, Minor axis = Z direction
        y_tangent = a * np.cos(phi)
        z_tangent = b * np.sin(phi)
        tangent_point = np.array([R, y_tangent, z_tangent])
        # Project onto sphere
        direction = tangent_point / np.linalg.norm(tangent_point)
        sphere_point = R * direction
        pupil_x.append(sphere_point[0])
        pupil_y.append(sphere_point[1])
        pupil_z.append(sphere_point[2])

    fig.add_trace(go.Scatter3d(
        x=pupil_x, y=pupil_y, z=pupil_z,
        mode='lines',
        line=dict(color='darkblue', width=4),
        name='Pupil Boundary',
    ))

    # Add labels for anatomy
    y_label = "Medial" if eye_side == "right" else "Lateral"
    neg_y_label = "Lateral" if eye_side == "right" else "Medial"

    # Layout
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>+Y = {y_label}, -Y = {neg_y_label}</sub>", x=0.5),
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (Anterior)', range=[-R*2, R*2]),
            yaxis=dict(title='Y (Subject Left)', range=[-R*2, R*2]),
            zaxis=dict(title='Z (Superior)', range=[-R*2, R*2]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    return fig


# =============================================================================
# GAZE TIMESERIES FIGURE
# =============================================================================

def create_gaze_timeseries_figure(
    kinematics: FerretEyeKinematics,
    title: str | None = None,
) -> go.Figure:
    """
    Create a figure showing gaze angles over time.
    """
    if title is None:
        title = f"Gaze Angles: {kinematics.name}"

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Horizontal (Adduction/Abduction)",
            "Vertical (Elevation/Depression)",
            "Torsion (Extorsion/Intorsion)",
        ),
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    t = kinematics.timestamps

    # Adduction angle
    adduction = kinematics.adduction_angle
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.degrees(adduction.values),
            mode='lines',
            name='Adduction (+) / Abduction (-)',
            line=dict(color='blue', width=1.5),
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # Elevation angle
    elevation = kinematics.elevation_angle
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.degrees(elevation.values),
            mode='lines',
            name='Elevation (+) / Depression (-)',
            line=dict(color='green', width=1.5),
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Torsion angle
    torsion = kinematics.torsion_angle
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.degrees(torsion.values),
            mode='lines+markers',
            name='Extorsion (+) / Intorsion (-)',
            line=dict(color='red', width=1.5),
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=550,
        showlegend=True,
        legend=dict(x=1.02, y=1),
    )

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    for row in range(1, 4):
        fig.update_yaxes(title_text="Angle (°)", row=row, col=1)

    return fig