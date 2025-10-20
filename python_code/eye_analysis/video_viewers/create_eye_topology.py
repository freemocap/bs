"""
Create eye tracking overlay topology with configurable visualization options.
"""

import cv2
import numpy as np

from python_code.eye_analysis.data_processing.pupil_ellipse_fit import fit_ellipse_to_points
from python_code.eye_analysis.video_viewers.image_overlay_system import (
    OverlayTopology,
    ComputedPoint,
    PointElement,
    LineElement,
    CircleElement,
    CrosshairElement,
    TextElement,
    EllipseElement,
    PointStyle,
    LineStyle,
    TextStyle,
)


def create_full_eye_topology(
        *,
        width: int,
        height: int,
        show_cleaned: bool = True,
        show_raw: bool = False,
        show_dots: bool = True,
        show_ellipse: bool = True,
        show_snake: bool = False,
        n_snake_points: int = 50,
) -> OverlayTopology:
    """
    Create a complete eye tracking overlay topology.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        show_cleaned: Show cleaned/processed landmarks
        show_raw: Show raw landmark detections
        show_dots: Show individual landmark points
        show_ellipse: Show fitted ellipse to pupil
        show_snake: Show active contour snake points
        n_snake_points: Number of snake contour points

    Returns:
        Configured OverlayTopology instance
    """
    required_points: list[str] = []

    if show_cleaned:
        required_points.extend([
            "p1.cleaned", "p2.cleaned", "p3.cleaned", "p4.cleaned",
            "p5.cleaned", "p6.cleaned", "p7.cleaned", "p8.cleaned",
            "tear_duct.cleaned", "outer_eye.cleaned"
        ])

    if show_raw:
        required_points.extend([
            "p1.raw", "p2.raw", "p3.raw", "p4.raw",
            "p5.raw", "p6.raw", "p7.raw", "p8.raw",
            "tear_duct.raw", "outer_eye.raw"
        ])

    topology = OverlayTopology(
        name="full_eye_tracking",
        width=width,
        height=height,
        required_points=required_points
    )

    # === COMPUTED PUPIL CENTERS ===

    if show_raw:
        def compute_pupil_center_raw(points: dict[str, np.ndarray]) -> np.ndarray:
            pupil_points = [points[f"p{i}.raw"] for i in range(1, 9) if f"p{i}.raw" in points]
            if not pupil_points:
                return np.array([np.nan, np.nan])
            stacked = np.stack(arrays=pupil_points, axis=0)
            return np.nanmean(a=stacked, axis=0)

        topology.computed_points.append(
            ComputedPoint(
                name="pupil_center.raw",
                computation=compute_pupil_center_raw,
                description="Mean of raw pupil points"
            )
        )

    if show_cleaned:
        def compute_pupil_center_cleaned(points: dict[str, np.ndarray]) -> np.ndarray:
            pupil_points = [points[f"p{i}.cleaned"] for i in range(1, 9) if f"p{i}.cleaned" in points]
            if not pupil_points:
                return np.array([np.nan, np.nan])
            stacked = np.stack(arrays=pupil_points, axis=0)
            return np.nanmean(a=stacked, axis=0)

        topology.computed_points.append(
            ComputedPoint(
                name="pupil_center.cleaned",
                computation=compute_pupil_center_cleaned,
                description="Mean of cleaned pupil points"
            )
        )

    # === CONNECTION LINES (CLEANED ONLY) ===

    if show_cleaned:
        line_style = LineStyle(stroke='rgb(0, 200, 255)', stroke_width=2)

        # Pupil outline (closed loop)
        connections = [
            ("p1.cleaned", "p2.cleaned"), ("p2.cleaned", "p3.cleaned"),
            ("p3.cleaned", "p4.cleaned"), ("p4.cleaned", "p5.cleaned"),
            ("p5.cleaned", "p6.cleaned"), ("p6.cleaned", "p7.cleaned"),
            ("p7.cleaned", "p8.cleaned"), ("p8.cleaned", "p1.cleaned")
        ]
        for i, (pa, pb) in enumerate(connections):
            topology.add(
                element=LineElement(
                    name=f"pupil_connection_{i}",
                    point_a=pa,
                    point_b=pb,
                    style=line_style
                )
            )

        # Eye corner connections
        topology.add(
            element=LineElement(
                name="eye_span",
                point_a="tear_duct.cleaned",
                point_b="outer_eye.cleaned",
                style=line_style
            )
        )
        topology.add(
            element=LineElement(
                name="tear_to_pupil",
                point_a="tear_duct.cleaned",
                point_b="p1.cleaned",
                style=line_style
            )
        )

    # === LANDMARK POINTS ===

    # CLEANED POINTS (cyan/blue)
    if show_cleaned and show_dots:
        cleaned_point_style = PointStyle(radius=3, fill='rgb(0, 200, 255)')

        # Pupil points
        for i in range(1, 9):
            topology.add(
                element=PointElement(
                    name=f"pupil_point_{i}.cleaned",
                    point_name=f"p{i}.cleaned",
                    style=cleaned_point_style,
                    label=f"p{i}",
                    label_offset=(5, -5)
                )
            )

        # Eye corners
        cleaned_corner_style = PointStyle(radius=4, fill='rgb(0, 220, 255)')
        for name in ["tear_duct", "outer_eye"]:
            topology.add(
                element=PointElement(
                    name=f"{name}_point.cleaned",
                    point_name=f"{name}.cleaned",
                    style=cleaned_corner_style,
                    label=name.replace('_', ' ').title(),
                    label_offset=(5, -5)
                )
            )

    # RAW POINTS (red/orange)
    if show_raw and show_dots:
        raw_point_style = PointStyle(radius=2, fill='rgb(255, 100, 50)')

        # Pupil points
        for i in range(1, 9):
            topology.add(
                element=PointElement(
                    name=f"pupil_point_{i}.raw",
                    point_name=f"p{i}.raw",
                    style=raw_point_style,
                    label=f"p{i}" if not show_cleaned else None,
                    label_offset=(5, -5)
                )
            )

        # Eye corners
        raw_corner_style = PointStyle(radius=4, fill='rgb(255, 120, 0)')
        for name in ["tear_duct", "outer_eye"]:
            topology.add(
                element=PointElement(
                    name=f"{name}_point.raw",
                    point_name=f"{name}.raw",
                    style=raw_corner_style,
                    label=name.replace('_', ' ').title() if not show_cleaned else None,
                    label_offset=(5, -5)
                )
            )

    # === FITTED ELLIPSES ===

    if show_cleaned and show_ellipse:
        def compute_fitted_ellipse_cleaned(points: dict[str, np.ndarray]) -> np.ndarray:
            """Fit ellipse to cleaned pupil points and return params as [cx, cy, a, b, theta]."""
            pupil_points = np.array([points[f"p{i}.cleaned"] for i in range(1, 9)])

            try:
                ellipse_params = fit_ellipse_to_points(points=pupil_points)
                return ellipse_params.to_array()
            except (ValueError, cv2.error):
                # Return NaN if fitting fails
                return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        topology.computed_points.append(
            ComputedPoint(
                name="fitted_ellipse.cleaned",
                computation=compute_fitted_ellipse_cleaned,
                description="Fitted ellipse parameters for cleaned pupil points"
            )
        )

        topology.add(
            element=EllipseElement(
                name="pupil_ellipse.cleaned",
                params_point="fitted_ellipse.cleaned",
                n_points=100,
                style=LineStyle(
                    stroke='rgb(255, 0, 255)',  # Magenta
                    stroke_width=2,
                    opacity=0.8
                )
            )
        )

    # === SNAKE CONTOURS ===

    if show_snake and show_cleaned:
        # Snake points (green)
        snake_point_style = PointStyle(radius=4, fill='rgb(0, 255, 100)', opacity=0.9)

        for i in range(n_snake_points):
            topology.add(
                element=PointElement(
                    name=f"snake_point_{i}.cleaned",
                    point_name=f"snake_contour_cleaned_{i}",
                    style=snake_point_style
                )
            )

        # Connect snake points in a loop
        snake_line_style = LineStyle(stroke='rgb(0, 255, 100)', stroke_width=2, opacity=0.7)
        for i in range(n_snake_points):
            next_i = (i + 1) % n_snake_points
            topology.add(
                element=LineElement(
                    name=f"snake_connection_{i}.cleaned",
                    point_a=f"snake_contour_cleaned_{i}",
                    point_b=f"snake_contour_cleaned_{next_i}",
                    style=snake_line_style
                )
            )

    if show_snake and show_raw:
        # Snake points (yellow)
        snake_point_style = PointStyle(radius=4, fill='rgb(255, 255, 0)', opacity=0.9)

        for i in range(n_snake_points):
            topology.add(
                element=PointElement(
                    name=f"snake_point_{i}.raw",
                    point_name=f"snake_contour_raw_{i}",
                    style=snake_point_style
                )
            )

        # Connect snake points in a loop
        snake_line_style = LineStyle(stroke='rgb(255, 255, 0)', stroke_width=2, opacity=0.7)
        for i in range(n_snake_points):
            next_i = (i + 1) % n_snake_points
            topology.add(
                element=LineElement(
                    name=f"snake_connection_{i}.raw",
                    point_a=f"snake_contour_raw_{i}",
                    point_b=f"snake_contour_raw_{next_i}",
                    style=snake_line_style
                )
            )

    # === PUPIL CENTERS ===

    if show_cleaned:
        topology.add(
            element=CircleElement(
                name="pupil_center_circle.cleaned",
                center_point="pupil_center.cleaned",
                radius=5,
                style=PointStyle(fill='rgb(255, 250, 0)')
            )
        )

        topology.add(
            element=CrosshairElement(
                name="pupil_center_crosshair.cleaned",
                center_point="pupil_center.cleaned",
                size=10,
                style=LineStyle(stroke='rgb(255, 250, 0)', stroke_width=2)
            )
        )

    if show_raw:
        topology.add(
            element=CircleElement(
                name="pupil_center_circle.raw",
                center_point="pupil_center.raw",
                radius=5,
                style=PointStyle(fill='rgb(255, 200, 0)')
            )
        )

        topology.add(
            element=CrosshairElement(
                name="pupil_center_crosshair.raw",
                center_point="pupil_center.raw",
                size=10,
                style=LineStyle(stroke='rgb(255, 200, 0)', stroke_width=2)
            )
        )

    # === FRAME INFO (DYNAMIC TEXT) ===

    topology.computed_points.append(
        ComputedPoint(
            name="info_corner",
            computation=lambda pts: np.array([10.0, 25.0]),
            description="Top-left corner for info text"
        )
    )

    def format_frame_info(metadata: dict[str, object]) -> str:
        """Generate dynamic frame info text from metadata."""
        frame_idx = metadata.get('frame_idx', 0)
        total_frames = metadata.get('total_frames', 0)
        view_mode = metadata.get('view_mode', 'unknown')
        snake_status = metadata.get('snake_status', '')
        status_str = f" | {snake_status}" if snake_status else ""
        return f"Frame: {frame_idx}/{total_frames} | Mode: {view_mode}{status_str}"

    topology.add(
        element=TextElement(
            name="frame_info",
            point_name="info_corner",
            text=format_frame_info,  # Now using callable!
            offset=(0, 0),
            style=TextStyle(
                font_size=16,
                font_family='Consolas, monospace',
                fill='white',
                stroke='black',
                stroke_width=2,
                text_anchor='start'
            )
        )
    )

    return topology