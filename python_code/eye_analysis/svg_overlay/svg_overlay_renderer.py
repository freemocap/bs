from pathlib import Path
from typing import Any

import cv2
import drawsvg as dw
import numpy as np
from PIL import ImageDraw, ImageFont, Image

from python_code.eye_analysis.svg_overlay import SVGTopology, PointElement, LineElement, CircleElement, \
    CrosshairElement, TextElement, EllipseElement


class SVGOverlayRenderer:
    """Renders SVG overlays from topology + data, and composites with images."""

    def __init__(self, *, topology: SVGTopology):
        """Initialize renderer with a topology.

        Args:
            topology: The SVG topology defining what to render
        """
        self.topology: SVGTopology = topology

    def render_svg(
            self,
            *,
            points: dict[str, np.ndarray],
            metadata: dict[str, Any] | None = None
    ) -> dw.Drawing:
        """Render topology to SVG using provided point coordinates.

        Args:
            points: Dictionary mapping point names to (x, y) coordinates
            metadata: Optional metadata for dynamic text/styling

        Returns:
            DrawSVG Drawing object
        """
        if metadata is None:
            metadata = {}

        # Validate required points
        missing_points = set(self.topology.required_points) - set(points.keys())
        if missing_points:
            raise ValueError(f"Missing required points: {missing_points}")

        # Compute derived points
        all_points = points.copy()
        for computed in self.topology.computed_points:
            try:
                all_points[computed.name] = computed.computation(all_points)
            except Exception as e:
                raise ValueError(
                    f"Failed to compute point '{computed.name}': {e}"
                )

        # Create drawing
        d = dw.Drawing(
            width=self.topology.width,
            height=self.topology.height
        )

        # Add CSS styles if provided
        if self.topology.css_styles:
            d.append(dw.Style(self.topology.css_styles))

        # Add background if not transparent
        if self.topology.background_opacity > 0:
            d.append(dw.Rectangle(
                x=0, y=0,
                width=self.topology.width,
                height=self.topology.height,
                fill='black',
                opacity=self.topology.background_opacity
            ))

        # Render elements
        for element in self.topology.elements:
            if not element.visible:
                continue

            if isinstance(element, PointElement):
                self._render_point(
                    drawing=d,
                    element=element,
                    points=all_points,
                    metadata=metadata
                )
            elif isinstance(element, LineElement):
                self._render_line(
                    drawing=d,
                    element=element,
                    points=all_points
                )
            elif isinstance(element, CircleElement):
                self._render_circle(
                    drawing=d,
                    element=element,
                    points=all_points
                )
            elif isinstance(element, CrosshairElement):
                self._render_crosshair(
                    drawing=d,
                    element=element,
                    points=all_points
                )
            elif isinstance(element, TextElement):
                self._render_text(
                    drawing=d,
                    element=element,
                    points=all_points,
                    metadata=metadata
                )
            elif isinstance(element, EllipseElement):
                self._render_ellipse(
                    drawing=d,
                    element=element,
                    points=all_points
                )

        return d

    def _is_valid_point(self, *, point: np.ndarray) -> bool:
        """Check if point has valid (non-NaN) coordinates."""
        return not np.isnan(point).any()

    def _render_point(
            self,
            *,
            drawing: dw.Drawing,
            element: PointElement,
            points: dict[str, np.ndarray],
            metadata: dict[str, Any]
    ) -> None:
        """Render a point element."""
        point = points.get(element.point_name)
        if point is None or not self._is_valid_point(point=point):
            return

        x, y = float(point[0]), float(point[1])

        # Draw circle
        circle = dw.Circle(
            cx=x, cy=y,
            r=element.style.radius,
            fill=element.style.fill,
            opacity=element.style.opacity
        )
        if element.style.stroke:
            circle.args['stroke'] = element.style.stroke
            circle.args['stroke_width'] = element.style.stroke_width

        drawing.append(circle)

        # Draw label if present
        if element.label:
            label_x = x + element.label_offset[0]
            label_y = y + element.label_offset[1]

            text = dw.Text(
                text=element.label,
                x=label_x,
                y=label_y,
                font_size=element.label_style.font_size,
                font_family=element.label_style.font_family,
                fill=element.label_style.fill,
                font_weight=element.label_style.font_weight,
                text_anchor=element.label_style.text_anchor
            )
            if element.label_style.stroke:
                text.args['stroke'] = element.label_style.stroke
                text.args['stroke_width'] = element.label_style.stroke_width
                text.args['paint_order'] = 'stroke fill'

            drawing.append(text)

    def _render_line(
            self,
            *,
            drawing: dw.Drawing,
            element: LineElement,
            points: dict[str, np.ndarray]
    ) -> None:
        """Render a line element."""
        pt_a = points.get(element.point_a)
        pt_b = points.get(element.point_b)

        if (pt_a is None or pt_b is None or
                not self._is_valid_point(point=pt_a) or
                not self._is_valid_point(point=pt_b)):
            return

        line = dw.Line(
            float(pt_a[0]), float(pt_a[1]),
            float(pt_b[0]), float(pt_b[1]),
            stroke=element.style.stroke,
            stroke_width=element.style.stroke_width,
            opacity=element.style.opacity
        )
        if element.style.stroke_dasharray:
            line.args['stroke_dasharray'] = element.style.stroke_dasharray

        drawing.append(line)

    def _render_circle(
            self,
            *,
            drawing: dw.Drawing,
            element: CircleElement,
            points: dict[str, np.ndarray]
    ) -> None:
        """Render a circle element."""
        center = points.get(element.center_point)
        if center is None or not self._is_valid_point(point=center):
            return

        circle = dw.Circle(
            cx=float(center[0]),
            cy=float(center[1]),
            r=element.radius,
            fill=element.style.fill,
            opacity=element.style.opacity
        )
        if element.style.stroke:
            circle.args['stroke'] = element.style.stroke
            circle.args['stroke_width'] = element.style.stroke_width

        drawing.append(circle)

    def _render_crosshair(
            self,
            *,
            drawing: dw.Drawing,
            element: CrosshairElement,
            points: dict[str, np.ndarray]
    ) -> None:
        """Render a crosshair element."""
        center = points.get(element.center_point)
        if center is None or not self._is_valid_point(point=center):
            return

        cx, cy = float(center[0]), float(center[1])
        size = element.size

        # Horizontal line
        drawing.append(dw.Line(
            cx - size, cy,
            cx + size, cy,
            stroke=element.style.stroke,
            stroke_width=element.style.stroke_width,
            opacity=element.style.opacity
        ))

        # Vertical line
        drawing.append(dw.Line(
            cx, cy - size,
            cx, cy + size,
            stroke=element.style.stroke,
            stroke_width=element.style.stroke_width,
            opacity=element.style.opacity
        ))

    def _render_text(
            self,
            *,
            drawing: dw.Drawing,
            element: TextElement,
            points: dict[str, np.ndarray],
            metadata: dict[str, Any]
    ) -> None:
        """Render a text element."""
        point = points.get(element.point_name)
        if point is None or not self._is_valid_point(point=point):
            return

        x = float(point[0]) + element.offset[0]
        y = float(point[1]) + element.offset[1]

        # Get text content (static or dynamic)
        if callable(element.text):
            text_content = element.text(metadata)
        else:
            text_content = element.text

        text = dw.Text(
            text=text_content,
            x=x, y=y,
            font_size=element.style.font_size,
            font_family=element.style.font_family,
            fill=element.style.fill,
            font_weight=element.style.font_weight,
            text_anchor=element.style.text_anchor
        )
        if element.style.stroke:
            text.args['stroke'] = element.style.stroke
            text.args['stroke_width'] = element.style.stroke_width
            text.args['paint_order'] = 'stroke fill'

        drawing.append(text)

    def _render_ellipse(
            self,
            *,
            drawing: dw.Drawing,
            element: EllipseElement,
            points: dict[str, np.ndarray]
    ) -> None:
        """Render an ellipse element."""
        params_array = points.get(element.params_point)
        if params_array is None or len(params_array) != 5:
            return

        # Check if parameters are valid
        if np.isnan(params_array).any():
            return

        # Extract ellipse parameters: [cx, cy, a, b, theta]
        cx, cy, semi_major, semi_minor, rotation = params_array

        # Generate ellipse points
        theta = np.linspace(start=0, stop=2 * np.pi, num=element.n_points)
        x_local = semi_major * np.cos(theta)
        y_local = semi_minor * np.sin(theta)

        # Rotate and translate
        cos_t = np.cos(rotation)
        sin_t = np.sin(rotation)
        x = cx + x_local * cos_t - y_local * sin_t
        y = cy + x_local * sin_t + y_local * cos_t

        # Draw as polyline (closed path)
        path_data = f"M {x[0]},{y[0]} "
        for xi, yi in zip(x[1:], y[1:]):
            path_data += f"L {xi},{yi} "
        path_data += "Z"  # Close path

        path = dw.Path(
            d=path_data,
            stroke=element.style.stroke,
            stroke_width=element.style.stroke_width,
            fill='none',
            opacity=element.style.opacity
        )
        if element.style.stroke_dasharray:
            path.args['stroke_dasharray'] = element.style.stroke_dasharray

        drawing.append(path)

    def _parse_rgb(self, *, color: str) -> tuple[int, int, int, int]:
        """Parse RGB color string to RGBA tuple.

        Args:
            color: Color string like 'rgb(255, 0, 0)' or named color

        Returns:
            RGBA tuple (r, g, b, a)
        """
        color = color.strip().lower()

        # Handle rgb() format
        if color.startswith('rgb(') and color.endswith(')'):
            values = color[4:-1].split(',')
            r, g, b = [int(v.strip()) for v in values]
            return (r, g, b, 255)

        # Handle named colors
        color_map = {
            'red': (255, 0, 0, 255),
            'green': (0, 255, 0, 255),
            'blue': (0, 0, 255, 255),
            'yellow': (255, 255, 0, 255),
            'lime': (0, 255, 0, 255),
            'white': (255, 255, 255, 255),
            'black': (0, 0, 0, 255),
            'cyan': (0, 255, 255, 255),
            'magenta': (255, 0, 255, 255),
            #TODO - add matplotlib/xkcd colors
        }

        return color_map.get(color, (255, 255, 255, 255))

    def _render_pil_point(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            element: PointElement,
            points: dict[str, np.ndarray],
            metadata: dict[str, Any]
    ) -> None:
        """Render a point element using Pillow."""
        point = points.get(element.point_name)
        if point is None or not self._is_valid_point(point=point):
            return

        x, y = float(point[0]), float(point[1])
        r = element.style.radius

        # Parse colors
        fill_color = self._parse_rgb(color=element.style.fill)

        # Apply opacity
        fill_color = (*fill_color[:3], int(fill_color[3] * element.style.opacity))

        # Draw circle
        draw.ellipse(
            xy=[(x - r, y - r), (x + r, y + r)],
            fill=fill_color,
            outline=self._parse_rgb(color=element.style.stroke) if element.style.stroke else None,
            width=element.style.stroke_width or 0
        )

        # Draw label if present
        if element.label:
            label_x = x + element.label_offset[0]
            label_y = y + element.label_offset[1]

            # Parse label colors
            label_fill = self._parse_rgb(color=element.label_style.fill)

            try:
                font = ImageFont.truetype(font='arial.ttf', size=element.label_style.font_size)
            except:
                font = ImageFont.load_default()

            # Draw stroke if present
            if element.label_style.stroke and element.label_style.stroke_width:
                stroke_color = self._parse_rgb(color=element.label_style.stroke)
                for dx in range(-element.label_style.stroke_width, element.label_style.stroke_width + 1):
                    for dy in range(-element.label_style.stroke_width, element.label_style.stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text(
                                xy=(label_x + dx, label_y + dy),
                                text=element.label,
                                fill=stroke_color,
                                font=font
                            )

            # Draw text
            draw.text(
                xy=(label_x, label_y),
                text=element.label,
                fill=label_fill,
                font=font
            )

    def _render_pil_line(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            element: LineElement,
            points: dict[str, np.ndarray]
    ) -> None:
        """Render a line element using Pillow."""
        pt_a = points.get(element.point_a)
        pt_b = points.get(element.point_b)

        if (pt_a is None or pt_b is None or
                not self._is_valid_point(point=pt_a) or
                not self._is_valid_point(point=pt_b)):
            return

        stroke_color = self._parse_rgb(color=element.style.stroke)
        stroke_color = (*stroke_color[:3], int(stroke_color[3] * element.style.opacity))

        draw.line(
            xy=[
                (float(pt_a[0]), float(pt_a[1])),
                (float(pt_b[0]), float(pt_b[1]))
            ],
            fill=stroke_color,
            width=element.style.stroke_width
        )

    def _render_pil_circle(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            element: CircleElement,
            points: dict[str, np.ndarray]
    ) -> None:
        """Render a circle element using Pillow."""
        center = points.get(element.center_point)
        if center is None or not self._is_valid_point(point=center):
            return

        cx, cy = float(center[0]), float(center[1])
        r = element.radius

        fill_color = self._parse_rgb(color=element.style.fill)
        fill_color = (*fill_color[:3], int(fill_color[3] * element.style.opacity))

        draw.ellipse(
            xy=[(cx - r, cy - r), (cx + r, cy + r)],
            fill=fill_color,
            outline=self._parse_rgb(color=element.style.stroke) if element.style.stroke else None,
            width=element.style.stroke_width or 0
        )

    def _render_pil_crosshair(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            element: CrosshairElement,
            points: dict[str, np.ndarray]
    ) -> None:
        """Render a crosshair element using Pillow."""
        center = points.get(element.center_point)
        if center is None or not self._is_valid_point(point=center):
            return

        cx, cy = float(center[0]), float(center[1])
        size = element.size

        stroke_color = self._parse_rgb(color=element.style.stroke)
        stroke_color = (*stroke_color[:3], int(stroke_color[3] * element.style.opacity))

        # Horizontal line
        draw.line(
            xy=[(cx - size, cy), (cx + size, cy)],
            fill=stroke_color,
            width=element.style.stroke_width
        )

        # Vertical line
        draw.line(
            xy=[(cx, cy - size), (cx, cy + size)],
            fill=stroke_color,
            width=element.style.stroke_width
        )

    def _render_pil_text(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            element: TextElement,
            points: dict[str, np.ndarray],
            metadata: dict[str, Any]
    ) -> None:
        """Render a text element using Pillow."""
        point = points.get(element.point_name)
        if point is None or not self._is_valid_point(point=point):
            return

        x = float(point[0]) + element.offset[0]
        y = float(point[1]) + element.offset[1]

        # Get text content (static or dynamic)
        if callable(element.text):
            text_content = element.text(metadata)
        else:
            text_content = element.text

        # Parse colors
        fill_color = self._parse_rgb(color=element.style.fill)

        # Load font
        try:
            font = ImageFont.truetype(font='arial.ttf', size=element.style.font_size)
        except:
            font = ImageFont.load_default()

        # Draw stroke if present
        if element.style.stroke and element.style.stroke_width:
            stroke_color = self._parse_rgb(color=element.style.stroke)
            for dx in range(-element.style.stroke_width, element.style.stroke_width + 1):
                for dy in range(-element.style.stroke_width, element.style.stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text(
                            xy=(x + dx, y + dy),
                            text=text_content,
                            fill=stroke_color,
                            font=font
                        )

        # Draw text
        draw.text(
            xy=(x, y),
            text=text_content,
            fill=fill_color,
            font=font
        )

    def _render_pil_ellipse(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            element: EllipseElement,
            points: dict[str, np.ndarray]
    ) -> None:
        """Render an ellipse element using Pillow."""
        params_array = points.get(element.params_point)
        if params_array is None or len(params_array) != 5:
            return

        # Check if parameters are valid
        if np.isnan(params_array).any():
            return

        # Extract ellipse parameters: [cx, cy, a, b, theta]
        cx, cy, semi_major, semi_minor, rotation = params_array

        # Generate ellipse points
        theta = np.linspace(start=0, stop=2 * np.pi, num=element.n_points)
        x_local = semi_major * np.cos(theta)
        y_local = semi_minor * np.sin(theta)

        # Rotate and translate
        cos_t = np.cos(rotation)
        sin_t = np.sin(rotation)
        x = cx + x_local * cos_t - y_local * sin_t
        y = cy + x_local * sin_t + y_local * cos_t

        # Create list of (x, y) tuples for polygon
        polygon_points = [(float(xi), float(yi)) for xi, yi in zip(x, y)]

        # Parse stroke color and apply opacity
        stroke_color = self._parse_rgb(color=element.style.stroke)
        stroke_color = (*stroke_color[:3], int(stroke_color[3] * element.style.opacity))

        # Draw as polygon (closed line)
        draw.line(
            xy=polygon_points + [polygon_points[0]],  # Close the loop
            fill=stroke_color,
            width=element.style.stroke_width
        )

    def composite_on_image(
            self,
            *,
            points: dict[str, np.ndarray],
            image: np.ndarray,
            metadata: dict[str, Any] | None = None
    ) -> np.ndarray:
        """Composite overlay onto image using pure Pillow.

        Args:
            points: Dictionary mapping point names to coordinates
            image: OpenCV image (BGR numpy array)
            metadata: Optional metadata for dynamic content

        Returns:
            Composited image (BGR numpy array)
        """
        if metadata is None:
            metadata = {}

        # Validate and compute all points
        missing_points = set(self.topology.required_points) - set(points.keys())
        if missing_points:
            raise ValueError(f"Missing required points: {missing_points}")

        # Compute derived points
        all_points = points.copy()
        for computed in self.topology.computed_points:
            try:
                all_points[computed.name] = computed.computation(all_points)
            except Exception as e:
                raise ValueError(f"Failed to compute point '{computed.name}': {e}")

        # Convert image to PIL
        image_rgb = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(obj=image_rgb).convert('RGBA')

        # Create transparent overlay
        overlay = Image.new(
            mode='RGBA',
            size=(self.topology.width, self.topology.height),
            color=(0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(im=overlay, mode='RGBA')

        # Render all elements using Pillow
        for element in self.topology.elements:
            if not element.visible:
                continue

            if isinstance(element, PointElement):
                self._render_pil_point(
                    draw=draw,
                    element=element,
                    points=all_points,
                    metadata=metadata
                )
            elif isinstance(element, LineElement):
                self._render_pil_line(
                    draw=draw,
                    element=element,
                    points=all_points
                )
            elif isinstance(element, CircleElement):
                self._render_pil_circle(
                    draw=draw,
                    element=element,
                    points=all_points
                )
            elif isinstance(element, CrosshairElement):
                self._render_pil_crosshair(
                    draw=draw,
                    element=element,
                    points=all_points
                )
            elif isinstance(element, TextElement):
                self._render_pil_text(
                    draw=draw,
                    element=element,
                    points=all_points,
                    metadata=metadata
                )
            elif isinstance(element, EllipseElement):
                self._render_pil_ellipse(
                    draw=draw,
                    element=element,
                    points=all_points
                )

        # Composite overlay onto image
        pil_image = Image.alpha_composite(im1=pil_image, im2=overlay)

        # Convert back to OpenCV format
        result_array = np.array(pil_image.convert('RGB'))
        return cv2.cvtColor(src=result_array, code=cv2.COLOR_RGB2BGR)

    def render_and_composite(
            self,
            *,
            image: np.ndarray,
            points: dict[str, np.ndarray],
            metadata: dict[str, Any] | None = None
    ) -> np.ndarray:
        """One-shot: render and composite overlay onto image.

        Args:
            image: OpenCV image (BGR numpy array)
            points: Dictionary mapping point names to coordinates
            metadata: Optional metadata for dynamic content

        Returns:
            Composited image (BGR numpy array)
        """
        return self.composite_on_image(
            points=points,
            image=image,
            metadata=metadata
        )

    def save_svg(
            self,
            *,
            svg_drawing: dw.Drawing,
            filepath: Path
    ) -> None:
        """Save SVG drawing to file.

        Args:
            svg_drawing: DrawSVG Drawing object
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        svg_drawing.save_svg(str(filepath))
