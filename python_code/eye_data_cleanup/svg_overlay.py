from pathlib import Path
from typing import Any, Callable

import numpy as np
import drawsvg as dw
from PIL import Image, ImageDraw, ImageFont
import cv2
from pydantic import BaseModel, Field


class PointStyle(BaseModel):
    """Visual style for a point."""
    radius: int = 3
    fill: str = 'rgb(0, 255, 0)'
    stroke: str | None = None
    stroke_width: int | None = None
    opacity: float = 1.0


class LineStyle(BaseModel):
    """Visual style for a line."""
    stroke: str = 'rgb(255, 55, 55)'
    stroke_width: int = 2
    opacity: float = 1.0
    stroke_dasharray: str | None = None  # e.g., "5,5" for dashed


class TextStyle(BaseModel):
    """Visual style for text."""
    font_size: int = 12
    font_family: str = 'Arial, sans-serif'
    fill: str = 'white'
    stroke: str | None = 'black'
    stroke_width: int | None = 1
    font_weight: str = 'normal'
    text_anchor: str = 'start'  # 'start', 'middle', 'end'


class SVGElement(BaseModel):
    """Base class for SVG topology elements."""
    name: str
    visible: bool = True


class PointElement(SVGElement):
    """A point to be drawn at a named coordinate."""
    point_name: str  # Which named point to draw at
    style: PointStyle = Field(default_factory=PointStyle)
    label: str | None = None
    label_offset: tuple[float, float] = (5, -5)
    label_style: TextStyle = Field(default_factory=TextStyle)


class LineElement(SVGElement):
    """A line between two named points."""
    point_a: str
    point_b: str
    style: LineStyle = Field(default_factory=LineStyle)


class CircleElement(SVGElement):
    """A circle centered at a named point."""
    center_point: str
    radius: float
    style: PointStyle = Field(default_factory=PointStyle)


class CrosshairElement(SVGElement):
    """A crosshair at a named point."""
    center_point: str
    size: float = 10
    style: LineStyle = Field(default_factory=LineStyle)


class TextElement(SVGElement):
    """Text at a named point."""
    point_name: str
    text: str | Callable[[dict[str, Any]], str]  # Static or dynamic
    offset: tuple[float, float] = (0, 0)
    style: TextStyle = Field(default_factory=TextStyle)


class ComputedPoint(BaseModel):
    """A point computed from other named points."""
    name: str
    computation: Callable[[dict[str, np.ndarray]], np.ndarray]
    description: str = ""


class SVGTopology(BaseModel):
    """Defines the structure of an SVG overlay independent of data.

    This is the "blueprint" - it defines what elements exist and how they relate,
    but not where they are positioned (that comes from the data).
    """

    name: str
    required_points: list[str] = Field(default_factory=list)
    computed_points: list[ComputedPoint] = Field(default_factory=list)
    elements: list[SVGElement] = Field(default_factory=list)
    width: int = 640
    height: int = 480
    background_opacity: float = 0.0  # 0 = transparent
    css_styles: str | None = None  # Optional CSS for advanced styling

    def add_point(
            self,
            *,
            name: str,
            point_name: str,
            style: PointStyle | None = None,
            label: str | None = None,
            label_offset: tuple[float, float] = (5, -5)
    ) -> "SVGTopology":
        """Add a point element to the topology."""
        element = PointElement(
            name=name,
            point_name=point_name,
            style=style or PointStyle(),
            label=label,
            label_offset=label_offset
        )
        self.elements.append(element)
        return self

    def add_line(
            self,
            *,
            name: str,
            point_a: str,
            point_b: str,
            style: LineStyle | None = None
    ) -> "SVGTopology":
        """Add a line element to the topology."""
        element = LineElement(
            name=name,
            point_a=point_a,
            point_b=point_b,
            style=style or LineStyle()
        )
        self.elements.append(element)
        return self

    def add_circle(
            self,
            *,
            name: str,
            center_point: str,
            radius: float,
            style: PointStyle | None = None
    ) -> "SVGTopology":
        """Add a circle element to the topology."""
        element = CircleElement(
            name=name,
            center_point=center_point,
            radius=radius,
            style=style or PointStyle()
        )
        self.elements.append(element)
        return self

    def add_crosshair(
            self,
            *,
            name: str,
            center_point: str,
            size: float = 10,
            style: LineStyle | None = None
    ) -> "SVGTopology":
        """Add a crosshair element to the topology."""
        element = CrosshairElement(
            name=name,
            center_point=center_point,
            size=size,
            style=style or LineStyle()
        )
        self.elements.append(element)
        return self

    def add_text(
            self,
            *,
            name: str,
            point_name: str,
            text: str | Callable[[dict[str, Any]], str],
            offset: tuple[float, float] = (0, 0),
            style: TextStyle | None = None
    ) -> "SVGTopology":
        """Add a text element to the topology."""
        element = TextElement(
            name=name,
            point_name=point_name,
            text=text,
            offset=offset,
            style=style or TextStyle()
        )
        self.elements.append(element)
        return self

    def add_computed_point(
            self,
            *,
            name: str,
            computation: Callable[[dict[str, np.ndarray]], np.ndarray],
            description: str = ""
    ) -> "SVGTopology":
        """Add a computed point (derived from other points)."""
        computed = ComputedPoint(
            name=name,
            computation=computation,
            description=description
        )
        self.computed_points.append(computed)
        return self


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