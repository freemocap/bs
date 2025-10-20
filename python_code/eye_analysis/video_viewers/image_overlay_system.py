"""
Overlay rendering system for computer vision annotations.
Python side: Topology definition, JSON export, and raster image compositing.
Browser side: Load topology JSON and render with Canvas/SVG.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from PIL import ImageDraw, ImageFont, Image
import cv2
import json


# ============================================================================
# STYLE CLASSES
# ============================================================================

class PointStyle(BaseModel):
    """Visual style for point markers."""
    radius: int = 3
    fill: str = 'rgb(0, 255, 0)'
    stroke: str | None = None
    stroke_width: int | None = None
    opacity: float = 1.0


class LineStyle(BaseModel):
    """Visual style for lines."""
    stroke: str = 'rgb(255, 55, 55)'
    stroke_width: int = 2
    opacity: float = 1.0
    stroke_dasharray: str | None = None


class TextStyle(BaseModel):
    """Visual style for text labels."""
    font_size: int = 12
    font_family: str = 'Arial, sans-serif'
    fill: str = 'white'
    stroke: str | None = 'black'
    stroke_width: int | None = 1
    font_weight: str = 'normal'
    text_anchor: str = 'start'


# ============================================================================
# ELEMENT CLASSES
# ============================================================================

class OverlayElement(BaseModel, ABC):
    """Base class for overlay elements."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    element_type: str
    name: str
    visible: bool = True
    
    @abstractmethod
    def render_pil(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        points: dict[str, np.ndarray],
        metadata: dict[str, Any],
        parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        """Render this element using PIL."""
        pass
    
def is_valid_point(*, point: np.ndarray | None) -> bool:
    """Check if point has valid coordinates."""
    return point is not None and not np.isnan(point).any()


class PointElement(OverlayElement):
    """A point marker with optional label."""
    element_type: str = 'point'
    point_name: str
    style: PointStyle = Field(default_factory=PointStyle)
    label: str | None = None
    label_offset: tuple[float, float] = (5, -5)
    label_style: TextStyle = Field(default_factory=TextStyle)

    def render_pil(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        points: dict[str, np.ndarray],
        metadata: dict[str, Any],
        parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        point = points.get(self.point_name)
        if not is_valid_point(point=point):
            return

        x, y = float(point[0]), float(point[1])
        r = self.style.radius

        fill_color_raw = parse_rgb(self.style.fill)
        fill_color = (*fill_color_raw[:3], int(fill_color_raw[3] * self.style.opacity))

        draw.ellipse(
            xy=[(x - r, y - r), (x + r, y + r)],
            fill=fill_color,
            outline=parse_rgb(self.style.stroke) if self.style.stroke else None,
            width=self.style.stroke_width or 0
        )

        if self.label:
            label_x = x + self.label_offset[0]
            label_y = y + self.label_offset[1]
            label_fill = parse_rgb(self.label_style.fill)

            try:
                font = ImageFont.truetype(font='arial.ttf', size=self.label_style.font_size)
            except:
                font = ImageFont.load_default()

            if self.label_style.stroke and self.label_style.stroke_width:
                stroke_color = parse_rgb(self.label_style.stroke)
                for dx in range(-self.label_style.stroke_width, self.label_style.stroke_width + 1):
                    for dy in range(-self.label_style.stroke_width, self.label_style.stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text(
                                xy=(label_x + dx, label_y + dy),
                                text=self.label,
                                fill=stroke_color,
                                font=font
                            )

            draw.text(xy=(label_x, label_y), text=self.label, fill=label_fill, font=font)


class LineElement(OverlayElement):
    """A line between two points."""
    element_type: str = 'line'
    point_a: str
    point_b: str
    style: LineStyle = Field(default_factory=LineStyle)

    def render_pil(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        points: dict[str, np.ndarray],
        metadata: dict[str, Any],
        parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        pt_a = points.get(self.point_a)
        pt_b = points.get(self.point_b)

        if not (is_valid_point(point=pt_a) and is_valid_point(point=pt_b)):
            return

        stroke_color = parse_rgb(self.style.stroke)
        stroke_color = (*stroke_color[:3], int(stroke_color[3] * self.style.opacity))

        draw.line(
            xy=[(float(pt_a[0]), float(pt_a[1])), (float(pt_b[0]), float(pt_b[1]))],
            fill=stroke_color,
            width=self.style.stroke_width
        )


class CircleElement(OverlayElement):
    """A circle centered at a point."""
    element_type: str = 'circle'
    center_point: str
    radius: float
    style: PointStyle = Field(default_factory=PointStyle)

    def render_pil(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        points: dict[str, np.ndarray],
        metadata: dict[str, Any],
        parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        center = points.get(self.center_point)
        if not is_valid_point(point=center):
            return

        cx, cy = float(center[0]), float(center[1])
        r = self.radius

        fill_color = parse_rgb(self.style.fill)
        fill_color = (*fill_color[:3], int(fill_color[3] * self.style.opacity))

        draw.ellipse(
            xy=[(cx - r, cy - r), (cx + r, cy + r)],
            fill=fill_color,
            outline=parse_rgb(self.style.stroke) if self.style.stroke else None,
            width=self.style.stroke_width or 0
        )


class CrosshairElement(OverlayElement):
    """A crosshair at a point."""
    element_type: str = 'crosshair'
    center_point: str
    size: float = 10
    style: LineStyle = Field(default_factory=LineStyle)

    def render_pil(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        points: dict[str, np.ndarray],
        metadata: dict[str, Any],
        parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        center = points.get(self.center_point)
        if not is_valid_point(point=center):
            return

        cx, cy = float(center[0]), float(center[1])
        stroke_color = parse_rgb(self.style.stroke)
        stroke_color = (*stroke_color[:3], int(stroke_color[3] * self.style.opacity))

        draw.line(
            xy=[(cx - self.size, cy), (cx + self.size, cy)],
            fill=stroke_color,
            width=self.style.stroke_width
        )

        draw.line(
            xy=[(cx, cy - self.size), (cx, cy + self.size)],
            fill=stroke_color,
            width=self.style.stroke_width
        )


class TextElement(OverlayElement):
    """Text label at a point with support for dynamic text via callable."""
    element_type: str = 'text'
    point_name: str
    text: str | Callable[[dict[str, Any]], str]
    offset: tuple[float, float] = (0, 0)
    style: TextStyle = Field(default_factory=TextStyle)

    def render_pil(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        points: dict[str, np.ndarray],
        metadata: dict[str, Any],
        parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        point = points.get(self.point_name)
        if not is_valid_point(point=point):
            return

        x = float(point[0]) + self.offset[0]
        y = float(point[1]) + self.offset[1]
        fill_color = parse_rgb(self.style.fill)

        # Support both static text and dynamic callable text
        if callable(self.text):
            text_to_render = self.text(metadata)
        else:
            text_to_render = self.text

        try:
            font = ImageFont.truetype(font='arial.ttf', size=self.style.font_size)
        except:
            font = ImageFont.load_default()

        if self.style.stroke and self.style.stroke_width:
            stroke_color = parse_rgb(self.style.stroke)
            for dx in range(-self.style.stroke_width, self.style.stroke_width + 1):
                for dy in range(-self.style.stroke_width, self.style.stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text(
                            xy=(x + dx, y + dy),
                            text=text_to_render,
                            fill=stroke_color,
                            font=font
                        )

        draw.text(xy=(x, y), text=text_to_render, fill=fill_color, font=font)


class EllipseElement(OverlayElement):
    """A fitted ellipse from parameters."""
    element_type: str = 'ellipse'
    params_point: str
    n_points: int = 100
    style: LineStyle = Field(default_factory=LineStyle)

    def render_pil(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        points: dict[str, np.ndarray],
        metadata: dict[str, Any],
        parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        params = points.get(self.params_point)
        if params is None or len(params) != 5 or np.isnan(params).any():
            return

        cx, cy, semi_major, semi_minor, rotation = params

        theta = np.linspace(start=0, stop=2 * np.pi, num=self.n_points)
        x_local = semi_major * np.cos(theta)
        y_local = semi_minor * np.sin(theta)

        cos_t, sin_t = np.cos(rotation), np.sin(rotation)
        x = cx + x_local * cos_t - y_local * sin_t
        y = cy + x_local * sin_t + y_local * cos_t

        path_points = [(float(xi), float(yi)) for xi, yi in zip(x, y)]

        stroke_color = parse_rgb(self.style.stroke)
        stroke_color = (*stroke_color[:3], int(stroke_color[3] * self.style.opacity))

        draw.line(
            xy=path_points + [path_points[0]],
            fill=stroke_color,
            width=self.style.stroke_width
        )


# ============================================================================
# TOPOLOGY
# ============================================================================

class ComputedPoint(BaseModel):
    """A point computed from other points."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    computation: Callable[[dict[str, np.ndarray]], np.ndarray]
    description: str = ""


class OverlayTopology(BaseModel):
    """Defines overlay structure independent of point data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    required_points: list[str] = Field(default_factory=list)
    computed_points: list[ComputedPoint] = Field(default_factory=list)
    elements: list[OverlayElement] = Field(default_factory=list)
    width: int = 640
    height: int = 480

    def add(self, *, element: OverlayElement) -> "OverlayTopology":
        """Add any element to the topology."""
        self.elements.append(element)
        return self

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for web."""
        return {
            'name': self.name,
            'required_points': self.required_points,
            'width': self.width,
            'height': self.height,
            'elements': [
                json.loads(elem.model_dump_json())
                for elem in self.elements
            ]
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_json_dict(), indent=2)

    def save_json(self, *, filepath: str) -> None:
        """Save topology to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())


# ============================================================================
# RENDERER
# ============================================================================

class OverlayRenderer(BaseModel):
    """Renders overlays onto raster images using PIL."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    topology: OverlayTopology
    color_map: dict[str, tuple[int, int, int, int]] = Field(default_factory=dict, init=False)

    def model_post_init(self, __context: Any) -> None:
        """Initialize color map after model creation."""
        self.color_map = {
            'red': (255, 0, 0, 255), 'green': (0, 255, 0, 255),
            'blue': (0, 0, 255, 255), 'yellow': (255, 255, 0, 255),
            'lime': (0, 255, 0, 255), 'white': (255, 255, 255, 255),
            'black': (0, 0, 0, 255), 'cyan': (0, 255, 255, 255),
            'magenta': (255, 0, 255, 255),
        }

    def _parse_rgb(self, color: str) -> tuple[int, int, int, int]:
        """Parse color string to RGBA tuple."""
        color = color.strip().lower()

        if color.startswith('rgb(') and color.endswith(')'):
            values = color[4:-1].split(',')
            r, g, b = [int(v.strip()) for v in values]
            return (r, g, b, 255)

        return self.color_map.get(color, (255, 255, 255, 255))

    def _compute_all_points(
        self,
        *,
        points: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Validate and compute derived points."""
        missing = set(self.topology.required_points) - set(points.keys())
        if missing:
            raise ValueError(f"Missing required points: {missing}")

        all_points = points.copy()
        for computed in self.topology.computed_points:
            try:
                all_points[computed.name] = computed.computation(all_points)
            except Exception as e:
                raise ValueError(f"Failed to compute '{computed.name}': {e}")

        return all_points

    def composite_on_image(
            self,
            *,
            image: np.ndarray,
            points: dict[str, np.ndarray],
            metadata: dict[str, Any] | None = None
    ) -> np.ndarray:
        """Composite overlay onto raster image.

        Args:
            image: OpenCV image (BGR numpy array)
            points: Dictionary mapping point names to (x, y) coordinates
            metadata: Optional metadata for dynamic content

        Returns:
            Composited image (BGR numpy array)
        """
        if metadata is None:
            metadata = {}

        all_points = self._compute_all_points(points=points)

        # Convert to PIL
        image_rgb = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(obj=image_rgb).convert('RGBA')

        # Use actual image dimensions instead of topology dimensions
        img_height, img_width = image.shape[:2]

        # Create overlay with same size as input image
        overlay = Image.new(
            mode='RGBA',
            size=(img_width, img_height),
            color=(0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(im=overlay, mode='RGBA')

        # Render all elements
        for element in self.topology.elements:
            if element.visible:
                element.render_pil(
                    draw=draw,
                    points=all_points,
                    metadata=metadata,
                    parse_rgb=self._parse_rgb
                )

        # Composite
        result = Image.alpha_composite(im1=pil_image, im2=overlay)
        result_array = np.array(result.convert('RGB'))
        return cv2.cvtColor(src=result_array, code=cv2.COLOR_RGB2BGR)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def overlay_image(
        *,
        image: np.ndarray,
        topology: OverlayTopology,
        points: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None
) -> np.ndarray:
    """Convenience function to render overlay on image."""
    return OverlayRenderer(topology=topology).composite_on_image(
        image=image,
        points=points,
        metadata=metadata
    )