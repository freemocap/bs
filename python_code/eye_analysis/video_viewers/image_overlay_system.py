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
    """Visual style for point keypoints."""
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
# POINT REFERENCE SYSTEM
# ============================================================================

class PointReference(BaseModel):
    """Reference to a point in the nested data structure.

    Supports paths like:
    - ('cleaned', 'p1') -> points['cleaned']['p1']
    - ('raw', 'tear_duct') -> points['raw']['tear_duct']
    - ('computed', 'pupil_center') -> points['computed']['pupil_center']
    """
    data_type: str  # e.g., 'cleaned', 'raw', 'computed'
    name: str  # e.g., 'p1', 'tear_duct'

    @classmethod
    def parse(cls, *, reference: str | tuple[str, str]) -> "PointReference":
        """Parse various reference formats into PointReference.

        Args:
            reference: Can be:
                - PointReference instance (returned as-is)
                - tuple like ('cleaned', 'p1')
                - string like 'cleaned.p1' or 'cleaned/p1'
        """
        if isinstance(reference, PointReference):
            return reference
        elif isinstance(reference, tuple):
            return cls(data_type=reference[0], name=reference[1])
        elif isinstance(reference, str):
            # Support both dot and slash separators
            if '.' in reference:
                parts = reference.split('.', maxsplit=1)
            elif '/' in reference:
                parts = reference.split('/', maxsplit=1)
            else:
                raise ValueError(f"Invalid point reference string: {reference}. Must contain '.' or '/'")

            if len(parts) != 2:
                raise ValueError(f"Invalid point reference: {reference}")
            return cls(data_type=parts[0], name=parts[1])
        else:
            raise ValueError(f"Invalid reference type: {type(reference)}")

    def get_point(self, *, points: dict[str, dict[str, np.ndarray]]) -> np.ndarray | None:
        """Retrieve point from nested dictionary."""
        data_type_dict = points.get(self.data_type)
        if data_type_dict is None:
            return None
        return data_type_dict.get(self.name)

    def __str__(self) -> str:
        return f"{self.data_type}.{self.name}"


def is_valid_point(*, point: np.ndarray | None) -> bool:
    """Check if point has valid coordinates."""
    return point is not None and not np.isnan(point).any()


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
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        """Render this element using PIL."""
        pass


class PointElement(OverlayElement):
    """A point keypoint with optional label."""
    element_type: str = 'point'
    point_ref: PointReference
    style: PointStyle = Field(default_factory=PointStyle)
    label: str | None = None
    label_offset: tuple[float, float] = (5, -5)
    label_style: TextStyle = Field(default_factory=TextStyle)

    def __init__(self, *, point_name: str | tuple[str, str] | PointReference, **kwargs: Any):
        """Initialize with flexible point reference."""
        point_ref = PointReference.parse(reference=point_name)
        super().__init__(point_ref=point_ref, **kwargs)

    def render_pil(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        point = self.point_ref.get_point(points=points)
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
    point_a_ref: PointReference
    point_b_ref: PointReference
    style: LineStyle = Field(default_factory=LineStyle)

    def __init__(
            self,
            *,
            point_a: str | tuple[str, str] | PointReference,
            point_b: str | tuple[str, str] | PointReference,
            **kwargs: Any
    ):
        """Initialize with flexible point references."""
        point_a_ref = PointReference.parse(reference=point_a)
        point_b_ref = PointReference.parse(reference=point_b)
        super().__init__(point_a_ref=point_a_ref, point_b_ref=point_b_ref, **kwargs)

    def render_pil(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        pt_a = self.point_a_ref.get_point(points=points)
        pt_b = self.point_b_ref.get_point(points=points)

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
    center_ref: PointReference
    radius: float
    style: PointStyle = Field(default_factory=PointStyle)

    def __init__(
            self,
            *,
            center_point: str | tuple[str, str] | PointReference,
            **kwargs: Any
    ):
        """Initialize with flexible point reference."""
        center_ref = PointReference.parse(reference=center_point)
        super().__init__(center_ref=center_ref, **kwargs)

    def render_pil(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        center = self.center_ref.get_point(points=points)
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
    center_ref: PointReference
    size: float = 10
    style: LineStyle = Field(default_factory=LineStyle)

    def __init__(
            self,
            *,
            center_point: str | tuple[str, str] | PointReference,
            **kwargs: Any
    ):
        """Initialize with flexible point reference."""
        center_ref = PointReference.parse(reference=center_point)
        super().__init__(center_ref=center_ref, **kwargs)

    def render_pil(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        center = self.center_ref.get_point(points=points)
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
    point_ref: PointReference
    text: str | Callable[[dict[str, Any]], str]
    offset: tuple[float, float] = (0, 0)
    style: TextStyle = Field(default_factory=TextStyle)

    def __init__(
            self,
            *,
            point_name: str | tuple[str, str] | PointReference,
            **kwargs: Any
    ):
        """Initialize with flexible point reference."""
        point_ref = PointReference.parse(reference=point_name)
        super().__init__(point_ref=point_ref, **kwargs)

    def render_pil(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        point = self.point_ref.get_point(points=points)
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
    params_ref: PointReference
    n_points: int = 100
    style: LineStyle = Field(default_factory=LineStyle)

    def __init__(
            self,
            *,
            params_point: str | tuple[str, str] | PointReference,
            **kwargs: Any
    ):
        """Initialize with flexible point reference."""
        params_ref = PointReference.parse(reference=params_point)
        super().__init__(params_ref=params_ref, **kwargs)

    def render_pil(
            self,
            *,
            draw: ImageDraw.ImageDraw,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        params = self.params_ref.get_point(points=points)
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

    data_type: str  # Which data type dict to store in: 'computed', 'cleaned', etc.
    name: str
    computation: Callable[[dict[str, dict[str, np.ndarray]]], np.ndarray]
    description: str = ""


class OverlayTopology(BaseModel):
    """Defines overlay structure independent of point data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    required_points: list[tuple[str, str]] = Field(default_factory=list)  # List of (data_type, name) tuples
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
            points: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Validate and compute derived points."""
        # Validate required points exist
        for data_type, name in self.topology.required_points:
            if data_type not in points or name not in points[data_type]:
                # Don't raise error - some points might be missing/NaN
                pass

        all_points = {k: dict(v) for k, v in points.items()}  # Deep copy

        # Compute derived points
        for computed in self.topology.computed_points:
            try:
                result = computed.computation(all_points)

                # Ensure the data_type dict exists
                if computed.data_type not in all_points:
                    all_points[computed.data_type] = {}

                all_points[computed.data_type][computed.name] = result
            except Exception as e:
                print(f"Warning: Failed to compute '{computed.data_type}.{computed.name}': {e}")
                # Continue even if computation fails

        return all_points

    def composite_on_image(
            self,
            *,
            image: np.ndarray,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any] | None = None
    ) -> np.ndarray:
        """Composite overlay onto raster image.

        Args:
            image: OpenCV image (BGR numpy array)
            points: Nested dict mapping data_type -> name -> (x, y) coordinates
                   e.g., {'cleaned': {'p1': array([x, y])}, 'raw': {'p1': array([x, y])}}
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
        points: dict[str, dict[str, np.ndarray]],
        metadata: dict[str, Any] | None = None
) -> np.ndarray:
    """Convenience function to render overlay on image."""
    return OverlayRenderer(topology=topology).composite_on_image(
        image=image,
        points=points,
        metadata=metadata
    )