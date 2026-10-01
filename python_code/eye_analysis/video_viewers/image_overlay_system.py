"""
Overlay rendering system for computer vision annotations.
Python side: Topology definition, JSON export, and raster image compositing.
Browser side: Load topology JSON and render with Canvas/SVG.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import cv2
import json

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _font_scale(font_size: int) -> float:
    """Convert a PIL-style point size into an approximate cv2 putText fontScale."""
    return max(font_size / 24.0, 0.3)


def _font_thickness(font_size: int) -> int:
    return max(1, round(font_size / 16))


def _rgba_to_bgr(rgba: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Drop alpha and reorder RGBA -> BGR for cv2 drawing calls."""
    r, g, b, _a = rgba
    return (int(b), int(g), int(r))


def _draw_with_opacity(
        *,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        opacity: float,
        draw_fn: Callable[[np.ndarray, int, int], None]
) -> None:
    """Draw via draw_fn, blended at the given opacity.

    draw_fn draws fully opaque onto the array it's given, using coordinates
    already shifted by the ROI's top-left corner (ox, oy). Only a padded ROI
    around bbox is copied/blended rather than the whole frame, so this stays
    cheap even though cv2 has no native per-pixel alpha compositing.
    """
    if opacity >= 0.999:
        draw_fn(image, 0, 0)
        return

    h, w = image.shape[:2]
    x0 = max(0, int(np.floor(bbox[0])))
    y0 = max(0, int(np.floor(bbox[1])))
    x1 = min(w, int(np.ceil(bbox[2])))
    y1 = min(h, int(np.ceil(bbox[3])))
    if x1 <= x0 or y1 <= y0:
        return

    roi = image[y0:y1, x0:x1]
    overlay_roi = roi.copy()
    draw_fn(overlay_roi, x0, y0)
    cv2.addWeighted(src1=overlay_roi, alpha=opacity, src2=roi, beta=1 - opacity, gamma=0, dst=roi)


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
    def render(
            self,
            *,
            image: np.ndarray,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        """Render this element directly onto a BGR image in-place."""
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

    def render(
            self,
            *,
            image: np.ndarray,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        point = self.point_ref.get_point(points=points)
        if not is_valid_point(point=point):
            return

        x, y = float(point[0]), float(point[1])
        r = self.style.radius
        stroke_width = self.style.stroke_width or 1

        fill_bgr = _rgba_to_bgr(parse_rgb(self.style.fill))
        stroke_bgr = _rgba_to_bgr(parse_rgb(self.style.stroke)) if self.style.stroke else None

        def draw_point(img: np.ndarray, ox: int, oy: int) -> None:
            center = (int(round(x - ox)), int(round(y - oy)))
            cv2.circle(img=img, center=center, radius=r, color=fill_bgr, thickness=-1, lineType=cv2.LINE_AA)
            if stroke_bgr is not None:
                cv2.circle(img=img, center=center, radius=r, color=stroke_bgr, thickness=stroke_width, lineType=cv2.LINE_AA)

        margin = r + stroke_width + 2
        _draw_with_opacity(
            image=image,
            bbox=(x - margin, y - margin, x + margin, y + margin),
            opacity=self.style.opacity,
            draw_fn=draw_point
        )

        if self.label:
            label_x = x + self.label_offset[0]
            label_y = y + self.label_offset[1]
            label_fill_bgr = _rgba_to_bgr(parse_rgb(self.label_style.fill))
            font_scale = _font_scale(self.label_style.font_size)
            thickness = _font_thickness(self.label_style.font_size)
            _, text_height = cv2.getTextSize(text=self.label, fontFace=_FONT, fontScale=font_scale, thickness=thickness)[0]
            origin = (int(round(label_x)), int(round(label_y + text_height)))

            if self.label_style.stroke and self.label_style.stroke_width:
                stroke_bgr = _rgba_to_bgr(parse_rgb(self.label_style.stroke))
                sw = self.label_style.stroke_width
                for dx in range(-sw, sw + 1):
                    for dy in range(-sw, sw + 1):
                        if dx != 0 or dy != 0:
                            cv2.putText(
                                img=image, text=self.label, org=(origin[0] + dx, origin[1] + dy),
                                fontFace=_FONT, fontScale=font_scale, color=stroke_bgr,
                                thickness=thickness, lineType=cv2.LINE_AA
                            )

            cv2.putText(
                img=image, text=self.label, org=origin, fontFace=_FONT, fontScale=font_scale,
                color=label_fill_bgr, thickness=thickness, lineType=cv2.LINE_AA
            )


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

    def render(
            self,
            *,
            image: np.ndarray,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        pt_a = self.point_a_ref.get_point(points=points)
        pt_b = self.point_b_ref.get_point(points=points)

        if not (is_valid_point(point=pt_a) and is_valid_point(point=pt_b)):
            return

        ax, ay = float(pt_a[0]), float(pt_a[1])
        bx, by = float(pt_b[0]), float(pt_b[1])
        stroke_bgr = _rgba_to_bgr(parse_rgb(self.style.stroke))
        width = self.style.stroke_width

        def draw_segment(img: np.ndarray, ox: int, oy: int) -> None:
            cv2.line(
                img=img,
                pt1=(int(round(ax - ox)), int(round(ay - oy))),
                pt2=(int(round(bx - ox)), int(round(by - oy))),
                color=stroke_bgr, thickness=width, lineType=cv2.LINE_AA
            )

        margin = width + 2
        _draw_with_opacity(
            image=image,
            bbox=(min(ax, bx) - margin, min(ay, by) - margin, max(ax, bx) + margin, max(ay, by) + margin),
            opacity=self.style.opacity,
            draw_fn=draw_segment
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

    def render(
            self,
            *,
            image: np.ndarray,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        center = self.center_ref.get_point(points=points)
        if not is_valid_point(point=center):
            return

        cx, cy = float(center[0]), float(center[1])
        r = self.radius
        stroke_width = self.style.stroke_width or 1

        fill_bgr = _rgba_to_bgr(parse_rgb(self.style.fill))
        stroke_bgr = _rgba_to_bgr(parse_rgb(self.style.stroke)) if self.style.stroke else None

        def draw_circle(img: np.ndarray, ox: int, oy: int) -> None:
            center_px = (int(round(cx - ox)), int(round(cy - oy)))
            r_px = int(round(r))
            cv2.circle(img=img, center=center_px, radius=r_px, color=fill_bgr, thickness=-1, lineType=cv2.LINE_AA)
            if stroke_bgr is not None:
                cv2.circle(img=img, center=center_px, radius=r_px, color=stroke_bgr, thickness=stroke_width, lineType=cv2.LINE_AA)

        margin = r + stroke_width + 2
        _draw_with_opacity(
            image=image,
            bbox=(cx - margin, cy - margin, cx + margin, cy + margin),
            opacity=self.style.opacity,
            draw_fn=draw_circle
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

    def render(
            self,
            *,
            image: np.ndarray,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        center = self.center_ref.get_point(points=points)
        if not is_valid_point(point=center):
            return

        cx, cy = float(center[0]), float(center[1])
        stroke_bgr = _rgba_to_bgr(parse_rgb(self.style.stroke))
        width = self.style.stroke_width
        size = self.size

        def draw_crosshair(img: np.ndarray, ox: int, oy: int) -> None:
            cx_px, cy_px = cx - ox, cy - oy
            cv2.line(
                img=img, pt1=(int(round(cx_px - size)), int(round(cy_px))),
                pt2=(int(round(cx_px + size)), int(round(cy_px))),
                color=stroke_bgr, thickness=width, lineType=cv2.LINE_AA
            )
            cv2.line(
                img=img, pt1=(int(round(cx_px)), int(round(cy_px - size))),
                pt2=(int(round(cx_px)), int(round(cy_px + size))),
                color=stroke_bgr, thickness=width, lineType=cv2.LINE_AA
            )

        margin = size + width + 2
        _draw_with_opacity(
            image=image,
            bbox=(cx - margin, cy - margin, cx + margin, cy + margin),
            opacity=self.style.opacity,
            draw_fn=draw_crosshair
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

    def render(
            self,
            *,
            image: np.ndarray,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        point = self.point_ref.get_point(points=points)
        if not is_valid_point(point=point):
            return

        x = float(point[0]) + self.offset[0]
        y = float(point[1]) + self.offset[1]
        fill_bgr = _rgba_to_bgr(parse_rgb(self.style.fill))

        # Support both static text and dynamic callable text
        if callable(self.text):
            text_to_render = self.text(metadata)
        else:
            text_to_render = self.text

        font_scale = _font_scale(self.style.font_size)
        thickness = _font_thickness(self.style.font_size)
        (_, text_height), _baseline = cv2.getTextSize(text=text_to_render, fontFace=_FONT, fontScale=font_scale, thickness=thickness)
        origin = (int(round(x)), int(round(y + text_height)))

        if self.style.stroke and self.style.stroke_width:
            stroke_bgr = _rgba_to_bgr(parse_rgb(self.style.stroke))
            sw = self.style.stroke_width
            for dx in range(-sw, sw + 1):
                for dy in range(-sw, sw + 1):
                    if dx != 0 or dy != 0:
                        cv2.putText(
                            img=image, text=text_to_render, org=(origin[0] + dx, origin[1] + dy),
                            fontFace=_FONT, fontScale=font_scale, color=stroke_bgr,
                            thickness=thickness, lineType=cv2.LINE_AA
                        )

        cv2.putText(
            img=image, text=text_to_render, org=origin, fontFace=_FONT, fontScale=font_scale,
            color=fill_bgr, thickness=thickness, lineType=cv2.LINE_AA
        )


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

    def render(
            self,
            *,
            image: np.ndarray,
            points: dict[str, dict[str, np.ndarray]],
            metadata: dict[str, Any],
            parse_rgb: Callable[[str], tuple[int, int, int, int]]
    ) -> None:
        params = self.params_ref.get_point(points=points)
        if params is None or len(params) != 5 or np.isnan(params).any():
            return

        cx, cy, semi_major, semi_minor, rotation = (float(p) for p in params)
        stroke_bgr = _rgba_to_bgr(parse_rgb(self.style.stroke))
        width = self.style.stroke_width

        def draw_ellipse(img: np.ndarray, ox: int, oy: int) -> None:
            cv2.ellipse(
                img=img,
                center=(int(round(cx - ox)), int(round(cy - oy))),
                axes=(int(round(semi_major)), int(round(semi_minor))),
                angle=np.degrees(rotation),
                startAngle=0, endAngle=360,
                color=stroke_bgr, thickness=width, lineType=cv2.LINE_AA
            )

        extent = max(semi_major, semi_minor) + width + 2
        _draw_with_opacity(
            image=image,
            bbox=(cx - extent, cy - extent, cx + extent, cy + extent),
            opacity=self.style.opacity,
            draw_fn=draw_ellipse
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
    """Renders overlays onto raster images using OpenCV."""
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

        result = image.copy()
        for element in self.topology.elements:
            if element.visible:
                element.render(
                    image=result,
                    points=all_points,
                    metadata=metadata,
                    parse_rgb=self._parse_rgb
                )

        return result


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