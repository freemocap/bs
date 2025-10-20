from typing import Any, Callable

import numpy as np
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


class EllipseElement(SVGElement):
    """A fitted ellipse drawn as an outline."""
    params_point: str  # Name of computed point containing ellipse params [cx, cy, a, b, theta]
    n_points: int = 100  # Number of points to use for drawing the ellipse
    style: LineStyle = Field(default_factory=LineStyle)


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

    def add_ellipse(
            self,
            *,
            name: str,
            params_point: str,
            n_points: int = 100,
            style: LineStyle | None = None
    ) -> "SVGTopology":
        """Add an ellipse element to the topology.

        Args:
            name: Element name
            params_point: Name of computed point containing ellipse params [cx, cy, a, b, theta]
            n_points: Number of points to approximate ellipse
            style: Line style for drawing ellipse outline
        """
        element = EllipseElement(
            name=name,
            params_point=params_point,
            n_points=n_points,
            style=style or LineStyle()
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


