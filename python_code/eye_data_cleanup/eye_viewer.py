"""Integration of SVG topology system with eye tracking viewer.

Shows how to define reusable topologies and integrate with existing cv2 workflow.
Supports toggling between raw, cleaned, and both data visualizations.
Includes fitted ellipse overlays and active contour snakes for pupil outline.
"""

from pathlib import Path
from enum import Enum

import cv2
import numpy as np

from python_code.eye_data_cleanup.csv_io import load_trajectory_csv, ABaseModel, TrajectoryDataset
from python_code.eye_data_cleanup.svg_overlay import (
    SVGTopology, SVGOverlayRenderer,
    PointStyle, LineStyle, TextStyle
)
from python_code.eye_data_cleanup.superellipse_fit import fit_ellipse_to_points
from python_code.eye_data_cleanup.active_contour_fit import (
    fit_snake_safe, SnakeParams, SnakeContour
)

DEFAULT_RESIZE_FACTOR: float = 1.0
DEFAULT_MIN_CONFIDENCE: float = 0.3


class ViewMode(str, Enum):
    """Visualization mode for eye tracking data."""
    RAW = "raw"
    CLEANED = "cleaned"
    BOTH = "both"


class VideoHelper(ABaseModel):
    """Helper class for managing video capture and metadata."""

    video_path: Path
    video_capture: cv2.VideoCapture | None = None
    timestamps: list[float] | None = None
    resize_factor: float = DEFAULT_RESIZE_FACTOR

    @property
    def video_name(self) -> str:
        """Get the video filename without extension."""
        return self.video_path.stem

    @property
    def width(self) -> int:
        """Get the video width after applying resize factor."""
        if self.video_capture is None:
            raise ValueError("Video capture is not initialized.")
        return int(self.video_capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH) * self.resize_factor)

    @property
    def height(self) -> int:
        """Get the video height after applying resize factor."""
        if self.video_capture is None:
            raise ValueError("Video capture is not initialized.")
        return int(self.video_capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT) * self.resize_factor)

    @classmethod
    def create(
        cls,
        *,
        video_path: Path,
        timestamps_npy_path: Path | None = None,
        resize_factor: float = DEFAULT_RESIZE_FACTOR,
    ) -> "VideoHelper":
        """Create a VideoHelper instance from a video file."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vid_cap: cv2.VideoCapture = cv2.VideoCapture(filename=str(video_path))
        if not vid_cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        timestamps_array: np.ndarray | None = None
        if timestamps_npy_path is not None:
            if not Path(timestamps_npy_path).exists():
                raise FileNotFoundError(f"Timestamps file not found: {timestamps_npy_path}")
            timestamps_array = np.load(file=timestamps_npy_path).astype(np.float64)

        return cls(
            video_path=video_path,
            video_capture=vid_cap,
            timestamps=list(timestamps_array) if timestamps_array is not None else None,
            resize_factor=resize_factor,
        )


class EyeVideoDataset(ABaseModel):
    """Dataset for eye tracking video with pupil landmarks."""

    data_name: str
    base_path: Path
    video: VideoHelper
    pixel_trajectories: TrajectoryDataset
    snake_params: SnakeParams = SnakeParams()

    # Define the landmark indices for pupil points
    landmarks: dict[str, int] = {
        "p1": 0, "p2": 1, "p3": 2, "p4": 3,
        "p5": 4, "p6": 5, "p7": 6, "p8": 7,
        "tear_duct": 8, "outer_eye": 9,
    }

    @classmethod
    def create(
        cls,
        *,
        data_name: str,
        base_path: Path,
        raw_video_path: Path,
        timestamps_npy_path: Path,
        data_csv_path: Path,
        resize_factor: float = DEFAULT_RESIZE_FACTOR,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        butterworth_cutoff: float = 6.0,
        butterworth_sampling_rate: float = 30.0,
        snake_params: SnakeParams | None = None
    ) -> "EyeVideoDataset":
        """Create an EyeVideoDataset instance."""
        return cls(
            data_name=data_name,
            base_path=base_path,
            video=VideoHelper.create(
                video_path=raw_video_path,
                timestamps_npy_path=timestamps_npy_path,
                resize_factor=resize_factor
            ),
            pixel_trajectories=load_trajectory_csv(
                filepath=data_csv_path,
                min_confidence=min_confidence,
                butterworth_cutoff=butterworth_cutoff,
                butterworth_sampling_rate=butterworth_sampling_rate
            ),
            snake_params=snake_params or SnakeParams()
        )


# ==================== TOPOLOGY DEFINITIONS ====================

def create_full_eye_topology(
    *,
    width: int,
    height: int,
    show_raw: bool = True,
    show_cleaned: bool = True,
    show_dots: bool = True,
    show_ellipse: bool = True,
    show_snake: bool = True,
    n_snake_points: int = 20
) -> SVGTopology:
    """Create full eye tracking topology with configurable raw/cleaned/snake display.

    Required points: p1-p8, tear_duct, outer_eye (with _raw and/or _cleaned suffixes)
    Optional snake points: snake_contour_raw_0..N, snake_contour_cleaned_0..N
    Computed points: pupil_center_raw, pupil_center_cleaned, fitted_ellipse_raw, fitted_ellipse_cleaned
    Elements: connections (cleaned only), points, labels, fitted ellipses, snake contours, info overlay

    Args:
        width: Video width
        height: Video height
        show_raw: Whether to show raw data
        show_cleaned: Whether to show cleaned data
        show_dots: Whether to show landmark dots
        show_ellipse: Whether to show fitted ellipses
        show_snake: Whether to show snake contours
        n_snake_points: Number of points in snake contour

    Returns:
        SVGTopology configured for specified view mode
    """
    required_points = []
    if show_cleaned:
        required_points.extend([
            "p1_cleaned", "p2_cleaned", "p3_cleaned", "p4_cleaned",
            "p5_cleaned", "p6_cleaned", "p7_cleaned", "p8_cleaned",
            "tear_duct_cleaned", "outer_eye_cleaned"
        ])
    if show_raw:
        required_points.extend([
            "p1_raw", "p2_raw", "p3_raw", "p4_raw",
            "p5_raw", "p6_raw", "p7_raw", "p8_raw",
            "tear_duct_raw", "outer_eye_raw"
        ])

    topology = SVGTopology(
        name="full_eye_tracking",
        width=width,
        height=height,
        required_points=required_points
    )

    # === COMPUTED PUPIL CENTERS ===
    if show_raw:
        def compute_pupil_center_raw(points: dict[str, np.ndarray]) -> np.ndarray:
            pupil_points = [points[f"p{i}_raw"] for i in range(1, 9) if f"p{i}_raw" in points]
            if not pupil_points:
                return np.array([np.nan, np.nan])
            stacked = np.stack(pupil_points, axis=0)
            return np.nanmean(stacked, axis=0)

        topology.add_computed_point(
            name="pupil_center_raw",
            computation=compute_pupil_center_raw,
            description="Mean of raw pupil points"
        )

    if show_cleaned:
        def compute_pupil_center_cleaned(points: dict[str, np.ndarray]) -> np.ndarray:
            pupil_points = [points[f"p{i}_cleaned"] for i in range(1, 9) if f"p{i}_cleaned" in points]
            if not pupil_points:
                return np.array([np.nan, np.nan])
            stacked = np.stack(pupil_points, axis=0)
            return np.nanmean(stacked, axis=0)

        topology.add_computed_point(
            name="pupil_center_cleaned",
            computation=compute_pupil_center_cleaned,
            description="Mean of cleaned pupil points"
        )

    # === CONNECTION LINES (CLEANED ONLY) ===
    if show_cleaned:
        line_style = LineStyle(stroke='rgb(0, 200, 255)', stroke_width=2)

        # Pupil outline (closed loop)
        connections = [
            ("p1_cleaned", "p2_cleaned"), ("p2_cleaned", "p3_cleaned"),
            ("p3_cleaned", "p4_cleaned"), ("p4_cleaned", "p5_cleaned"),
            ("p5_cleaned", "p6_cleaned"), ("p6_cleaned", "p7_cleaned"),
            ("p7_cleaned", "p8_cleaned"), ("p8_cleaned", "p1_cleaned")
        ]
        for i, (pa, pb) in enumerate(connections):
            topology.add_line(
                name=f"pupil_connection_{i}",
                point_a=pa,
                point_b=pb,
                style=line_style
            )

        # Eye corner connections
        topology.add_line(
            name="eye_span",
            point_a="tear_duct_cleaned",
            point_b="outer_eye_cleaned",
            style=line_style
        )
        topology.add_line(
            name="tear_to_pupil",
            point_a="tear_duct_cleaned",
            point_b="p1_cleaned",
            style=line_style
        )

    # === LANDMARK POINTS ===

    # CLEANED POINTS (cyan/blue)
    if show_cleaned and show_dots:
        cleaned_point_style = PointStyle(radius=3, fill='rgb(0, 200, 255)')

        # Pupil points
        for i in range(1, 9):
            topology.add_point(
                name=f"pupil_point_{i}_cleaned",
                point_name=f"p{i}_cleaned",
                style=cleaned_point_style,
                label=f"p{i}",
                label_offset=(5, -5)
            )

        # Eye corners
        cleaned_corner_style = PointStyle(radius=4, fill='rgb(0, 220, 255)')
        for name in ["tear_duct", "outer_eye"]:
            topology.add_point(
                name=f"{name}_point_cleaned",
                point_name=f"{name}_cleaned",
                style=cleaned_corner_style,
                label=name.replace('_', ' ').title(),
                label_offset=(5, -5)
            )

    # RAW POINTS (red/orange)
    if show_raw and show_dots:
        raw_point_style = PointStyle(radius=2, fill='rgb(255, 100, 50)')

        # Pupil points
        for i in range(1, 9):
            topology.add_point(
                name=f"pupil_point_{i}_raw",
                point_name=f"p{i}_raw",
                style=raw_point_style,
                label=f"p{i}" if not show_cleaned else None,
                label_offset=(5, -5)
            )

        # Eye corners
        raw_corner_style = PointStyle(radius=4, fill='rgb(255, 120, 0)')
        for name in ["tear_duct", "outer_eye"]:
            topology.add_point(
                name=f"{name}_point_raw",
                point_name=f"{name}_raw",
                style=raw_corner_style,
                label=name.replace('_', ' ').title() if not show_cleaned else None,
                label_offset=(5, -5)
            )

    # === FITTED ELLIPSES ===

    if show_cleaned and show_ellipse:
        def compute_fitted_ellipse_cleaned(points: dict[str, np.ndarray]) -> np.ndarray:
            """Fit ellipse to cleaned pupil points and return params as [cx, cy, a, b, theta]."""
            pupil_points = np.array([points[f"p{i}_cleaned"] for i in range(1, 9)])

            try:
                ellipse_params = fit_ellipse_to_points(points=pupil_points)
                return ellipse_params.to_array()
            except (ValueError, cv2.error):
                # Return NaN if fitting fails
                return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        topology.add_computed_point(
            name="fitted_ellipse_cleaned",
            computation=compute_fitted_ellipse_cleaned,
            description="Fitted ellipse parameters for cleaned pupil points"
        )

        topology.add_ellipse(
            name="pupil_ellipse_cleaned",
            params_point="fitted_ellipse_cleaned",
            n_points=100,
            style=LineStyle(
                stroke='rgb(255, 0, 255)',  # Magenta
                stroke_width=2,
                opacity=0.8
            )
        )

    if show_raw and show_ellipse:
        def compute_fitted_ellipse_raw(points: dict[str, np.ndarray]) -> np.ndarray:
            """Fit ellipse to raw pupil points and return params as [cx, cy, a, b, theta]."""
            pupil_points = np.array([points[f"p{i}_raw"] for i in range(1, 9)])

            try:
                ellipse_params = fit_ellipse_to_points(points=pupil_points)
                return ellipse_params.to_array()
            except (ValueError, cv2.error):
                # Return NaN if fitting fails
                return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

        topology.add_computed_point(
            name="fitted_ellipse_raw",
            computation=compute_fitted_ellipse_raw,
            description="Fitted ellipse parameters for raw pupil points"
        )

        topology.add_ellipse(
            name="pupil_ellipse_raw",
            params_point="fitted_ellipse_raw",
            n_points=100,
            style=LineStyle(
                stroke='rgb(255, 150, 0)',  # Orange
                stroke_width=2,
                opacity=0.6
            )
        )

    # === SNAKE CONTOURS ===

    if show_snake and show_cleaned:
        # Snake points (green)
        snake_point_style = PointStyle(radius=4, fill='rgb(0, 255, 100)', opacity=0.9)

        for i in range(n_snake_points):
            topology.add_point(
                name=f"snake_point_{i}_cleaned",
                point_name=f"snake_contour_cleaned_{i}",
                style=snake_point_style
            )

        # Connect snake points in a loop
        snake_line_style = LineStyle(stroke='rgb(0, 255, 100)', stroke_width=2, opacity=0.7)
        for i in range(n_snake_points):
            next_i = (i + 1) % n_snake_points
            topology.add_line(
                name=f"snake_connection_{i}_cleaned",
                point_a=f"snake_contour_cleaned_{i}",
                point_b=f"snake_contour_cleaned_{next_i}",
                style=snake_line_style
            )

    if show_snake and show_raw:
        # Snake points (yellow)
        snake_point_style = PointStyle(radius=4, fill='rgb(255, 255, 0)', opacity=0.9)

        for i in range(n_snake_points):
            topology.add_point(
                name=f"snake_point_{i}_raw",
                point_name=f"snake_contour_raw_{i}",
                style=snake_point_style
            )

        # Connect snake points in a loop
        snake_line_style = LineStyle(stroke='rgb(255, 255, 0)', stroke_width=2, opacity=0.7)
        for i in range(n_snake_points):
            next_i = (i + 1) % n_snake_points
            topology.add_line(
                name=f"snake_connection_{i}_raw",
                point_a=f"snake_contour_raw_{i}",
                point_b=f"snake_contour_raw_{next_i}",
                style=snake_line_style
            )

    # === PUPIL CENTERS ===

    if show_cleaned:
        topology.add_circle(
            name="pupil_center_circle_cleaned",
            center_point="pupil_center_cleaned",
            radius=5,
            style=PointStyle(fill='rgb(255, 250, 0)')
        )

        topology.add_crosshair(
            name="pupil_center_crosshair_cleaned",
            center_point="pupil_center_cleaned",
            size=10,
            style=LineStyle(stroke='rgb(255, 250, 0)', stroke_width=2)
        )

    if show_raw:
        topology.add_circle(
            name="pupil_center_circle_raw",
            center_point="pupil_center_raw",
            radius=5,
            style=PointStyle(fill='rgb(255, 200, 0)')
        )

        topology.add_crosshair(
            name="pupil_center_crosshair_raw",
            center_point="pupil_center_raw",
            size=10,
            style=LineStyle(stroke='rgb(255, 200, 0)', stroke_width=2)
        )

    # === FRAME INFO ===
    topology.add_computed_point(
        name="info_corner",
        computation=lambda pts: np.array([10.0, 25.0]),
        description="Top-left corner for info text"
    )

    def format_frame_info(metadata: dict) -> str:
        frame_idx = metadata.get('frame_idx', 0)
        total_frames = metadata.get('total_frames', 0)
        view_mode = metadata.get('view_mode', 'unknown')
        snake_status = metadata.get('snake_status', '')
        status_str = f" | {snake_status}" if snake_status else ""
        return f"Frame: {frame_idx}/{total_frames} | Mode: {view_mode.upper()}{status_str}"

    topology.add_text(
        name="frame_info",
        point_name="info_corner",
        text=format_frame_info,
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

    return topology


# ==================== SVG VIEWER ====================

class SVGEyeTrackingViewer:
    """Eye tracking viewer using SVG topology system with raw/cleaned/snake toggle."""

    def __init__(
        self,
        *,
        dataset: EyeVideoDataset,
        window_name: str = "SVG Eye Tracking Viewer",
        initial_view_mode: ViewMode = ViewMode.CLEANED,
        enable_dots: bool = True,
        enable_ellipse: bool = True,
        enable_snake: bool = True,
        skip_frames: int = 0
    ) -> None:
        """Initialize viewer with dataset and SVG topology.

        Args:
            dataset: Eye tracking dataset
            window_name: Display window name
            initial_view_mode: Starting view mode (raw, cleaned, or both)
            enable_dots: Whether to show landmark dots
            enable_ellipse: Whether to show fitted ellipses
            enable_snake: Whether to enable snake contour fitting
            skip_frames: Number of frames to skip per iteration (0 = show every frame)
        """
        self.dataset: EyeVideoDataset = dataset
        self.window_name: str = window_name
        self.view_mode: ViewMode = initial_view_mode
        self.enable_dots: bool = enable_dots
        self.enable_ellipse: bool = enable_ellipse
        self.enable_snake: bool = enable_snake
        self.skip_frames: int = skip_frames

        # Get trajectory data arrays
        self.raw_trajectories: np.ndarray = self.dataset.pixel_trajectories.to_array(use_cleaned=False)
        self.cleaned_trajectories: np.ndarray = self.dataset.pixel_trajectories.to_array(use_cleaned=True)

        # Create initial renderer
        self.renderer: SVGOverlayRenderer = self._create_renderer()

    def _create_renderer(self) -> SVGOverlayRenderer:
        """Create renderer based on current view mode."""
        show_raw = self.view_mode in [ViewMode.RAW, ViewMode.BOTH]
        show_cleaned = self.view_mode in [ViewMode.CLEANED, ViewMode.BOTH]

        topology = create_full_eye_topology(
            width=self.dataset.video.width,
            height=self.dataset.video.height,
            show_raw=show_raw,
            show_cleaned=show_cleaned,
            show_dots=self.enable_dots,
            show_ellipse=self.enable_ellipse,
            show_snake=self.enable_snake,
            n_snake_points=self.dataset.snake_params.n_points
        )

        return SVGOverlayRenderer(topology=topology)

    def set_view_mode(self, *, mode: ViewMode) -> None:
        """Change view mode and recreate renderer.

        Args:
            mode: New view mode
        """
        if mode != self.view_mode:
            self.view_mode = mode
            self.renderer = self._create_renderer()
            print(f"View mode: {mode.value.upper()}")

    def toggle_snake(self) -> None:
        """Toggle snake contour visualization."""
        self.enable_snake = not self.enable_snake
        self.renderer = self._create_renderer()
        print(f"Snake contours: {'ENABLED' if self.enable_snake else 'DISABLED'}")

    def toggle_dots(self) -> None:
        """Toggle landmark dots visualization."""
        self.enable_dots = not self.enable_dots
        self.renderer = self._create_renderer()
        print(f"Landmark dots: {'ENABLED' if self.enable_dots else 'DISABLED'}")

    def toggle_ellipse(self) -> None:
        """Toggle fitted ellipse visualization."""
        self.enable_ellipse = not self.enable_ellipse
        self.renderer = self._create_renderer()
        print(f"Fitted ellipse: {'ENABLED' if self.enable_ellipse else 'DISABLED'}")

    def increment_skip_frames(self) -> None:
        """Increment frame skip count."""
        self.skip_frames += 1
        print(f"Skip frames: {self.skip_frames}")

    def decrement_skip_frames(self) -> None:
        """Decrement frame skip count (minimum 0)."""
        self.skip_frames = max(0, self.skip_frames - 1)
        print(f"Skip frames: {self.skip_frames}")

    def _compute_snake_contour(
        self,
        *,
        frame: np.ndarray,
        ellipse_params: np.ndarray
    ) -> SnakeContour | None:
        """Compute snake contour from ellipse parameters and image.

        Args:
            frame: Video frame
            ellipse_params: Ellipse parameters [cx, cy, a, b, theta]

        Returns:
            SnakeContour if successful, None otherwise
        """
        return fit_snake_safe(
            image=frame,
            initial_ellipse_params=ellipse_params,
            params=self.dataset.snake_params
        )

    def get_frame_points(
        self,
        *,
        frame_idx: int,
        frame: np.ndarray | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Get all named points for a specific frame, including snake contours.

        Args:
            frame_idx: Frame index
            frame: Video frame (required if snake fitting is enabled)

        Returns:
            Tuple of (points dict, metadata dict)
        """
        points = {}
        metadata = {}

        # Add raw points if showing raw data
        if self.view_mode in [ViewMode.RAW, ViewMode.BOTH]:
            landmarks_array = self.raw_trajectories[frame_idx]
            for name, idx in self.dataset.landmarks.items():
                points[f"{name}_raw"] = landmarks_array[idx]

        # Add cleaned points if showing cleaned data
        if self.view_mode in [ViewMode.CLEANED, ViewMode.BOTH]:
            landmarks_array = self.cleaned_trajectories[frame_idx]
            for name, idx in self.dataset.landmarks.items():
                points[f"{name}_cleaned"] = landmarks_array[idx]

        # Compute and add snake contours if enabled
        snake_status_parts = []

        if self.enable_snake and frame is not None:
            # Fit snake to cleaned data
            if self.view_mode in [ViewMode.CLEANED, ViewMode.BOTH]:
                try:
                    # Get ellipse parameters
                    pupil_points = np.array([points[f"p{i}_cleaned"] for i in range(1, 9)])
                    ellipse_params = fit_ellipse_to_points(points=pupil_points)

                    # Fit snake
                    snake_contour = self._compute_snake_contour(
                        frame=frame,
                        ellipse_params=ellipse_params.to_array()
                    )

                    if snake_contour is not None:
                        # Add snake points
                        for i, pt in enumerate(snake_contour.points):
                            points[f"snake_contour_cleaned_{i}"] = pt
                        snake_status_parts.append("C-Snake:OK")
                    else:
                        snake_status_parts.append("C-Snake:FAIL")

                except (ValueError, cv2.error):
                    snake_status_parts.append("C-Snake:FAIL")

            # Fit snake to raw data
            if self.view_mode in [ViewMode.RAW, ViewMode.BOTH]:
                try:
                    # Get ellipse parameters
                    pupil_points = np.array([points[f"p{i}_raw"] for i in range(1, 9)])
                    ellipse_params = fit_ellipse_to_points(points=pupil_points)

                    # Fit snake
                    snake_contour = self._compute_snake_contour(
                        frame=frame,
                        ellipse_params=ellipse_params.to_array()
                    )

                    if snake_contour is not None:
                        # Add snake points
                        for i, pt in enumerate(snake_contour.points):
                            points[f"snake_contour_raw_{i}"] = pt
                        snake_status_parts.append("R-Snake:OK")
                    else:
                        snake_status_parts.append("R-Snake:FAIL")

                except (ValueError, cv2.error):
                    snake_status_parts.append("R-Snake:FAIL")

        metadata['snake_status'] = ' | '.join(snake_status_parts) if snake_status_parts else ''

        return points, metadata

    def render_frame_overlay(
        self,
        *,
        frame: np.ndarray,
        frame_idx: int
    ) -> np.ndarray:
        """Render overlay on a frame using SVG topology.

        Args:
            frame: Video frame (BGR numpy array)
            frame_idx: Current frame index

        Returns:
            Frame with SVG overlay composited
        """
        points, snake_metadata = self.get_frame_points(frame_idx=frame_idx, frame=frame)

        metadata = {
            'frame_idx': frame_idx,
            'total_frames': len(self.raw_trajectories) - 1,
            'dataset_name': self.dataset.data_name,
            'view_mode': self.view_mode.value,
            **snake_metadata
        }

        return self.renderer.render_and_composite(
            image=frame,
            points=points,
            metadata=metadata,
        )

    def save_frame_svg(
        self,
        *,
        frame_idx: int,
        frame: np.ndarray,
        output_path: Path
    ) -> None:
        """Save frame overlay as standalone SVG file.

        Args:
            frame_idx: Frame index
            frame: Video frame (needed for snake fitting)
            output_path: Output SVG file path
        """
        points, snake_metadata = self.get_frame_points(frame_idx=frame_idx, frame=frame)
        metadata = {
            'frame_idx': frame_idx,
            'total_frames': len(self.raw_trajectories) - 1,
            'view_mode': self.view_mode.value,
            **snake_metadata
        }

        svg = self.renderer.render_svg(points=points, metadata=metadata)
        self.renderer.save_svg(svg_drawing=svg, filepath=output_path)

    def run(self, *, start_frame: int = 0, save_dir: Path | None = None) -> None:
        """Run the interactive viewer.

        Controls:
            - Space: Pause/Resume
            - 's': Save current frame (PNG)
            - 'v': Save current frame as SVG
            - 'r': Show RAW data only
            - 'c': Show CLEANED data only
            - 'b': Show BOTH raw and cleaned
            - 'd': Toggle DOTS (landmark points)
            - 'e': Toggle ELLIPSE (fitted ellipse)
            - 'n': Toggle SNAKE contours
            - '+/=': Increase frame skip
            - '-': Decrease frame skip
            - 'q' or ESC: Quit
            - Right Arrow: Next frame (when paused)
            - Left Arrow: Previous frame (when paused)

        Args:
            start_frame: Frame to start viewing from
            save_dir: Optional directory to save frames to
        """
        if self.dataset.video.video_capture is None:
            raise ValueError("Video capture is not initialized")

        cv2.namedWindow(winname=self.window_name)

        current_frame: int = start_frame
        paused: bool = False

        self.dataset.video.video_capture.set(
            propId=cv2.CAP_PROP_POS_FRAMES,
            value=current_frame
        )

        print("\nControls:")
        print("  Space: Pause/Resume")
        print("  's': Save current frame as PNG")
        print("  'v': Save current frame as SVG")
        print("  'r': Show RAW data only (orange ellipse, yellow snake)")
        print("  'c': Show CLEANED data only (magenta ellipse, green snake)")
        print("  'b': Show BOTH raw and cleaned")
        print("  'd': Toggle DOTS (landmark points)")
        print("  'e': Toggle ELLIPSE (fitted ellipse)")
        print("  'n': Toggle SNAKE contours")
        print("  '+/=': Increase frame skip")
        print("  '-': Decrease frame skip")
        print("  'q' or ESC: Quit")
        print("  Right Arrow: Next frame (when paused)")
        print("  Left Arrow: Previous frame (when paused)")
        print(f"\nCurrent view mode: {self.view_mode.value.upper()}")
        print(f"Landmark dots: {'ENABLED' if self.enable_dots else 'DISABLED'}")
        print(f"Fitted ellipse: {'ENABLED' if self.enable_ellipse else 'DISABLED'}")
        print(f"Snake contours: {'ENABLED' if self.enable_snake else 'DISABLED'}")
        print(f"Skip frames: {self.skip_frames}")
        print()

        while True:
            if not paused:
                ret, frame = self.dataset.video.video_capture.read()
                if not ret:
                    print("End of video reached")
                    break
                current_frame = int(
                    self.dataset.video.video_capture.get(propId=cv2.CAP_PROP_POS_FRAMES)
                ) - 1

                # Skip frames if configured
                if self.skip_frames > 0:
                    for _ in range(self.skip_frames):
                        ret, _ = self.dataset.video.video_capture.read()
                        if not ret:
                            print("End of video reached")
                            break
                    if not ret:
                        break
                    current_frame = int(
                        self.dataset.video.video_capture.get(propId=cv2.CAP_PROP_POS_FRAMES)
                    ) - 1
            else:
                self.dataset.video.video_capture.set(
                    propId=cv2.CAP_PROP_POS_FRAMES,
                    value=current_frame
                )
                ret, frame = self.dataset.video.video_capture.read()
                if not ret:
                    break

            # Resize frame if needed
            if self.dataset.video.resize_factor != 1.0:
                new_width: int = int(frame.shape[1] * self.dataset.video.resize_factor)
                new_height: int = int(frame.shape[0] * self.dataset.video.resize_factor)
                frame = cv2.resize(
                    src=frame,
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_LINEAR
                )

            # Render SVG overlay
            overlay_frame: np.ndarray = self.render_frame_overlay(
                frame=frame,
                frame_idx=current_frame
            )

            # Display
            cv2.imshow(winname=self.window_name, mat=overlay_frame)

            # Handle keyboard input
            key: int = cv2.waitKey(delay=30 if not paused else 0) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'} at frame {current_frame}")
            elif key == ord('r'):
                self.set_view_mode(mode=ViewMode.RAW)
            elif key == ord('c'):
                self.set_view_mode(mode=ViewMode.CLEANED)
            elif key == ord('b'):
                self.set_view_mode(mode=ViewMode.BOTH)
            elif key == ord('d'):
                self.toggle_dots()
            elif key == ord('e'):
                self.toggle_ellipse()
            elif key == ord('n'):
                self.toggle_snake()
            elif key == ord('+') or key == ord('='):
                self.increment_skip_frames()
            elif key == ord('-') or key == ord('_'):
                self.decrement_skip_frames()
            elif key == ord('s'):
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    output_path = save_dir / f"frame_{current_frame:06d}.png"
                    cv2.imwrite(filename=str(output_path), img=overlay_frame)
                    print(f"Saved PNG: {output_path}")
                else:
                    print("No save directory specified")
            elif key == ord('v'):
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    output_path = save_dir / f"frame_{current_frame:06d}.svg"
                    self.save_frame_svg(frame_idx=current_frame, frame=frame, output_path=output_path)
                    print(f"Saved SVG: {output_path}")
                else:
                    print("No save directory specified")
            elif paused:
                if key == 83:  # Right arrow
                    current_frame = min(current_frame + 1 + self.skip_frames, len(self.raw_trajectories) - 1)
                elif key == 81:  # Left arrow
                    current_frame = max(current_frame - 1 - self.skip_frames, 0)

        cv2.destroyAllWindows()