"""Integration of SVG topology system with eye tracking viewer.

Shows how to define reusable topologies and integrate with existing cv2 workflow.
Supports toggling between raw, cleaned, and both data visualizations.
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
        butterworth_sampling_rate: float = 30.0
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
            )
        )


# ==================== TOPOLOGY DEFINITIONS ====================

def create_full_eye_topology(
    *,
    width: int,
    height: int,
    show_raw: bool = True,
    show_cleaned: bool = True
) -> SVGTopology:
    """Create full eye tracking topology with configurable raw/cleaned display.

    Required points: p1-p8, tear_duct, outer_eye (with _raw and/or _cleaned suffixes)
    Computed points: pupil_center_raw, pupil_center_cleaned
    Elements: connections (cleaned only), points, labels, info overlay

    Args:
        width: Video width
        height: Video height
        show_raw: Whether to show raw data
        show_cleaned: Whether to show cleaned data

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
    if show_cleaned:
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
    if show_raw:
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
        return f"Frame: {frame_idx}/{total_frames} | Mode: {view_mode.upper()}"

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
    """Eye tracking viewer using SVG topology system with raw/cleaned toggle."""

    def __init__(
        self,
        *,
        dataset: EyeVideoDataset,
        window_name: str = "SVG Eye Tracking Viewer",
        initial_view_mode: ViewMode = ViewMode.CLEANED
    ) -> None:
        """Initialize viewer with dataset and SVG topology.

        Args:
            dataset: Eye tracking dataset
            window_name: Display window name
            initial_view_mode: Starting view mode (raw, cleaned, or both)
        """
        self.dataset: EyeVideoDataset = dataset
        self.window_name: str = window_name
        self.view_mode: ViewMode = initial_view_mode

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
            show_cleaned=show_cleaned
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

    def get_frame_points(self, *, frame_idx: int) -> dict[str, np.ndarray]:
        """Get all named points for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Dictionary mapping point names to (x, y) coordinates
        """
        points = {}

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

        return points

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
        points = self.get_frame_points(frame_idx=frame_idx)

        metadata = {
            'frame_idx': frame_idx,
            'total_frames': len(self.raw_trajectories) - 1,
            'dataset_name': self.dataset.data_name,
            'view_mode': self.view_mode.value
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
        output_path: Path
    ) -> None:
        """Save frame overlay as standalone SVG file.

        Args:
            frame_idx: Frame index
            output_path: Output SVG file path
        """
        points = self.get_frame_points(frame_idx=frame_idx)
        metadata = {
            'frame_idx': frame_idx,
            'total_frames': len(self.raw_trajectories) - 1,
            'view_mode': self.view_mode.value
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
        print("  'r': Show RAW data only")
        print("  'c': Show CLEANED data only")
        print("  'b': Show BOTH raw and cleaned")
        print("  'q' or ESC: Quit")
        print("  Right Arrow: Next frame (when paused)")
        print("  Left Arrow: Previous frame (when paused)")
        print(f"\nCurrent view mode: {self.view_mode.value.upper()}")
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
                    self.save_frame_svg(frame_idx=current_frame, output_path=output_path)
                    print(f"Saved SVG: {output_path}")
                else:
                    print("No save directory specified")
            elif paused:
                if key == 83:  # Right arrow
                    current_frame = min(current_frame + 1, len(self.raw_trajectories) - 1)
                elif key == 81:  # Left arrow
                    current_frame = max(current_frame - 1, 0)

        cv2.destroyAllWindows()