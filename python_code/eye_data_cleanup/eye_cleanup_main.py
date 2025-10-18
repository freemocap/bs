"""Integration of SVG topology system with eye tracking viewer.

Shows how to define reusable topologies and integrate with existing cv2 workflow.
"""

from pathlib import Path

import cv2
import numpy as np

from csv_io import load_trajectory_csv, ABaseModel, TrajectoryDataset
from svg_overlay import (
    SVGTopology, SVGOverlayRenderer,
    PointStyle, LineStyle, TextStyle
)

DEFAULT_RESIZE_FACTOR: float = 1.0
DEFAULT_MIN_CONFIDENCE: float = 0.3


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

    # Define connections between landmarks (for visualization)
    connections: tuple[tuple[int, int], ...] = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 7), (7, 0),
        (8, 9), (8, 0),
    )

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
        min_confidence: float = DEFAULT_MIN_CONFIDENCE
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
            )
        )

    def get_pupil_centers(self) -> np.ndarray:
        """Calculate mean position of pupil points (p1-p8) over time."""
        pupil_trajectories = []
        for landmark_name in ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]:
            if landmark_name in self.pixel_trajectories.marker_names:
                pupil_trajectories.append(self.pixel_trajectories.data[landmark_name].data)

        if not pupil_trajectories:
            raise ValueError("No pupil landmarks found in dataset")

        stacked = np.stack(pupil_trajectories, axis=1)
        with np.errstate(invalid='ignore'):
            return np.nanmean(stacked, axis=1)


# ==================== TOPOLOGY DEFINITIONS ====================

def create_simple_pupil_topology(*, width: int, height: int) -> SVGTopology:
    """Create a simple topology: just pupil points and center.

    Required points: p1-p8
    Computed points: pupil_center (mean of p1-p8)
    """
    topology = SVGTopology(
        name="simple_pupil",
        width=width,
        height=height,
        required_points=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
    )

    # Add computed pupil center
    def compute_pupil_center(points: dict[str, np.ndarray]) -> np.ndarray:
        pupil_points = [points[f"p{i}"] for i in range(1, 9) if f"p{i}" in points]
        stacked = np.stack(pupil_points, axis=0)
        return np.nanmean(stacked, axis=0)

    topology.add_computed_point(
        name="pupil_center",
        computation=compute_pupil_center,
        description="Mean of pupil points p1-p8"
    )

    # Add pupil points (green circles with labels)
    point_style = PointStyle(radius=3, fill='rgb(0, 255, 0)')
    label_style = TextStyle(
        font_size=10,
        fill='lime',
        stroke='black',
        stroke_width=1
    )

    for i in range(1, 9):
        topology.add_point(
            name=f"pupil_point_{i}",
            point_name=f"p{i}",
            style=point_style,
            label=f"p{i}",
            label_offset=(5, -5)
        )

    # Add pupil center (yellow circle with crosshair)
    topology.add_circle(
        name="pupil_center_circle",
        center_point="pupil_center",
        radius=5,
        style=PointStyle(fill='rgb(255, 250, 0)')
    )

    topology.add_crosshair(
        name="pupil_center_crosshair",
        center_point="pupil_center",
        size=10,
        style=LineStyle(stroke='rgb(255, 250, 0)', stroke_width=2)
    )

    # Add label
    topology.add_text(
        name="pupil_center_label",
        point_name="pupil_center",
        text="Pupil Center",
        offset=(0, -15),
        style=TextStyle(
            font_size=14,
            fill='yellow',
            stroke='black',
            stroke_width=2,
            text_anchor='middle'
        )
    )

    return topology


def create_full_eye_topology(*, width: int, height: int) -> SVGTopology:
    """Create full eye tracking topology with all features.

    Required points: p1-p8, tear_duct, outer_eye
    Computed points: pupil_center
    Elements: connections, points, labels, info overlay
    """
    topology = SVGTopology(
        name="full_eye_tracking",
        width=width,
        height=height,
        required_points=[
            "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
            "tear_duct", "outer_eye"
        ]
    )

    # Computed pupil center
    def compute_pupil_center(points: dict[str, np.ndarray]) -> np.ndarray:
        pupil_points = [points[f"p{i}"] for i in range(1, 9) if f"p{i}" in points]
        if not pupil_points:
            return np.array([np.nan, np.nan])
        stacked = np.stack(pupil_points, axis=0)
        return np.nanmean(stacked, axis=0)

    topology.add_computed_point(
        name="pupil_center",
        computation=compute_pupil_center,
        description="Mean of pupil points"
    )

    # === CONNECTION LINES (drawn first, so they're behind points) ===
    line_style = LineStyle(stroke='rgb(255, 55, 55)', stroke_width=2)

    # Pupil outline (closed loop)
    connections = [
        ("p1", "p2"), ("p2", "p3"), ("p3", "p4"), ("p4", "p5"),
        ("p5", "p6"), ("p6", "p7"), ("p7", "p8"), ("p8", "p1")
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
        point_a="tear_duct",
        point_b="outer_eye",
        style=line_style
    )
    topology.add_line(
        name="tear_to_pupil",
        point_a="tear_duct",
        point_b="p1",
        style=line_style
    )

    # === LANDMARK POINTS ===
    landmark_style = PointStyle(radius=3, fill='rgb(0, 255, 0)')
    label_style = TextStyle(
        font_size=10,
        fill='lime',
        stroke='black',
        stroke_width=1
    )

    # Pupil points
    for i in range(1, 9):
        topology.add_point(
            name=f"pupil_point_{i}",
            point_name=f"p{i}",
            style=landmark_style,
            label=f"p{i}",
            label_offset=(5, -5)
        )

    # Eye corners
    corner_style = PointStyle(radius=4, fill='rgb(0, 200, 255)')
    for name in ["tear_duct", "outer_eye"]:
        topology.add_point(
            name=f"{name}_point",
            point_name=name,
            style=corner_style,
            label=name.replace('_', ' ').title(),
            label_offset=(5, -5)
        )

    # === PUPIL CENTER ===
    topology.add_circle(
        name="pupil_center_circle",
        center_point="pupil_center",
        radius=5,
        style=PointStyle(fill='rgb(255, 250, 0)')
    )

    topology.add_crosshair(
        name="pupil_center_crosshair",
        center_point="pupil_center",
        size=10,
        style=LineStyle(stroke='rgb(255, 250, 0)', stroke_width=2)
    )

    # topology.add_text(
    #     name="pupil_center_label",
    #     point_name="pupil_center",
    #     text="Pupil Center",
    #     offset=(0, -15),
    #     style=TextStyle(
    #         font_size=14,
    #         fill='yellow',
    #         stroke='black',
    #         stroke_width=2,
    #         text_anchor='middle',
    #         font_weight='bold'
    #     )
    # )

    # === FRAME INFO (top-left corner) ===
    # Use a fixed point for frame info
    topology.add_computed_point(
        name="info_corner",
        computation=lambda pts: np.array([10.0, 25.0]),
        description="Top-left corner for info text"
    )

    # Dynamic frame info text
    def format_frame_info(metadata: dict) -> str:
        frame_idx = metadata.get('frame_idx', 0)
        total_frames = metadata.get('total_frames', 0)
        return f"Frame: {frame_idx}/{total_frames}"

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
    """Eye tracking viewer using SVG topology system."""

    def __init__(
        self,
        *,
        dataset: EyeVideoDataset,
        topology: SVGTopology,
        window_name: str = "SVG Eye Tracking Viewer"
    ) -> None:
        """Initialize viewer with dataset and SVG topology.

        Args:
            dataset: Eye tracking dataset
            topology: SVG topology defining the overlay structure
            window_name: Display window name
        """
        self.dataset: EyeVideoDataset = dataset
        self.window_name: str = window_name

        # Get all trajectory data
        self.trajectories: np.ndarray = self.dataset.pixel_trajectories.to_array()
        self.pupil_centers: np.ndarray = self.dataset.get_pupil_centers()

        # Create renderer with topology
        self.renderer: SVGOverlayRenderer = SVGOverlayRenderer(topology=topology)

    def get_frame_points(self, *, frame_idx: int) -> dict[str, np.ndarray]:
        """Get all named points for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Dictionary mapping point names to (x, y) coordinates
        """
        points = {}

        # Get landmarks from dataset
        landmarks_array = self.trajectories[frame_idx]
        for name, idx in self.dataset.landmarks.items():
            points[name] = landmarks_array[idx]

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
        # Get point coordinates for this frame
        points = self.get_frame_points(frame_idx=frame_idx)

        # Metadata for dynamic content
        metadata = {
            'frame_idx': frame_idx,
            'total_frames': len(self.trajectories) - 1,
            'dataset_name': self.dataset.data_name
        }

        # Render and composite (using Pillow - Windows compatible!)
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

        Open the SVG in a browser to inspect the structure!
        Perfect for learning SVG before moving to D3.

        Args:
            frame_idx: Frame index
            output_path: Output SVG file path
        """
        points = self.get_frame_points(frame_idx=frame_idx)
        metadata = {
            'frame_idx': frame_idx,
            'total_frames': len(self.trajectories) - 1
        }

        svg = self.renderer.render_svg(points=points, metadata=metadata)
        self.renderer.save_svg(svg_drawing=svg, filepath=output_path)

    def run(self, *, start_frame: int = 0, save_dir: Path | None = None) -> None:
        """Run the interactive viewer.

        Controls:
            - Space: Pause/Resume
            - 's': Save current frame (PNG)
            - 'v': Save current frame as SVG
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
        print("  'q' or ESC: Quit")
        print("  Right Arrow: Next frame (when paused)")
        print("  Left Arrow: Previous frame (when paused)")
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
                    current_frame = min(current_frame + 1, len(self.trajectories) - 1)
                elif key == 81:  # Left arrow
                    current_frame = max(current_frame - 1, 0)

        cv2.destroyAllWindows()


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Setup paths (same as your original)
    base_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37"
    )
    video_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4"
    )
    timestamps_npy_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy"
    )
    csv_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )

    # Create dataset
    eye_dataset: EyeVideoDataset = EyeVideoDataset.create(
        data_name="ferret_757_eye_tracking",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
    )


    # Option 2: Full topology (all features)
    topology = create_full_eye_topology(
        width=eye_dataset.video.width,
        height=eye_dataset.video.height
    )

    # Create viewer with SVG topology
    viewer: SVGEyeTrackingViewer = SVGEyeTrackingViewer(
        dataset=eye_dataset,
        topology=topology,
        window_name="SVG Pupil Tracking"
    )

    # Optional save directory
    save_directory: Path | None = Path("./saved_frames_svg")

    # Run viewer
    viewer.run(start_frame=0, save_dir=save_directory)
