from pathlib import Path

import cv2
import numpy as np

from python_code.data_loaders.trajectory_loader.trajectory_csv_io import load_trajectory_csv
from python_code.data_loaders.trajectory_loader.trajectory_dataset import (
    ABaseModel,
    TrajectoryND,
    TrajectoryDataset,
    TrajectoryType,
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
        """
        Create a VideoHelper instance from a video file.

        Args:
            video_path: Path to the video file
            timestamps_npy_path: Optional path to numpy array of timestamps
            resize_factor: Factor to resize video frames

        Returns:
            VideoHelper instance

        Raises:
            FileNotFoundError: If video file doesn't exist
            IOError: If video cannot be opened
        """
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


class RerunVideoDataset(ABaseModel):
    """Base model representing video data and associated tracking."""

    data_name: str
    base_path: Path
    video: VideoHelper
    pixel_trajectories: TrajectoryDataset

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
    ) -> "RerunVideoDataset":
        """
        Create a RerunVideoDataset instance.

        Args:
            data_name: Descriptive name for this dataset
            base_path: Base directory path
            raw_video_path: Path to video file
            timestamps_npy_path: Path to timestamps numpy array
            data_csv_path: Path to trajectory CSV data
            resize_factor: Video resize factor
            min_confidence: Minimum confidence threshold for trajectories

        Returns:
            RerunVideoDataset instance
        """
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
                trajectory_type=TrajectoryType.POSITION_2D,
            )
        )


class EyeVideoDataset(RerunVideoDataset):
    """Dataset for eye tracking video with pupil landmarks."""

    # Define the landmark indices for pupil points
    landmarks: dict[str, int] = {
        "p1": 0,
        "p2": 1,
        "p3": 2,
        "p4": 3,
        "p5": 4,
        "p6": 5,
        "p7": 6,
        "p8": 7,
        "tear_duct": 8,
        "outer_eye": 9,
    }

    # Define connections between landmarks (for visualization)
    connections: tuple[tuple[int, int], ...] = (
        # Pupil outline (closed loop)
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 7), (7, 0),
        # Additional connections
        (8, 9),  # tear duct to outer eye
        (8, 0),  # tear duct to p1
    )

    @property
    def pupil_mean_x(self) -> TrajectoryND:
        """
        Calculate mean x-coordinate of pupil points (p1-p8).

        Returns:
            1D array of mean x coordinates over time
        """
        pupil_points_x: list[np.ndarray] = []
        for landmark_name in self.landmarks:
            if "p" in landmark_name and landmark_name != "tear_duct":
                point_idx: int = self.landmarks[landmark_name]
                pupil_points_x.append(self.pixel_trajectories.to_array()[:, point_idx, 0])
        return np.nanmean(a=np.array(pupil_points_x), axis=0)

    @property
    def pupil_mean_y(self) -> TrajectoryND:
        """
        Calculate mean y-coordinate of pupil points (p1-p8).

        Returns:
            1D array of mean y coordinates over time
        """
        pupil_points_y: list[np.ndarray] = []
        for landmark_name in self.landmarks:
            if "p" in landmark_name and landmark_name != "tear_duct":
                point_idx: int = self.landmarks[landmark_name]
                pupil_points_y.append(self.pixel_trajectories.to_array()[:, point_idx, 1])
        return np.nanmean(a=np.array(pupil_points_y), axis=0)





class EyeTrackingViewer:
    """Interactive viewer for eye tracking data with pupil overlay."""

    def __init__(
            self,
            *,
            dataset: EyeVideoDataset,
            window_name: str = "Eye Tracking Viewer",
            point_radius: int = 3,
            center_radius: int = 5,
            line_thickness: int = 2,
            point_color: tuple[int, int, int] = (0, 255, 0),
            center_color: tuple[int, int, int] = (255, 250, 0),
            line_color: tuple[int, int, int] = (255, 55, 55),
            text_color: tuple[int, int, int] = (255, 0, 255),
    ) -> None:
        """
        Initialize the eye tracking viewer.

        Args:
            dataset: EyeVideoDataset containing video and tracking data
            window_name: Name of the display window
            point_radius: Radius for landmark points
            center_radius: Radius for pupil center point
            line_thickness: Thickness of connection lines
            point_color: BGR color for landmark points
            center_color: BGR color for pupil center
            line_color: BGR color for connection lines
            text_color: BGR color for text labels
        """
        self.dataset: EyeVideoDataset = dataset
        self.window_name: str = window_name
        self.point_radius: int = point_radius
        self.center_radius: int = center_radius
        self.line_thickness: int = line_thickness
        self.point_color: tuple[int, int, int] = point_color
        self.center_color: tuple[int, int, int] = center_color
        self.line_color: tuple[int, int, int] = line_color
        self.text_color: tuple[int, int, int] = text_color

        # Get pupil centers for all frames
        self.pupil_centers_x: np.ndarray = self.dataset.pupil_mean_x.data
        self.pupil_centers_y: np.ndarray = self.dataset.pupil_mean_y.data

        # Get all trajectory data as array (n_frames, n_landmarks, 2)
        self.trajectories: np.ndarray = self.dataset.pixel_trajectories.to_array()

    def draw_frame_overlay(
            self,
            *,
            frame: np.ndarray,
            frame_idx: int
    ) -> np.ndarray:
        """
        Draw pupil tracking overlay on a frame.

        Args:
            frame: Video frame to draw on
            frame_idx: Current frame index

        Returns:
            Frame with overlay drawn
        """
        # Make a copy to avoid modifying original
        overlay: np.ndarray = frame.copy()

        # Get landmarks for this frame (n_landmarks, 2)
        landmarks: np.ndarray = self.trajectories[frame_idx]

        # Draw connections between landmarks
        for connection in self.dataset.connections:
            pt1_idx, pt2_idx = connection
            pt1: np.ndarray = landmarks[pt1_idx]
            pt2: np.ndarray = landmarks[pt2_idx]

            # Only draw if both points are valid
            if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                cv2.line(
                    img=overlay,
                    pt1=(int(pt1[0]), int(pt1[1])),
                    pt2=(int(pt2[0]), int(pt2[1])),
                    color=self.line_color,
                    thickness=self.line_thickness
                )

        # Draw landmark points
        for landmark_name, landmark_idx in self.dataset.landmarks.items():
            point: np.ndarray = landmarks[landmark_idx]

            if not np.isnan(point).any():
                x, y = int(point[0]), int(point[1])

                # Draw point
                cv2.circle(
                    img=overlay,
                    center=(x, y),
                    radius=self.point_radius,
                    color=self.point_color,
                    thickness=-1  # Filled circle
                )

                # Draw label for key points
                # if landmark_name in ["tear_duct", "outer_eye", "p1", "p5", "p8"]:
                cv2.putText(
                        img=overlay,
                        text=landmark_name,
                        org=(x + 5, y - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=self.text_color,
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )

        # Draw pupil center
        center_x: float = self.pupil_centers_x[frame_idx]
        center_y: float = self.pupil_centers_y[frame_idx]

        if not (np.isnan(center_x) or np.isnan(center_y)):
            cx, cy = int(center_x), int(center_y)

            # Draw center point
            cv2.circle(
                img=overlay,
                center=(cx, cy),
                radius=self.center_radius,
                color=self.center_color,
                thickness=-1
            )

            # Draw crosshair
            crosshair_size: int = 10
            cv2.line(
                img=overlay,
                pt1=(cx - crosshair_size, cy),
                pt2=(cx + crosshair_size, cy),
                color=self.center_color,
                thickness=2
            )
            cv2.line(
                img=overlay,
                pt1=(cx, cy - crosshair_size),
                pt2=(cx, cy + crosshair_size),
                color=self.center_color,
                thickness=2
            )

            # Label pupil center
            cv2.putText(
                img=overlay,
                text="Pupil Center",
                org=(cx + 10, cy + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=self.text_color,
                thickness=1,
                lineType=cv2.LINE_AA
            )

        # Add frame info
        info_text: str = f"Frame: {frame_idx}/{len(self.trajectories) - 1}"
        cv2.putText(
            img=overlay,
            text=info_text,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=self.text_color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        return overlay

    def save_frame(
            self,
            *,
            frame: np.ndarray,
            frame_idx: int,
            output_dir: Path
    ) -> None:
        """
        Save a frame with overlay to disk.

        Args:
            frame: Frame to save
            frame_idx: Current frame index
            output_dir: Directory to save frames to
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path: Path = output_dir / f"frame_{frame_idx:06d}.png"

        cv2.imwrite(
            filename=str(output_path),
            img=frame
        )
        print(f"Saved frame to: {output_path}")

    def run(
            self,
            *,
            start_frame: int = 0,
            save_dir: Path | None = None
    ) -> None:
        """
        Run the interactive viewer.

        Controls:
            - Space: Pause/Resume
            - 's': Save current frame
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

        # Set video to start frame
        self.dataset.video.video_capture.set(
            propId=cv2.CAP_PROP_POS_FRAMES,
            value=current_frame
        )

        print("\nControls:")
        print("  Space: Pause/Resume")
        print("  's': Save current frame")
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
                # When paused, read the current frame
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

            # Draw overlay
            overlay_frame: np.ndarray = self.draw_frame_overlay(
                frame=frame,
                frame_idx=current_frame
            )

            # Display frame
            cv2.imshow(winname=self.window_name, mat=overlay_frame)

            # Handle keyboard input
            key: int = cv2.waitKey(delay=30 if not paused else 0) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # Space
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'} at frame {current_frame}")
            elif key == ord('s'):  # 's' for save
                if save_dir:
                    self.save_frame(
                        frame=overlay_frame,
                        frame_idx=current_frame,
                        output_dir=save_dir
                    )
                else:
                    print("No save directory specified")
            elif paused:
                if key == 83:  # Right arrow
                    current_frame = min(current_frame + 1, len(self.trajectories) - 1)
                elif key == 81:  # Left arrow
                    current_frame = max(current_frame - 1, 0)

        cv2.destroyAllWindows()




if __name__ == "__main__":
    _recording_name: str = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name: str = "0m_37s-1m_37s"
    recording_name_clip: str = _recording_name + "_" + clip_name
    base_path: Path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37")
    video_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4")
    timestamps_npy_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy")
    csv_path: Path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv")
    rerun_eye_video_dataset: EyeVideoDataset = EyeVideoDataset.create(
        data_name=f"{recording_name_clip}_eye_videos",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
    )
    # Create viewer
    viewer: EyeTrackingViewer = EyeTrackingViewer(
        dataset=rerun_eye_video_dataset,
        window_name="Pupil Tracking Viewer"
    )

    # Optional: specify a directory to save frames
    save_directory: Path | None = None
    # save_directory = Path("./saved_frames")

    # Run the viewer
    viewer.run(
        start_frame=0,
        save_dir=save_directory
    )