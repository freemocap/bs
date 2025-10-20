"""Integration of SVG topology system with eye tracking viewer.

Shows how to define reusable topologies and integrate with existing cv2 workflow.
Supports toggling between raw, cleaned, and both data visualizations.
Includes fitted ellipse overlays and active contour snakes for pupil outline.
"""

from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from python_code.eye_analysis.csv_io import ABaseModel
from python_code.eye_analysis.eye_video_viewers.create_eye_topology import create_full_eye_topology
from python_code.eye_analysis.eye_video_dataset import EyeVideoData
from python_code.eye_analysis.svg_overlay.image_overlay_system import OverlayTopology, overlay_image

DEFAULT_RESIZE_FACTOR: float = 1.0
DEFAULT_MIN_CONFIDENCE: float = 0.3


class ViewMode(str, Enum):
    """Visualization mode for eye tracking data."""
    RAW = "raw"
    CLEANED = "cleaned"
    BOTH = "both"


class PlaybackControls(ABaseModel):
    skip_frames: int = 0  # Number of frames to skip between displayed frames
    paused: bool = False  # Whether playback is paused

class EyeVideoDataViewer(ABaseModel):
    dataset: EyeVideoData

    topology: OverlayTopology
    dataset: EyeVideoData
    window_name: str = "Eye Tracking Viewer"
    controls: PlaybackControls = PlaybackControls()

    @classmethod
    def create(cls,
               dataset: EyeVideoData,
               ) :

        return cls(
            dataset=dataset,
            topology=create_full_eye_topology(
                width=dataset.videos.width,
                height=dataset.videos.height,
            )
            ,
        )

    def get_raw_trajectories(self) -> np.ndarray:
        return self.dataset.dataset.to_array(use_cleaned=False)

    def get_cleaned_trajectories(self) -> np.ndarray:
        return self.dataset.dataset.to_array(use_cleaned=True)

    def run(self, *, start_frame: int = 0, save_dir: Path | None = None) -> None:

        if self.dataset.videos.video_capture is None:
            raise ValueError("Video capture is not initialized")

        cv2.namedWindow(winname=self.window_name)
        cv2_window_size = cv2.getWindowImageRect(self.window_name)

        current_frame: int = start_frame
        paused: bool = False

        self.dataset.videos.video_capture.set(
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
        print("  '+/=': Increase frame skip")
        print("  '-': Decrease frame skip")
        print("  'q' or ESC: Quit")
        print("  Right Arrow: Next frame (when paused)")
        print("  Left Arrow: Previous frame (when paused)")

        print()

        while True:
            if not paused:
                ret, frame = self.dataset.videos.video_capture.read()
                if not ret:
                    print("End of video reached")
                    break
                current_frame = int(
                    self.dataset.videos.video_capture.get(propId=cv2.CAP_PROP_POS_FRAMES)
                ) - 1

                # Skip frames if configured
                if self.skip_frames > 0:
                    for _ in range(self.skip_frames):
                        ret, _ = self.dataset.videos.video_capture.read()
                        if not ret:
                            print("End of video reached")
                            break
                    if not ret:
                        break
                    current_frame = int(
                        self.dataset.videos.video_capture.get(propId=cv2.CAP_PROP_POS_FRAMES)
                    ) - 1
            else:
                self.dataset.videos.video_capture.set(
                    propId=cv2.CAP_PROP_POS_FRAMES,
                    value=current_frame
                )
                ret, frame = self.dataset.videos.video_capture.read()
                if not ret:
                    break

            # Resize frame to fit window
            #TODO - preserve aspect ratio
            window_width, window_height = cv2_window_size[2], cv2_window_size[3]
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            scale_width = window_width / frame_width
            scale_height = frame_height / window_height
            frame = cv2.resize(
                src=frame,
                dsize=(
                    int(frame_width * scale_width),
                    int(frame_height * scale_height)
                ),
                interpolation=cv2.INTER_LINEAR
            )

            # Render SVG overlay
            overlay_frame: np.ndarray = overlay_image(
                image=frame,
                toplogy=self.topology,
                points=self.get_frame_points(current_frame),
                metadata=self.get_frame_info(current_frame),
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
                raise NotImplementedError
            elif key == ord('c'):
                raise NotImplementedError
            elif key == ord('b'):
                raise NotImplementedError
            elif key == ord('d'):
                raise NotImplementedError
            elif key == ord('e'):
                raise NotImplementedError
            elif key == ord('+') or key == ord('='):
                raise NotImplementedError
            elif key == ord('-') or key == ord('_'):
                raise NotImplementedError
            elif paused:
                if key == 83:  # Right arrow
                    current_frame = min(current_frame + 1 + self.skip_frames, len(self.raw_trajectories) - 1)
                elif key == 81:  # Left arrow
                    current_frame = max(current_frame - 1 - self.skip_frames, 0)

        cv2.destroyAllWindows()
