"""Integration of SVG topology system with eye tracking viewer.

Shows how to define reusable topologies and integrate with existing cv2 workflow.
Supports toggling between raw, cleaned, and both data visualizations.
Includes fitted ellipse overlays and active contour snakes for pupil outline.
"""

from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from python_code.eye_analysis.data_models.abase_model import ABaseModel
from python_code.eye_analysis.video_viewers.create_eye_topology import create_full_eye_topology
from python_code.eye_analysis.data_models.eye_video_dataset import EyeVideoData
from python_code.eye_analysis.video_viewers.image_overlay_system import OverlayTopology, overlay_image


class ViewControls(ABaseModel):
    show_raw_dots: bool = True
    show_raw_lines: bool = True
    show_raw_ellipse: bool = True
    show_cleaned_dots: bool = True
    show_cleaned_lines: bool = True
    show_cleaned_ellipse: bool = True

    def show_raw(self) -> None:
        self.show_raw_dots = True
        self.show_raw_lines = True
        self.show_raw_ellipse = True

    def show_cleaned(self) -> None:
        self.show_cleaned_dots = True
        self.show_cleaned_lines = True
        self.show_cleaned_ellipse = True

    def show_both(self) -> None:
        self.show_raw_dots = True
        self.show_raw_lines = True
        self.show_raw_ellipse = True
        self.show_cleaned_dots = True
        self.show_cleaned_lines = True
        self.show_cleaned_ellipse = True

    def show_none(self) -> None:
        self.show_raw_dots = False
        self.show_raw_lines = False
        self.show_raw_ellipse = False
        self.show_cleaned_dots = False
        self.show_cleaned_lines = False
        self.show_cleaned_ellipse = False


class PlaybackControls(ABaseModel):
    skip_frames: int = 0  # Number of frames to skip between displayed frames
    paused: bool = False  # Whether playback is paused


class EyeVideoDataViewer(ABaseModel):
    dataset: EyeVideoData
    topology: OverlayTopology
    window_name: str = "Eye Tracking Viewer"
    playback_controls: PlaybackControls = PlaybackControls()
    view_config: ViewControls = ViewControls()

    @classmethod
    def create(
        cls,
        *,
        dataset: EyeVideoData,
    ) -> "EyeVideoDataViewer":
        return cls(
            dataset=dataset,
            topology=create_full_eye_topology(
                width=dataset.video.width,
                height=dataset.video.height,
            ),
        )

    def get_raw_trajectories(self) -> np.ndarray:
        return self.dataset.dataset.to_array(use_cleaned=False)

    def get_cleaned_trajectories(self) -> np.ndarray:
        return self.dataset.dataset.to_array(use_cleaned=True)

    def get_frame_points(self, *, frame_index: int) -> dict[str, np.ndarray]:
        """Get point coordinates for frame based on current view configuration.

        Returns dict compatible with topology overlay system.
        """
        include_raw: bool = (
            self.view_config.show_raw_dots
            or self.view_config.show_raw_lines
            or self.view_config.show_raw_ellipse
        )
        include_cleaned: bool = (
            self.view_config.show_cleaned_dots
            or self.view_config.show_cleaned_lines
            or self.view_config.show_cleaned_ellipse
        )

        return self.dataset.dataset.get_frame_points(
            frame_idx=frame_index,
            include_raw=include_raw,
            include_cleaned=include_cleaned,
        )

    def scale_points(
        self, *, points: dict[str, np.ndarray], scale: float
    ) -> dict[str, np.ndarray]:
        """Scale all point coordinates by a factor.

        Args:
            points: Dictionary mapping point names to (x, y) coordinates
            scale: Scaling factor to apply

        Returns:
            New dictionary with scaled coordinates
        """
        scaled_points: dict[str, np.ndarray] = {}
        for name, point in points.items():
            if point is not None and not np.isnan(point).any():
                scaled_points[name] = point * scale
            else:
                scaled_points[name] = point
        return scaled_points

    def get_frame_info(self, *, frame_index: int) -> dict[str, object]:
        """Get metadata for current frame to pass to overlay renderer."""
        view_mode: str = (
            "both"
            if (self.view_config.show_raw_dots and self.view_config.show_cleaned_dots)
            else "cleaned"
            if self.view_config.show_cleaned_dots
            else "raw"
        )

        return {
            "frame_idx": frame_index,
            "total_frames": self.dataset.dataset.n_frames,
            "view_mode": view_mode,
        }

    def run(self, *, start_frame: int = 0, save_dir: Path | None = None) -> None:
        """Run the video viewer with overlay rendering.

        Args:
            start_frame: Frame index to start playback from
            save_dir: Directory to save frames to (unused for now)
        """
        if self.dataset.video.video_capture is None:
            raise ValueError("Video capture is not initialized")

        cv2.namedWindow(winname=self.window_name)
        cv2_window_size = cv2.getWindowImageRect(self.window_name)

        current_frame: int = start_frame
        paused: bool = False

        self.dataset.video.video_capture.set(
            propId=cv2.CAP_PROP_POS_FRAMES, value=current_frame
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

        # Get original video dimensions
        original_width: int = self.dataset.video.width
        original_height: int = self.dataset.video.height

        while True:
            if not self.playback_controls.paused:
                ret, frame = self.dataset.video.video_capture.read()
                if not ret:
                    print("End of video reached")
                    break
                current_frame = (
                    int(
                        self.dataset.video.video_capture.get(
                            propId=cv2.CAP_PROP_POS_FRAMES
                        )
                    )
                    - 1
                )

                if self.playback_controls.skip_frames > 0:
                    for _ in range(self.playback_controls.skip_frames):
                        ret, _ = self.dataset.video.video_capture.read()
                        if not ret:
                            print("End of video reached")
                            break
                    if not ret:
                        break
                    current_frame = (
                        int(
                            self.dataset.video.video_capture.get(
                                propId=cv2.CAP_PROP_POS_FRAMES
                            )
                        )
                        - 1
                    )
            else:
                self.dataset.video.video_capture.set(
                    propId=cv2.CAP_PROP_POS_FRAMES, value=current_frame
                )
                ret, frame = self.dataset.video.video_capture.read()
                if not ret:
                    break

            frame_height: int = frame.shape[0]
            frame_width: int = frame.shape[1]

            scale = 2
            new_width: int = int(frame_width * scale)
            new_height: int = int(frame_height * scale)

            frame = cv2.resize(
                src=frame, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR
            )

            # Get points and scale them to match the resized frame
            points = self.get_frame_points(frame_index=current_frame)
            scaled_points = self.scale_points(points=points, scale=scale)

            overlay_frame: np.ndarray = overlay_image(
                image=frame,
                topology=self.topology,
                points=scaled_points,
                metadata=self.get_frame_info(frame_index=current_frame),
            )

            cv2.imshow(winname=self.window_name, mat=overlay_frame)

            key: int = cv2.waitKey(delay=30 if not paused else 0) & 0xFF

            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'} at frame {current_frame}")
            elif key == ord("r"):
                self.view_config.show_none()
                self.view_config.show_raw()
                print("Showing RAW data only")
            elif key == ord("c"):
                self.view_config.show_none()
                self.view_config.show_cleaned()
                print("Showing CLEANED data only")
            elif key == ord("b"):
                self.view_config.show_both()
                print("Showing BOTH raw and cleaned")
            elif key == ord("d"):
                self.view_config.show_raw_dots = not self.view_config.show_raw_dots
                self.view_config.show_cleaned_dots = (
                    not self.view_config.show_cleaned_dots
                )
                print(f"Dots: {'ON' if self.view_config.show_raw_dots else 'OFF'}")
            elif key == ord("e"):
                self.view_config.show_raw_ellipse = (
                    not self.view_config.show_raw_ellipse
                )
                self.view_config.show_cleaned_ellipse = (
                    not self.view_config.show_cleaned_ellipse
                )
                print(f"Ellipse: {'ON' if self.view_config.show_raw_ellipse else 'OFF'}")
            elif key == ord("+") or key == ord("="):
                self.playback_controls.skip_frames += 1
                print(f"Frame skip: {self.playback_controls.skip_frames}")
            elif key == ord("-") or key == ord("_"):
                self.playback_controls.skip_frames = max(
                    0, self.playback_controls.skip_frames - 1
                )
                print(f"Frame skip: {self.playback_controls.skip_frames}")
            elif paused:
                if key == 83:
                    current_frame = min(
                        current_frame + 1 + self.playback_controls.skip_frames,
                        self.dataset.dataset.n_frames - 1,
                    )
                elif key == 81:
                    current_frame = max(
                        current_frame - 1 - self.playback_controls.skip_frames, 0
                    )

        cv2.destroyAllWindows()