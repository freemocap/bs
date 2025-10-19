"""Stabilized eye tracking viewer with anatomical coordinate alignment.

The viewer creates a larger canvas and applies spatial correction frame-by-frame,
keeping the tear duct fixed in position while the eye image rotates and translates
to maintain anatomical alignment.
"""

from pathlib import Path
from enum import Enum

import cv2
import numpy as np

from python_code.eye_data_cleanup.csv_io import ABaseModel
from python_code.eye_data_cleanup.eye_analysis.spatial_correction import compute_spatial_correction_parameters
from python_code.eye_data_cleanup.eye_viewer import (
    EyeVideoDataset, ViewMode, create_full_eye_topology
)
from python_code.eye_data_cleanup.svg_overlay import SVGOverlayRenderer


class StabilizedEyeTrackingViewer:
    """Eye tracking viewer with anatomical stabilization and spatial correction.

    Creates a larger canvas (2x original size) and applies frame-by-frame spatial
    correction to maintain the tear duct in a fixed position with anatomical alignment.
    """

    def __init__(
        self,
        *,
        dataset: EyeVideoDataset,
        window_name: str = "Stabilized Eye Tracking Viewer",
        initial_view_mode: ViewMode = ViewMode.CLEANED,
        enable_dots: bool = True,
        enable_ellipse: bool = True,
        enable_snake: bool = False,  # Snake disabled by default for performance
        skip_frames: int = 0,
        canvas_scale: float = 2.0,
        tear_duct_position: tuple[float, float] | None = None
    ) -> None:
        """Initialize stabilized viewer.

        Args:
            dataset: Eye tracking dataset
            window_name: Display window name
            initial_view_mode: Starting view mode (raw, cleaned, or both)
            enable_dots: Whether to show landmark dots
            enable_ellipse: Whether to show fitted ellipses
            enable_snake: Whether to enable snake contour fitting
            skip_frames: Number of frames to skip per iteration
            canvas_scale: Scale factor for canvas size (2.0 = 2x original size)
            tear_duct_position: Fixed position for tear duct in canvas coords (default: center-right)
        """
        self.dataset: EyeVideoDataset = dataset
        self.window_name: str = window_name
        self.view_mode: ViewMode = initial_view_mode
        self.enable_dots: bool = enable_dots
        self.enable_ellipse: bool = enable_ellipse
        self.enable_snake: bool = enable_snake
        self.skip_frames: int = skip_frames
        self.canvas_scale: float = canvas_scale

        # Original image dimensions
        self.img_width: int = dataset.video.width
        self.img_height: int = dataset.video.height

        # Canvas dimensions
        self.canvas_width: int = int(self.img_width * canvas_scale)
        self.canvas_height: int = int(self.img_height * canvas_scale)

        # Fixed tear duct position in canvas (default: center-right)
        if tear_duct_position is None:
            self.tear_duct_position: np.ndarray = np.array([
                self.canvas_width * 0.75,  # 75% to the right
                self.canvas_height * 0.5   # Centered vertically
            ])
        else:
            self.tear_duct_position = np.array(tear_duct_position)

        # Get trajectory data arrays
        self.raw_trajectories: np.ndarray = self.dataset.pixel_trajectories.to_array(
            use_cleaned=False
        )
        self.cleaned_trajectories: np.ndarray = self.dataset.pixel_trajectories.to_array(
            use_cleaned=True
        )

        # Precompute correction parameters for all frames
        self._precompute_correction_parameters()

        # Create initial renderer
        self.renderer: SVGOverlayRenderer = self._create_renderer()

        # Show axes
        self.show_axes: bool = True

    def _precompute_correction_parameters(self) -> None:
        """Precompute spatial correction parameters for all frames."""
        print("Precomputing spatial correction parameters...")

        # Get trajectories
        tear_duct_traj = self.dataset.pixel_trajectories.pairs['tear_duct'].cleaned
        outer_eye_traj = self.dataset.pixel_trajectories.pairs['outer_eye'].cleaned
        pupil_trajs = [
            self.dataset.pixel_trajectories.pairs[f'p{i}'].cleaned
            for i in range(1, 9)
        ]

        # Compute correction parameters
        (self.tear_duct_positions,
         self.rotation_angles,
         self.mode_offset) = compute_spatial_correction_parameters(
            tear_duct_trajectory=tear_duct_traj,
            outer_eye_trajectory=outer_eye_traj,
            pupil_trajectories=pupil_trajs,
            n_bins=50
        )

        print(f"  Mode offset: ({self.mode_offset[0]:.2f}, {self.mode_offset[1]:.2f})")
        print(f"  Rotation range: [{np.degrees(self.rotation_angles.min()):.1f}°, "
              f"{np.degrees(self.rotation_angles.max()):.1f}°]")

    def _create_renderer(self) -> SVGOverlayRenderer:
        """Create renderer based on current view mode."""
        show_raw = self.view_mode in [ViewMode.RAW, ViewMode.BOTH]
        show_cleaned = self.view_mode in [ViewMode.CLEANED, ViewMode.BOTH]

        topology = create_full_eye_topology(
            width=self.canvas_width,
            height=self.canvas_height,
            show_raw=show_raw,
            show_cleaned=show_cleaned,
            show_dots=self.enable_dots,
            show_ellipse=self.enable_ellipse,
            show_snake=self.enable_snake,
            n_snake_points=self.dataset.snake_params.n_points
        )

        return SVGOverlayRenderer(topology=topology)

    def _apply_frame_correction(
        self,
        *,
        frame_idx: int,
        image: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Apply spatial correction to image and compute transformed overlay points.

        Args:
            frame_idx: Frame index
            image: Original video frame

        Returns:
            Tuple of (transformed_canvas, corrected_points_dict)
        """
        # Get correction parameters for this frame
        tear_duct_pos = self.tear_duct_positions[frame_idx]
        rotation_angle = self.rotation_angles[frame_idx]
        mode_offset = self.mode_offset

        # Create blank canvas
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)

        # Compute transformation to place image in canvas
        # We want: tear_duct (in pixel coords) -> self.tear_duct_position (in canvas coords)
        # After rotation and mode offset correction

        # Build 3x3 homogeneous transformation matrices for composition
        # Step 1: Translate so tear duct is at origin
        T1 = np.array([
            [1, 0, -tear_duct_pos[0]],
            [0, 1, -tear_duct_pos[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 2: Rotate around origin
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 3: Translate by mode offset (to center pupil at origin)
        T2 = np.array([
            [1, 0, -mode_offset[0]],
            [0, 1, -mode_offset[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 4: Translate to fixed tear duct position in canvas
        T3 = np.array([
            [1, 0, self.tear_duct_position[0]],
            [0, 1, self.tear_duct_position[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Combine transformations (matrix multiplication from right to left)
        # First translate to origin, then rotate, then translate by mode, then to canvas
        transform_3x3 = T3 @ T2 @ R @ T1

        # Extract 2x3 affine transformation for cv2.warpAffine
        transform_2x3 = transform_3x3[:2, :]

        # Apply transformation to image
        transformed_image = cv2.warpAffine(
            src=image,
            M=transform_2x3,
            dsize=(self.canvas_width, self.canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        canvas = transformed_image

        # Transform overlay points
        corrected_points = {}

        # Process raw points if needed
        if self.view_mode in [ViewMode.RAW, ViewMode.BOTH]:
            landmarks_array = self.raw_trajectories[frame_idx]
            for name, idx in self.dataset.landmarks.items():
                # Apply same transformation to overlay points
                point_pixel = landmarks_array[idx]
                point_homogeneous = np.array([point_pixel[0], point_pixel[1], 1.0])
                point_transformed = transform_3x3 @ point_homogeneous
                corrected_points[f"{name}_raw"] = point_transformed[:2]

        # Process cleaned points if needed
        if self.view_mode in [ViewMode.CLEANED, ViewMode.BOTH]:
            landmarks_array = self.cleaned_trajectories[frame_idx]
            for name, idx in self.dataset.landmarks.items():
                # Apply same transformation to overlay points
                point_pixel = landmarks_array[idx]
                point_homogeneous = np.array([point_pixel[0], point_pixel[1], 1.0])
                point_transformed = transform_3x3 @ point_homogeneous
                corrected_points[f"{name}_cleaned"] = point_transformed[:2]

        return canvas, corrected_points

    def _draw_anatomical_axes(self, *, canvas: np.ndarray) -> np.ndarray:
        """Draw anatomical reference axes on canvas.

        Args:
            canvas: Canvas to draw on

        Returns:
            Canvas with axes drawn
        """
        if not self.show_axes:
            return canvas

        canvas_out = canvas.copy()

        # Origin is at tear duct position
        origin = self.tear_duct_position.astype(int)

        # Axis length
        axis_len = 100

        # X-axis (lateral-nasal): red
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(origin[0] + axis_len, origin[1]),
            color=(0, 0, 255),  # Red in BGR
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Lateral (+X)",
            org=(origin[0] + axis_len + 10, origin[1] + 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1
        )

        # -X direction (nasal)
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(origin[0] - axis_len, origin[1]),
            color=(0, 0, 180),  # Dark red
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Nasal (-X)",
            org=(origin[0] - axis_len - 100, origin[1] + 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 180),
            thickness=1
        )

        # Y-axis (superior-inferior): green
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(origin[0], origin[1] - axis_len),
            color=(0, 255, 0),  # Green in BGR
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Superior (+Y)",
            org=(origin[0] + 10, origin[1] - axis_len - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1
        )

        # -Y direction (inferior)
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(origin[0], origin[1] + axis_len),
            color=(0, 180, 0),  # Dark green
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Inferior (-Y)",
            org=(origin[0] + 10, origin[1] + axis_len + 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 180, 0),
            thickness=1
        )

        # Draw origin marker (tear duct position)
        cv2.circle(
            img=canvas_out,
            center=tuple(origin),
            radius=5,
            color=(255, 255, 0),  # Cyan
            thickness=-1
        )

        return canvas_out

    def render_frame_overlay(
        self,
        *,
        frame: np.ndarray,
        frame_idx: int
    ) -> np.ndarray:
        """Render stabilized frame with anatomical overlay.

        Args:
            frame: Original video frame
            frame_idx: Current frame index

        Returns:
            Canvas with corrected image and overlay
        """
        # Apply spatial correction
        canvas, corrected_points = self._apply_frame_correction(
            frame_idx=frame_idx,
            image=frame
        )

        # Draw anatomical axes
        canvas = self._draw_anatomical_axes(canvas=canvas)

        # Prepare metadata
        metadata = {
            'frame_idx': frame_idx,
            'total_frames': len(self.raw_trajectories) - 1,
            'dataset_name': self.dataset.data_name,
            'view_mode': self.view_mode.value,
            'rotation_angle_deg': float(np.degrees(self.rotation_angles[frame_idx]))
        }

        # Render SVG overlay
        overlay_canvas = self.renderer.render_and_composite(
            image=canvas,
            points=corrected_points,
            metadata=metadata
        )

        return overlay_canvas

    def set_view_mode(self, *, mode: ViewMode) -> None:
        """Change view mode and recreate renderer."""
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

    def toggle_axes(self) -> None:
        """Toggle anatomical axes display."""
        self.show_axes = not self.show_axes
        print(f"Anatomical axes: {'ENABLED' if self.show_axes else 'DISABLED'}")

    def increment_skip_frames(self) -> None:
        """Increment frame skip count."""
        self.skip_frames += 1
        print(f"Skip frames: {self.skip_frames}")

    def decrement_skip_frames(self) -> None:
        """Decrement frame skip count (minimum 0)."""
        self.skip_frames = max(0, self.skip_frames - 1)
        print(f"Skip frames: {self.skip_frames}")

    def run(self, *, start_frame: int = 0, save_dir: Path | None = None) -> None:
        """Run the interactive stabilized viewer.

        Controls:
            - Space: Pause/Resume
            - 's': Save current frame (PNG)
            - 'r': Show RAW data only
            - 'c': Show CLEANED data only
            - 'b': Show BOTH raw and cleaned
            - 'd': Toggle DOTS (landmark points)
            - 'e': Toggle ELLIPSE (fitted ellipse)
            - 'a': Toggle AXES (anatomical reference)
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

        print("\n" + "="*60)
        print("STABILIZED EYE TRACKING VIEWER")
        print("="*60)
        print("\nControls:")
        print("  Space: Pause/Resume")
        print("  's': Save current frame as PNG")
        print("  'r': Show RAW data only")
        print("  'c': Show CLEANED data only")
        print("  'b': Show BOTH raw and cleaned")
        print("  'd': Toggle DOTS (landmark points)")
        print("  'e': Toggle ELLIPSE (fitted ellipse)")
        print("  'a': Toggle AXES (anatomical reference)")
        print("  'n': Toggle SNAKE contours")
        print("  '+/=': Increase frame skip")
        print("  '-': Decrease frame skip")
        print("  'q' or ESC: Quit")
        print("  Right Arrow: Next frame (when paused)")
        print("  Left Arrow: Previous frame (when paused)")
        print(f"\nCanvas size: {self.canvas_width} x {self.canvas_height}")
        print(f"Image size:  {self.img_width} x {self.img_height}")
        print(f"Scale factor: {self.canvas_scale:.1f}x")
        print(f"Fixed tear duct position: ({self.tear_duct_position[0]:.0f}, "
              f"{self.tear_duct_position[1]:.0f})")
        print(f"\nCurrent view mode: {self.view_mode.value.upper()}")
        print(f"Landmark dots: {'ENABLED' if self.enable_dots else 'DISABLED'}")
        print(f"Fitted ellipse: {'ENABLED' if self.enable_ellipse else 'DISABLED'}")
        print(f"Anatomical axes: {'ENABLED' if self.show_axes else 'DISABLED'}")
        print(f"Snake contours: {'ENABLED' if self.enable_snake else 'DISABLED'}")
        print(f"Skip frames: {self.skip_frames}")
        print("="*60 + "\n")

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

            # Render stabilized frame with overlay
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
            elif key == ord('a'):
                self.toggle_axes()
            elif key == ord('n'):
                self.toggle_snake()
            elif key == ord('+') or key == ord('='):
                self.increment_skip_frames()
            elif key == ord('-') or key == ord('_'):
                self.decrement_skip_frames()
            elif key == ord('s'):
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    output_path = save_dir / f"stabilized_frame_{current_frame:06d}.png"
                    cv2.imwrite(filename=str(output_path), img=overlay_frame)
                    print(f"Saved PNG: {output_path}")
                else:
                    print("No save directory specified")
            elif paused:
                if key == 83:  # Right arrow
                    current_frame = min(
                        current_frame + 1 + self.skip_frames,
                        len(self.raw_trajectories) - 1
                    )
                elif key == 81:  # Left arrow
                    current_frame = max(current_frame - 1 - self.skip_frames, 0)

        cv2.destroyAllWindows()

def main() -> None:
    """Run stabilized eye tracking viewer."""
    # Setup paths
    base_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37"
    )
    video_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_clipped_4371_11541.mp4"
    )
    timestamps_npy_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\eye_videos\eye1_timestamps_utc_clipped_4371_11541.npy"
    )
    csv_path = Path(
        r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EYeCameras_P43_E15__1\clips\0m_37s-1m_37\eye_data\dlc_output\model_outputs_iteration_11\eye1_clipped_4371_11541DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv"
    )

    # Create dataset
    print("Loading eye tracking dataset...")
    eye_dataset = EyeVideoDataset.create(
        data_name="ferret_757_eye_tracking",
        base_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
        butterworth_cutoff=6.0,
        butterworth_sampling_rate=90.0
    )

    # Create stabilized viewer
    print("Creating stabilized viewer...")
    viewer = StabilizedEyeTrackingViewer(
        dataset=eye_dataset,
        window_name="Stabilized Eye Tracking - Anatomical Reference Frame",
        initial_view_mode=ViewMode.CLEANED,
        enable_dots=True,
        enable_ellipse=True,
        enable_snake=False,  # Disable snake for better performance
        skip_frames=0,
        canvas_scale=2.0,  # 2x original size
        tear_duct_position=None  # Use default (center-right of canvas)
    )

    # Run viewer
    print("\nStarting viewer...")
    print("\nFeatures:")
    print("  - Tear duct position is FIXED in the canvas")
    print("  - Eye image rotates and translates to maintain anatomical alignment")
    print("  - Red/Green axes show anatomical directions")
    print("  - Pupil center is aligned to anatomical origin (mode position)")
    print()

    viewer.run(
        start_frame=0,
        save_dir=base_path / "stabilized_frames"
    )

    print("\nViewer closed.")


if __name__ == "__main__":
    main()