"""Stabilized eye tracking viewer with automatic canvas sizing.

Computes optimal canvas dimensions to minimize black space while ensuring
all transformed frames are fully visible. Centers the canvas on the median
pupil position, making it the anatomical origin (0,0).
"""

from pathlib import Path
from enum import Enum

import cv2
import numpy as np

from python_code.eye_data_cleanup.eye_analysis.spatial_correction import compute_spatial_correction_parameters
from python_code.eye_data_cleanup.eye_viewer import (
    EyeVideoDataset, ViewMode, create_full_eye_topology
)
from python_code.eye_data_cleanup.svg_overlay import SVGOverlayRenderer


class StabilizedEyeTrackingViewer:
    """Eye tracking viewer with anatomical stabilization and optimized, pupil-centered canvas."""

    def __init__(
        self,
        *,
        dataset: EyeVideoDataset,
        window_name: str = "Stabilized Eye Tracking Viewer",
        initial_view_mode: ViewMode = ViewMode.CLEANED,
        enable_dots: bool = True,
        enable_ellipse: bool = True,
        skip_frames: int = 0,
        padding: int = 50,
    ) -> None:
        """Initialize stabilized viewer.

        Args:
            dataset: Eye tracking dataset
            window_name: Display window name
            initial_view_mode: Starting view mode (raw, cleaned, or both)
            enable_dots: Whether to show landmark dots
            enable_ellipse: Whether to show fitted ellipses
            skip_frames: Number of frames to skip per iteration
            padding: Extra padding around the computed bounds (pixels)
        """
        self.dataset: EyeVideoDataset = dataset
        self.window_name: str = window_name
        self.view_mode: ViewMode = initial_view_mode
        self.enable_dots: bool = enable_dots
        self.enable_ellipse: bool = enable_ellipse
        self.skip_frames: int = skip_frames
        self.padding: int = padding

        # Original image dimensions
        self.img_width: int = dataset.video.width
        self.img_height: int = dataset.video.height

        # Get trajectory data arrays
        self.raw_trajectories: np.ndarray = self.dataset.pixel_trajectories.to_array(
            use_cleaned=False
        )
        self.cleaned_trajectories: np.ndarray = self.dataset.pixel_trajectories.to_array(
            use_cleaned=True
        )

        # Precompute correction parameters and optimal canvas
        self._precompute_correction_parameters()
        self._compute_optimal_canvas()

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

    def _compute_optimal_canvas(self) -> None:
        """Compute optimal canvas size by transforming image corners across all frames.

        Centers the canvas so that the median pupil position is at the origin (canvas center).
        """
        print("Computing optimal canvas dimensions...")

        # Image corners in pixel coordinates
        corners = np.array([
            [0, 0],
            [self.img_width, 0],
            [self.img_width, self.img_height],
            [0, self.img_height]
        ], dtype=np.float32)

        # Get pupil centers for all frames
        pupil_names = [f'p{i}' for i in range(1, 9)]
        pupil_points_all_frames = []

        for frame_idx in range(len(self.tear_duct_positions)):
            frame_pupil_points = []
            for pname in pupil_names:
                traj = self.dataset.pixel_trajectories.pairs[pname].cleaned
                frame_pupil_points.append(traj.data[frame_idx])
            pupil_points_all_frames.append(np.mean(frame_pupil_points, axis=0))

        pupil_centers = np.array(pupil_points_all_frames)  # (n_frames, 2)

        # Transform pupil centers and corners for each frame
        all_transformed_corners = []
        all_transformed_pupil_centers = []

        n_frames = len(self.tear_duct_positions)

        for frame_idx in range(n_frames):
            tear_duct_pos = self.tear_duct_positions[frame_idx]
            rotation_angle = self.rotation_angles[frame_idx]

            # Build transformation matrix
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)

            # Transform pupil center
            pupil_center = pupil_centers[frame_idx]
            translated = pupil_center - tear_duct_pos
            rotated = np.array([
                cos_a * translated[0] - sin_a * translated[1],
                sin_a * translated[0] + cos_a * translated[1]
            ])
            final = rotated - self.mode_offset
            all_transformed_pupil_centers.append(final)

            # Transform each corner
            for corner in corners:
                # Step 1: Translate by tear duct
                translated = corner - tear_duct_pos

                # Step 2: Rotate
                rotated = np.array([
                    cos_a * translated[0] - sin_a * translated[1],
                    sin_a * translated[0] + cos_a * translated[1]
                ])

                # Step 3: Translate by mode offset
                final = rotated - self.mode_offset

                all_transformed_corners.append(final)

        all_transformed_corners = np.array(all_transformed_corners)
        all_transformed_pupil_centers = np.array(all_transformed_pupil_centers)

        # Compute median pupil position
        median_pupil_x = np.median(all_transformed_pupil_centers[:, 0])
        median_pupil_y = np.median(all_transformed_pupil_centers[:, 1])
        self.median_pupil_position = np.array([median_pupil_x, median_pupil_y])

        print(f"  Median pupil position (before centering): ({median_pupil_x:.1f}, {median_pupil_y:.1f})")

        # Shift everything so median pupil is at origin
        # We'll shift by -median_pupil_position
        self.pupil_centering_offset = -self.median_pupil_position

        # Apply centering offset to corners
        all_transformed_corners += self.pupil_centering_offset

        # Find bounds after centering (pupil is now at origin 0,0)
        min_x = np.min(all_transformed_corners[:, 0])
        max_x = np.max(all_transformed_corners[:, 0])
        min_y = np.min(all_transformed_corners[:, 1])
        max_y = np.max(all_transformed_corners[:, 1])

        print(f"  Transformed bounds (pupil-centered): X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")

        # Make canvas symmetric around origin (0,0) so pupil ends up at canvas center
        # Canvas needs to extend from most negative to most positive in each direction
        max_extent_x = max(abs(min_x), abs(max_x)) + self.padding
        max_extent_y = max(abs(min_y), abs(max_y)) + self.padding

        self.canvas_width = int(np.ceil(2 * max_extent_x))
        self.canvas_height = int(np.ceil(2 * max_extent_y))

        # Canvas offset: to place (0,0) at canvas center
        self.canvas_offset_x = -self.canvas_width / 2
        self.canvas_offset_y = -self.canvas_height / 2

        # Median pupil position in canvas coordinates
        # Since pupil is at (0,0) in transformed space and canvas is centered on (0,0),
        # the pupil should be at the canvas center
        self.median_pupil_canvas_position = np.array([self.canvas_width / 2, self.canvas_height / 2])

        # Compute tear duct position in canvas
        mean_tear_duct = np.mean(self.tear_duct_positions, axis=0)
        mean_rotation = np.mean(self.rotation_angles)
        cos_a = np.cos(mean_rotation)
        sin_a = np.sin(mean_rotation)

        # Transform mean tear duct (which is at origin after step 1)
        translated = np.array([0.0, 0.0])  # Tear duct at origin after translation by itself
        rotated = np.array([
            cos_a * translated[0] - sin_a * translated[1],
            sin_a * translated[0] + cos_a * translated[1]
        ])
        tear_duct_transformed = rotated - self.mode_offset + self.pupil_centering_offset

        # Position in canvas coordinates
        self.tear_duct_position = tear_duct_transformed - np.array([self.canvas_offset_x, self.canvas_offset_y])

        print(f"  Canvas size: {self.canvas_width} x {self.canvas_height}")
        print(f"  Canvas offset: ({self.canvas_offset_x:.1f}, {self.canvas_offset_y:.1f})")
        print(f"  Median pupil at canvas center: ({self.median_pupil_canvas_position[0]:.1f}, "
              f"{self.median_pupil_canvas_position[1]:.1f})")
        print(f"  Tear duct position in canvas: ({self.tear_duct_position[0]:.1f}, "
              f"{self.tear_duct_position[1]:.1f})")

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

        # Build 3x3 homogeneous transformation matrices
        # Step 1: Translate so tear duct is at origin
        T1 = np.array([
            [1, 0, -tear_duct_pos[0]],
            [0, 1, -tear_duct_pos[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 2: Rotate around origin (tear duct)
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 3: Translate by mode offset
        T2 = np.array([
            [1, 0, -mode_offset[0]],
            [0, 1, -mode_offset[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 4: Center on median pupil position
        T3 = np.array([
            [1, 0, self.pupil_centering_offset[0]],
            [0, 1, self.pupil_centering_offset[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 5: Translate to account for canvas offset (to avoid negative coords)
        T4 = np.array([
            [1, 0, -self.canvas_offset_x],
            [0, 1, -self.canvas_offset_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # Combine transformations
        transform_3x3 = T4 @ T3 @ T2 @ R @ T1

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

        # Transform overlay points
        corrected_points = {}

        # Process raw points if needed
        if self.view_mode in [ViewMode.RAW, ViewMode.BOTH]:
            landmarks_array = self.raw_trajectories[frame_idx]
            for name, idx in self.dataset.landmarks.items():
                point_pixel = landmarks_array[idx]
                point_homogeneous = np.array([point_pixel[0], point_pixel[1], 1.0])
                point_transformed = transform_3x3 @ point_homogeneous
                corrected_points[f"{name}_raw"] = point_transformed[:2]

        # Process cleaned points if needed
        if self.view_mode in [ViewMode.CLEANED, ViewMode.BOTH]:
            landmarks_array = self.cleaned_trajectories[frame_idx]
            for name, idx in self.dataset.landmarks.items():
                point_pixel = landmarks_array[idx]
                point_homogeneous = np.array([point_pixel[0], point_pixel[1], 1.0])
                point_transformed = transform_3x3 @ point_homogeneous
                corrected_points[f"{name}_cleaned"] = point_transformed[:2]

        return transformed_image, corrected_points

    def _draw_anatomical_axes(self, *, canvas: np.ndarray) -> np.ndarray:
        """Draw anatomical reference axes on canvas.

        Origin is at the median pupil position (canvas center).

        Args:
            canvas: Canvas to draw on

        Returns:
            Canvas with axes drawn
        """
        if not self.show_axes:
            return canvas

        canvas_out = canvas.copy()

        # Origin is at median pupil position (canvas center)
        origin = self.median_pupil_canvas_position.astype(int)

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

        # Y-axis (superior-inferior): green/cyan
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(origin[0], origin[1] - axis_len),
            color=(255, 0, 0),
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Superior (+Y)",
            org=(origin[0] + 10, origin[1] - axis_len - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 0, 0),
            thickness=1
        )

        # -Y direction (inferior)
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(origin[0], origin[1] + axis_len),
            color=(180, 0, 0),  # Dark green
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Inferior (-Y)",
            org=(origin[0] + 10, origin[1] + axis_len + 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(250, 0, 0),
            thickness=1
        )

        # Draw origin marker at median pupil position
        cv2.circle(
            img=canvas_out,
            center=tuple(origin),
            radius=5,
            color=(255, 255, 0),  # Cyan
            thickness=-1
        )

        # Draw tear duct marker (for reference)
        tear_duct_canvas = self.tear_duct_position.astype(int)
        cv2.circle(
            img=canvas_out,
            center=tuple(tear_duct_canvas),
            radius=4,
            color=(0, 255, 255),  # Yellow
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
        print("STABILIZED EYE TRACKING VIEWER (PUPIL-CENTERED)")
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
        print(f"\nOptimized canvas size: {self.canvas_width} x {self.canvas_height}")
        print(f"Original image size:   {self.img_width} x {self.img_height}")
        print(f"Padding: {self.padding}px")
        print(f"Median pupil at canvas center: ({self.median_pupil_canvas_position[0]:.0f}, "
              f"{self.median_pupil_canvas_position[1]:.0f})")
        print(f"Tear duct position: ({self.tear_duct_position[0]:.0f}, "
              f"{self.tear_duct_position[1]:.0f})")
        print(f"\nCurrent view mode: {self.view_mode.value.upper()}")
        print(f"Landmark dots: {'ENABLED' if self.enable_dots else 'DISABLED'}")
        print(f"Fitted ellipse: {'ENABLED' if self.enable_ellipse else 'DISABLED'}")
        print(f"Anatomical axes: {'ENABLED' if self.show_axes else 'DISABLED'}")
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
    """Run stabilized eye tracking viewer with pupil-centered canvas."""
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

    # Create stabilized viewer with optimized canvas
    print("Creating stabilized viewer...")
    viewer = StabilizedEyeTrackingViewer(
        dataset=eye_dataset,
        window_name="Stabilized Eye Tracking - Pupil-Centered",
        initial_view_mode=ViewMode.CLEANED,
        enable_dots=True,
        enable_ellipse=True,
        skip_frames=0,
        padding=50  # Adjust padding as needed (smaller = tighter crop)
    )

    # Run viewer
    print("\nStarting viewer...")
    print("\nFeatures:")
    print("  - Automatically computed canvas size for minimal black space")
    print("  - All transformed frames guaranteed to be fully visible")
    print("  - Median pupil position centered at canvas origin")
    print("  - Anatomical axes show median pupil as origin (0,0)")
    print("  - Tear duct marked with yellow circle for reference")
    print("  - Adjustable padding parameter for fine-tuning")
    print()

    viewer.run(
        start_frame=0,
        save_dir=base_path / "stabilized_frames"
    )

    print("\nViewer closed.")


if __name__ == "__main__":
    main()