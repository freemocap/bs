"""Stabilized eye tracking viewer with anatomical alignment.

Extends EyeVideoDataViewer to apply spatial correction that establishes
an anatomical coordinate system centered on the resting pupil position.
"""

from pathlib import Path

import cv2
import numpy as np

from python_code.eye_analysis.data_processing.align_data.eye_anatomical_alignment import (
    compute_spatial_correction_parameters,
    apply_spatial_correction_to_dataset
)
from python_code.eye_analysis.video_viewers.eye_viewer import (
    EyeVideoDataViewer,
    ViewControls,
    PlaybackControls
)
from python_code.eye_analysis.video_viewers.create_eye_topology import create_full_eye_topology
from python_code.eye_analysis.data_models.eye_video_dataset import EyeVideoData
from python_code.eye_analysis.video_viewers.image_overlay_system import OverlayTopology, overlay_image


class StabilizedEyeViewer(EyeVideoDataViewer):
    """Eye tracking viewer with anatomical stabilization and pupil-centered canvas."""

    # Stabilization-specific fields
    padding: int = 50
    show_axes: bool = True

    # Canvas dimensions (computed)
    canvas_width: int
    canvas_height: int
    canvas_offset_x: float
    canvas_offset_y: float

    # Correction parameters (precomputed)
    tear_duct_positions: np.ndarray
    rotation_angles: np.ndarray
    frame_centering_offset: np.ndarray
    median_pupil_canvas_position: np.ndarray

    # Original image dimensions
    img_width: int
    img_height: int

    @classmethod
    def create(
        cls,
        *,
        eye_video_data: EyeVideoData,
        window_name: str = "Stabilized Eye Tracking Viewer",
        padding: int = 50,
        show_axes: bool = True,
    ) -> "StabilizedEyeViewer":
        """Create stabilized viewer with precomputed correction parameters.

        Args:
            eye_video_data: Eye tracking dataset
            window_name: Display window name
            padding: Extra padding around computed bounds (pixels)
            show_axes: Whether to show anatomical reference axes

        Returns:
            Initialized StabilizedEyeViewer instance
        """
        # Original image dimensions
        img_width: int = eye_video_data.video.width
        img_height: int = eye_video_data.video.height

        # Precompute correction parameters using anatomical alignment module
        print("Precomputing spatial correction parameters...")
        (tear_duct_positions,
         rotation_angles,
         frame_centering_offset,
         ) = cls._compute_correction_parameters(eye_video_data=eye_video_data)

        # Compute optimal canvas dimensions
        print("Computing optimal canvas dimensions...")
        (canvas_width,
         canvas_height,
         canvas_offset_x,
         canvas_offset_y,
         median_pupil_canvas_position) = cls._compute_canvas_dimensions(
            img_width=img_width,
            img_height=img_height,
            tear_duct_positions=tear_duct_positions,
            rotation_angles=rotation_angles,
            frame_centering_offset=frame_centering_offset,
            padding=padding
        )

        # Create topology with computed canvas dimensions
        topology: OverlayTopology = create_full_eye_topology(
            width=canvas_width,
            height=canvas_height,
        )

        # Create instance with all computed parameters
        return cls(
            dataset=eye_video_data,
            topology=topology,
            window_name=window_name,
            playback_controls=PlaybackControls(),
            view_config=ViewControls(),
            padding=padding,
            show_axes=show_axes,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            canvas_offset_x=canvas_offset_x,
            canvas_offset_y=canvas_offset_y,
            tear_duct_positions=tear_duct_positions,
            rotation_angles=rotation_angles,
            frame_centering_offset=frame_centering_offset,
            median_pupil_canvas_position=median_pupil_canvas_position,
            img_width=img_width,
            img_height=img_height,
        )

    @staticmethod
    def _compute_correction_parameters(
        *,
        eye_video_data: EyeVideoData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spatial correction parameters using anatomical alignment module.

        Returns:
            Tuple of (tear_duct_positions, rotation_angles, frame_centering_offsets,
                     median_pupil_position)
        """
        # Get pupil center points and compute median
        pupil_names: list[str] = [f'p{i}' for i in range(1, 9)]
        pupil_center_points: np.ndarray = np.asarray(
            [eye_video_data.dataset.trajectories[pname].cleaned.data for pname in pupil_names]
        )
        pupil_center_median_trajectory: np.ndarray = np.nanmedian(
            a=pupil_center_points,
            axis=0
        )
        median_pupil_position: np.ndarray = np.median(
            a=pupil_center_median_trajectory,
            axis=0
        )

        # Use anatomical alignment module to compute correction parameters
        tear_duct_positions: np.ndarray
        rotation_angles: np.ndarray
        frame_centering_offsets: np.ndarray
        pupil_point_names = [f'p{i}' for i in range(1, 9)]
        pupil_center_points = np.asarray([eye_video_data.dataset.trajectories[pname].cleaned.data for pname in pupil_point_names])
        pupil_center_median_trajectory = np.nanmedian(pupil_center_points, axis=0)

        tear_duct_positions, rotation_angles, frame_centering_offset = (
            compute_spatial_correction_parameters(
                stabilize_on=eye_video_data.dataset.trajectories["tear_duct"].cleaned.data,
                align_to=eye_video_data.dataset.trajectories["outer_eye"].cleaned.data,
                center_on=pupil_center_median_trajectory
            )
        )

        return (tear_duct_positions, rotation_angles, frame_centering_offset)

    @staticmethod
    def _compute_canvas_dimensions(
        *,
        img_width: int,
        img_height: int,
        tear_duct_positions: np.ndarray,
        rotation_angles: np.ndarray,
        frame_centering_offset: np.ndarray,
        padding: int
    ) -> tuple[int, int, float, float, np.ndarray]:
        """Compute optimal canvas size by transforming image corners.

        Returns:
            Tuple of (canvas_width, canvas_height, canvas_offset_x, canvas_offset_y,
                     median_pupil_canvas_position)
        """
        # Image corners in pixel coordinates
        corners: np.ndarray = np.array(
            [[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]],
            dtype=np.float32
        )

        # Transform corners for each frame
        all_transformed_corners: list[np.ndarray] = []
        n_frames: int = len(tear_duct_positions)

        for frame_idx in range(n_frames):
            tear_duct_pos: np.ndarray = tear_duct_positions[frame_idx]
            rotation_angle: float = float(rotation_angles[frame_idx])

            # Build rotation matrix
            cos_a: float = float(np.cos(rotation_angle))
            sin_a: float = float(np.sin(rotation_angle))
            rotation_matrix: np.ndarray = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])

            # Transform each corner
            for corner in corners:
                # Step 1: Translate by tear duct
                translated: np.ndarray = corner - tear_duct_pos
                # Step 2: Rotate
                rotated: np.ndarray = rotation_matrix @ translated
                # Step 3: Apply centering offset
                final: np.ndarray = rotated - frame_centering_offset
                all_transformed_corners.append(final)

        all_transformed_corners_array: np.ndarray = np.array(all_transformed_corners)

        # Find bounds after centering
        min_x: float = float(np.min(all_transformed_corners_array[:, 0]))
        max_x: float = float(np.max(all_transformed_corners_array[:, 0]))
        min_y: float = float(np.min(all_transformed_corners_array[:, 1]))
        max_y: float = float(np.max(all_transformed_corners_array[:, 1]))

        print(f"  Transformed bounds (pupil-centered): X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")

        # Make canvas symmetric around origin
        max_extent_x: float = max(abs(min_x), abs(max_x)) + padding
        max_extent_y: float = max(abs(min_y), abs(max_y)) + padding

        canvas_width: int = int(np.ceil(2 * max_extent_x))
        canvas_height: int = int(np.ceil(2 * max_extent_y))

        # Canvas offset: to place (0,0) at canvas center
        canvas_offset_x: float = -canvas_width / 2
        canvas_offset_y: float = -canvas_height / 2

        # Median pupil position in canvas coordinates (should be at center)
        median_pupil_canvas_position: np.ndarray = np.array(
            [canvas_width / 2, canvas_height / 2]
        )

        print(f"  Canvas size: {canvas_width} x {canvas_height}")
        print(f"  Canvas offset: ({canvas_offset_x:.1f}, {canvas_offset_y:.1f})")
        print(f"  Median pupil at canvas center: ({median_pupil_canvas_position[0]:.1f}, "
              f"{median_pupil_canvas_position[1]:.1f})")

        return (canvas_width, canvas_height, canvas_offset_x, canvas_offset_y,
                median_pupil_canvas_position)

    def _apply_frame_correction(
            self,
            *,
            frame_idx: int,
            image: np.ndarray
    ) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        """Apply spatial correction to image and compute transformed overlay points.

        Uses the anatomical alignment transformation approach.

        Args:
            frame_idx: Frame index
            image: Original video frame

        Returns:
            Tuple of (transformed_canvas, corrected_points_dict)
            where corrected_points_dict maintains nested structure:
            {'raw': {'p1': array, ...}, 'cleaned': {'p1': array, ...}}
        """
        # Get correction parameters for this frame
        tear_duct_pos: np.ndarray = self.tear_duct_positions[frame_idx]
        rotation_angle: float = float(self.rotation_angles[frame_idx])
        centering_offset: np.ndarray = self.frame_centering_offset

        # Build 3x3 homogeneous transformation matrices
        # Step 1: Translate so tear duct is at origin
        T1: np.ndarray = np.array([
            [1, 0, -tear_duct_pos[0]],
            [0, 1, -tear_duct_pos[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 2: Rotate around origin (tear duct)
        cos_a: float = float(np.cos(rotation_angle))
        sin_a: float = float(np.sin(rotation_angle))
        R: np.ndarray = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 3: Apply centering offset (combines mode offset and pupil centering)
        T2: np.ndarray = np.array([
            [1, 0, -centering_offset[0]],
            [0, 1, -centering_offset[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 4: Translate to account for canvas offset
        T3: np.ndarray = np.array([
            [1, 0, -self.canvas_offset_x],
            [0, 1, -self.canvas_offset_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # Combine transformations
        transform_3x3: np.ndarray = T3 @ T2 @ R @ T1

        # Extract 2x3 affine transformation for cv2.warpAffine
        transform_2x3: np.ndarray = transform_3x3[:2, :]

        # Apply transformation to image
        transformed_image: np.ndarray = cv2.warpAffine(
            src=image,
            M=transform_2x3,
            dsize=(self.canvas_width, self.canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Transform overlay points - MAINTAIN NESTED STRUCTURE
        corrected_points: dict[str, dict[str, np.ndarray]] = {}

        # Get frame points from parent method - returns nested dict
        original_points: dict[str, dict[str, np.ndarray]] = super().get_frame_points(frame_index=frame_idx)

        # Transform each point while maintaining nested structure
        for data_type, points_dict in original_points.items():
            corrected_points[data_type] = {}

            for name, point in points_dict.items():
                if point is not None and not np.isnan(point).any():
                    point_homogeneous: np.ndarray = np.array([point[0], point[1], 1.0])
                    point_transformed: np.ndarray = transform_3x3 @ point_homogeneous
                    corrected_points[data_type][name] = point_transformed[:2]
                else:
                    corrected_points[data_type][name] = point

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

        canvas_out: np.ndarray = canvas.copy()

        # Origin is at median pupil position (canvas center)
        origin: np.ndarray = self.median_pupil_canvas_position.astype(int)

        # Axis length
        axis_len: int = 100

        # X-axis (lateral-nasal): red
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(int(origin[0] + axis_len), int(origin[1])),
            color=(0, 0, 255),  # Red in BGR
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Lateral (+X)",
            org=(int(origin[0] + axis_len + 10), int(origin[1] + 5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1
        )

        # -X direction (nasal)
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(int(origin[0] - axis_len), int(origin[1])),
            color=(0, 0, 180),  # Dark red
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Nasal (-X)",
            org=(int(origin[0] - axis_len - 100), int(origin[1] + 5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 180),
            thickness=1
        )

        # Y-axis (superior-inferior): green/cyan
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(int(origin[0]), int(origin[1] - axis_len)),
            color=(255, 0, 0),
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Superior (+Y)",
            org=(int(origin[0] + 10), int(origin[1] - axis_len - 10)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 0, 0),
            thickness=1
        )

        # -Y direction (inferior)
        cv2.arrowedLine(
            img=canvas_out,
            pt1=tuple(origin),
            pt2=(int(origin[0]), int(origin[1] + axis_len)),
            color=(180, 0, 0),  # Dark green
            thickness=2,
            tipLength=0.2
        )
        cv2.putText(
            img=canvas_out,
            text="Inferior (-Y)",
            org=(int(origin[0] + 10), int(origin[1] + axis_len + 20)),
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

        return canvas_out

    def toggle_axes(self) -> None:
        """Toggle anatomical axes display."""
        self.show_axes = not self.show_axes
        print(f"Anatomical axes: {'ENABLED' if self.show_axes else 'DISABLED'}")

    def run(self, *, start_frame: int = 0, save_dir: Path | None = None, display: bool = True) -> None:
        """Run the stabilized video viewer with overlay rendering.

        Args:
            start_frame: Frame index to start playback from
            save_dir: Directory to save frames to (unused for now)
        """
        # TODO: add video writer to this
        # TODO: make sure the raw points get drawn on top of everything else as unconnected dots
        if self.dataset.video.video_capture is None:
            raise ValueError("Video capture is not initialized")

        cv2.namedWindow(winname=self.window_name)

        current_frame: int = start_frame
        paused: bool = False

        # TODO: replace with slow accurate position setting
        self.dataset.video.video_capture.set(
            propId=cv2.CAP_PROP_POS_FRAMES, value=current_frame
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
        print("="*60 + "\n")

        while True:
            if not paused:
                ret: bool
                frame: np.ndarray
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

            # Apply spatial correction to frame and points
            canvas: np.ndarray
            corrected_points: dict[str, np.ndarray]
            canvas, corrected_points = self._apply_frame_correction(
                frame_idx=current_frame,
                image=frame
            )

            # Draw anatomical axes
            canvas = self._draw_anatomical_axes(canvas=canvas)

            # Prepare metadata
            metadata: dict[str, object] = {
                'frame_idx': current_frame,
                'total_frames': self.dataset.dataset.n_frames,
                'view_mode': 'both' if (self.view_config.show_raw_dots and self.view_config.show_cleaned_dots)
                            else 'cleaned' if self.view_config.show_cleaned_dots
                            else 'raw',
                'rotation_angle_deg': float(np.degrees(self.rotation_angles[current_frame]))
            }

            # Render overlay
            overlay_frame: np.ndarray = overlay_image(
                image=canvas,
                topology=self.topology,
                points=corrected_points,
                metadata=metadata
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
                self.view_config.show_cleaned_dots = not self.view_config.show_cleaned_dots
                print(f"Dots: {'ON' if self.view_config.show_raw_dots else 'OFF'}")
            elif key == ord("e"):
                self.view_config.show_raw_ellipse = not self.view_config.show_raw_ellipse
                self.view_config.show_cleaned_ellipse = not self.view_config.show_cleaned_ellipse
                print(f"Ellipse: {'ON' if self.view_config.show_raw_ellipse else 'OFF'}")
            elif key == ord("a"):
                self.toggle_axes()
            elif key == ord("+") or key == ord("="):
                self.playback_controls.skip_frames += 1
                print(f"Frame skip: {self.playback_controls.skip_frames}")
            elif key == ord("-") or key == ord("_"):
                self.playback_controls.skip_frames = max(
                    0, self.playback_controls.skip_frames - 1
                )
                print(f"Frame skip: {self.playback_controls.skip_frames}")
            elif paused:
                if key == 83:  # Right arrow
                    current_frame = min(
                        current_frame + 1 + self.playback_controls.skip_frames,
                        self.dataset.dataset.n_frames - 1,
                    )
                elif key == 81:  # Left arrow
                    current_frame = max(
                        current_frame - 1 - self.playback_controls.skip_frames, 0
                    )

        cv2.destroyAllWindows()


def main() -> None:
    """Run stabilized eye tracking viewer with pupil-centered canvas."""
    # Setup paths
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
    print("Loading eye tracking dataset...")
    eye_video_data: EyeVideoData = EyeVideoData.create(
        data_name="ferret_757_eye_tracking",
        recording_path=base_path,
        raw_video_path=video_path,
        timestamps_npy_path=timestamps_npy_path,
        data_csv_path=csv_path,
        butterworth_cutoff=6.0,
    )

    # Create stabilized viewer
    print("Creating stabilized viewer...")
    viewer: StabilizedEyeViewer = StabilizedEyeViewer.create(
        eye_video_data=eye_video_data,
        window_name="Stabilized Eye Tracking - Pupil-Centered",
        padding=50,
        show_axes=True
    )

    # Run viewer
    print("\nStarting viewer...")
    print("\nFeatures:")
    print("  - Uses anatomical alignment methods from eye_anatomical_alignment module")
    print("  - Automatically computed canvas size for minimal black space")
    print("  - All transformed frames guaranteed to be fully visible")
    print("  - Median pupil position centered at canvas origin")
    print("  - Anatomical axes show median pupil as origin (0,0)")
    print("  - Adjustable padding parameter for fine-tuning")
    print()

    viewer.run(start_frame=0)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()