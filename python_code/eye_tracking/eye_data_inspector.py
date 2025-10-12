"""Simple OpenCV-based eye tracking data inspector."""

from pathlib import Path
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def inspect_eye_data_cv2(
    *,
    csv_path: Path,
    video_path: Path | None = None,
    point_radius: int = 4,
    line_thickness: int = 2
) -> None:
    """
    Simple CV2 inspector - shows video with pupil points and center overlaid.

    Controls:
    - SPACE: Play/Pause
    - RIGHT ARROW: Next frame
    - LEFT ARROW: Previous frame
    - Q or ESC: Quit
    - S: Save current frame as image

    Args:
        csv_path: Path to DLC CSV with pupil data
        video_path: Path to video file (optional)
        point_radius: Size of drawn points
        line_thickness: Thickness of connecting lines
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    logger.info("="*60)
    logger.info("CV2 EYE TRACKING INSPECTOR")
    logger.info("="*60)

    # Load pupil data
    from eye_loaders import load_pupil_centers

    data = load_pupil_centers(filepath=csv_path)

    pupil_centers = data['pupil_centers']
    n_valid_points = data['n_valid_points']
    n_frames = len(pupil_centers)

    # Get raw data to access individual points for visualization
    raw_df = data['raw_data']

    # Extract individual points for visualization
    point_cols = [col for col in raw_df.columns if col.startswith('p') and col.endswith('_x')]
    point_names = [col[:-2] for col in point_cols]
    n_points = len(point_names)

    pupil_points = np.zeros(shape=(n_frames, n_points, 2))
    for i, point_name in enumerate(point_names):
        pupil_points[:, i, 0] = raw_df[f"{point_name}_x"].values
        pupil_points[:, i, 1] = raw_df[f"{point_name}_y"].values

    logger.info(f"Loaded {n_frames} frames with {n_points} points each")

    # Open video if provided
    cap = None
    img_height, img_width = 192, 192

    if video_path is not None and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            cap = None
        else:
            img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Opened video: {img_width}x{img_height} @ {video_fps:.1f} fps")
    else:
        logger.info("No video - showing data only")

    # State
    current_frame = 0
    playing = False

    # Create window
    window_name = 'Eye Inspector [SPACE=play/pause, ARROWS=navigate, Q=quit, S=save]'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, img_width * 2, img_height * 2)

    logger.info("\nControls:")
    logger.info("  SPACE: Play/Pause")
    logger.info("  RIGHT ARROW: Next frame")
    logger.info("  LEFT ARROW: Previous frame")
    logger.info("  Q or ESC: Quit")
    logger.info("  S: Save current frame as image")
    logger.info("")

    def draw_frame(frame_idx: int) -> np.ndarray:
        """Draw frame with pupil annotations."""
        # Get base image
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros(shape=(img_height, img_width, 3), dtype=np.uint8)
        else:
            frame = np.zeros(shape=(img_height, img_width, 3), dtype=np.uint8)

        # Get pupil data for this frame
        points = pupil_points[frame_idx]
        center = pupil_centers[frame_idx]
        n_valid = n_valid_points[frame_idx]

        # Filter valid points
        valid_indices = ~np.isnan(points[:, 0])
        valid_points = points[valid_indices]

        # Draw connecting lines between valid points
        if len(valid_points) > 1:
            for i in range(len(valid_points)):
                pt1 = tuple(valid_points[i].astype(int))
                pt2 = tuple(valid_points[(i + 1) % len(valid_points)].astype(int))
                cv2.line(img=frame, pt1=pt1, pt2=pt2, color=(0, 255, 255), thickness=line_thickness)

        # Draw individual points
        for i in range(n_points):
            if not np.isnan(points[i, 0]):
                pt = tuple(points[i].astype(int))

                # Color gradient: red -> yellow -> green based on index
                ratio = i / max(n_points - 1, 1)
                if ratio < 0.5:
                    color = (0, int(255 * ratio * 2), 255)  # Blue to cyan
                else:
                    color = (0, 255, int(255 * (1 - (ratio - 0.5) * 2)))  # Cyan to green

                cv2.circle(img=frame, center=pt, radius=point_radius, color=color, thickness=-1)
                cv2.circle(img=frame, center=pt, radius=point_radius, color=(255, 255, 255), thickness=1)

                # Draw point number
                cv2.putText(
                    img=frame,
                    text=f"{i+1}",
                    org=(pt[0] + 8, pt[1] - 8),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    thickness=1
                )

        # Draw center cross if valid
        if not np.isnan(center[0]):
            center_pt = tuple(center.astype(int))
            cv2.drawMarker(
                img=frame,
                position=center_pt,
                color=(255, 0, 255),  # Magenta for center
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=3
            )

            # Draw center circle
            cv2.circle(img=frame, center=center_pt, radius=8, color=(255, 0, 255), thickness=2)

        # Add text info
        info_lines = [
            f"Frame: {frame_idx}/{n_frames-1}",
            f"Valid: {n_valid}/{n_points}",
        ]

        if not np.isnan(center[0]):
            info_lines.append(f"Center: ({center[0]:.1f}, {center[1]:.1f})")

        # Draw text background
        y_offset = 25
        for i, line in enumerate(info_lines):
            y = y_offset + i * 25
            (w, h), _ = cv2.getTextSize(text=line, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2)
            cv2.rectangle(img=frame, pt1=(5, y - h - 5), pt2=(15 + w, y + 5), color=(0, 0, 0), thickness=-1)
            cv2.putText(
                img=frame,
                text=line,
                org=(10, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 0),
                thickness=2
            )

        # Add status indicator
        status = "PLAYING" if playing else "PAUSED"
        status_color = (0, 255, 0) if playing else (0, 165, 255)
        cv2.putText(
            img=frame,
            text=status,
            org=(img_width - 120, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=status_color,
            thickness=2
        )

        return frame

    # Main loop
    try:
        while True:
            # Draw current frame
            frame = draw_frame(frame_idx=current_frame)
            cv2.imshow(window_name, frame)

            # Handle input
            wait_time = 33 if playing else 0  # 30fps when playing
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                logger.info("Quitting...")
                break
            elif key == ord(' '):  # SPACE
                playing = not playing
                logger.info(f"{'Playing' if playing else 'Paused'}")
            elif key == 83 or key == 3:  # RIGHT ARROW
                current_frame = min(current_frame + 1, n_frames - 1)
                playing = False
            elif key == 81 or key == 2:  # LEFT ARROW
                current_frame = max(current_frame - 1, 0)
                playing = False
            elif key == ord('s') or key == ord('S'):  # Save frame
                output_path = Path(f"frame_{current_frame:05d}.png")
                cv2.imwrite(filename=str(output_path), img=frame)
                logger.info(f"Saved frame to: {output_path}")

            # Auto-advance when playing
            if playing:
                current_frame += 1
                if current_frame >= n_frames:
                    current_frame = 0  # Loop

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Done!")


if __name__ == "__main__":
    csv_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\dlc_output\model_outputs_iteration_11\eye0_clipped_4354_11523DLC_Resnet50_eye_model_v1_shuffle1_snapshot_050.csv")
    video_path = Path(r"D:\bs\ferret_recordings\2025-07-11_ferret_757_EyeCameras_P43_E15__1\clips\0m_37s-1m_37s\eye_data\eye_videos\eye0_clipped_4354_11523.mp4")

    if not video_path.exists():
        logger.warning(f"Video not found: {video_path}")
        video_path = None

    inspect_eye_data_cv2(
        csv_path=csv_path,
        video_path=video_path,
        point_radius=5,
        line_thickness=2
    )