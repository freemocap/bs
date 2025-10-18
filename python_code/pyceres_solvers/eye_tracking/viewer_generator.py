"""Generate interactive HTML viewer with embedded data."""

from pathlib import Path
import pandas as pd
import logging
import shutil

logger = logging.getLogger(__name__)


def save_interactive_viewer(
    *,
    output_dir: Path,
    csv_filepath: Path,
    video_path: Path | None = None
) -> None:
    """
    Generate a HTML viewer with data pre-loaded.

    Args:
        output_dir: Directory to save the viewer
        csv_filepath: Path to the CSV file with tracking results
        video_path: Optional path to video file (will be copied to output dir)
    """
    logger.info("\nGenerating interactive HTML viewer...")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV data
    df = pd.read_csv(filepath_or_buffer=csv_filepath)

    # Convert to JSON-serializable format
    data_json = df.to_json(orient='records')

    # Handle video - copy to output directory
    video_filename = ""
    if video_path is not None and video_path.exists():
        video_filename = video_path.name
        video_dest = output_dir / video_filename

        # Copy video to output directory
        if not video_dest.exists():
            logger.info(f"  Copying video to output directory...")
            shutil.copy2(src=video_path, dst=video_dest)
            logger.info(f"  ✓ Video copied: {video_filename}")
        else:
            logger.info(f"  Video already exists: {video_filename}")

    # Generate the HTML with embedded data using template
    html_content = _generate_viewer_html(
        data_json=data_json,
        video_filename=video_filename,
        n_frames=len(df)
    )

    # Save the viewer
    viewer_path = output_dir / "eye_tracking_viewer.html"
    viewer_path.write_text(data=html_content, encoding='utf-8')

    logger.info(f"  ✓ Saved interactive viewer: {viewer_path.name}")
    logger.info(f"  → Open {viewer_path} in a web browser to view results")


def _generate_viewer_html(
    *,
    data_json: str,
    video_filename: str,
    n_frames: int
) -> str:
    """
    Generate the complete HTML viewer with embedded data using an external template.

    Args:
        data_json: JSON string of the tracking data
        video_filename: Filename of video (relative to HTML file)
        n_frames: Number of frames in the dataset

    Returns:
        Complete HTML as string
    """
    # Load HTML template file located in templates/ relative to this module
    template_path = Path(__file__).with_name("templates") / "eye_tracking_viewer.html"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    template = template_path.read_text(encoding="utf-8")

    # Build dynamic pieces
    video_src_attr = f'src="{video_filename}"' if video_filename else ""
    style_eyevideo_display = "display: block;" if video_filename else "display: none;"
    style_videoplaceholder_display = "display: none;" if video_filename else ""
    data_badge_video_html = (
        '<div class="data-badge">🎥 Video: Loaded</div>'
        if video_filename
        else '<div class="data-badge" style="opacity: 0.5;">🎥 Video: Not Available</div>'
    )
    video_loaded_bool = "true" if video_filename else "false"

    # Perform placeholder substitutions
    html = (
        template
        .replace("__N_FRAMES__", str(n_frames))
        .replace("__FRAME_SLIDER_MAX__", str(max(n_frames - 1, 0)))
        .replace("__DATA_JSON__", data_json)
        .replace("__VIDEO_LOADED_BOOL__", video_loaded_bool)
        .replace("__VIDEO_SRC_ATTR__", video_src_attr)
        .replace("__STYLE_EYEVIDEO_DISPLAY__", style_eyevideo_display)
        .replace("__STYLE_VIDEOPLACEHOLDER_DISPLAY__", style_videoplaceholder_display)
        .replace("__DATA_BADGE_VIDEO_HTML__", data_badge_video_html)
    )

    return html
