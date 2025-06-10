import os
import cv2
from pathlib import Path
from typing import List


def find_mp4_files(folder_path: str) -> List[Path]:
    """Find all MP4 files in the specified folder and its subfolders."""
    folder = Path(folder_path)
    return list(folder.glob("**/*.mp4"))


def clip_video(video_path: Path, output_path: Path, start_frame: int, end_frame: int) -> bool:
    """Clip a video to the specified frame range."""
    try:
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate frame range
        if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
            print(f"Error: Invalid frame range ({start_frame}-{end_frame}) for video with {total_frames} frames")
            cap.release()
            return False

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            fps, 
            (width, height)
        )

        # Set the frame position to start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and write frames
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            out.write(frame)
            current_frame += 1

        # Release resources
        cap.release()
        out.release()
        
        print(f"Successfully clipped {video_path.name} to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False






if __name__ == "__main__":
    
    START_FRAME = 10900  # Starting frame (inclusive)
    END_FRAME = 11800  # Ending frame (inclusive)
    INPUT_FOLDER = r"C:\Users\jonma\Sync\freemocap-stuff\freemocap-clients\ben-scholl\data\2025-05-04\ferret_9C04_NoImplant_P41_E9\synchronized_videos"  # Folder containing MP4 files
    RECORDING_NAME= Path(INPUT_FOLDER).parent.name 
    OUTPUT_FOLDER = Path(INPUT_FOLDER).parent / f"clips/{RECORDING_NAME}_{START_FRAME}-{END_FRAME}"  # Output folder for clipped videos


    if END_FRAME < START_FRAME:
        print("Error: END_FRAME must be greater than or equal to START_FRAME")
        raise ValueError("END_FRAME must be greater than or equal to START_FRAME")
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder {INPUT_FOLDER} does not exist")
        raise FileNotFoundError(f"Input folder {INPUT_FOLDER} does not exist")
    
    
    # Find all MP4 files
    mp4_files = find_mp4_files(INPUT_FOLDER)
    
    if not mp4_files:
        print(f"No MP4 files found in {INPUT_FOLDER}")
        raise FileNotFoundError(f"No MP4 files found in {INPUT_FOLDER}")

    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Found {len(mp4_files)} MP4 files")
    
    # Process each video
    for video_path in mp4_files:
        output_path = output_dir / f"clipped_{video_path.name}"
        clip_video(video_path, output_path, START_FRAME, END_FRAME)
