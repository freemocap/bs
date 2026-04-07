from pathlib import Path
import shutil
import cv2
from enum import Enum

from python_code.cameras.synchronization.timestamp_converter import TimestampConverter
from python_code.cameras.synchronization.timestamp_synchronize import TimestampSynchronize

def copy_files(files: list[Path], destination: Path):
    if len(files) == 0:
        raise ValueError(f"attemped to copy files to {str(destination)} but no files provided")
    print(f"Copying files {[file.name for file in files]} to {str(destination)}")
    for file_path in files:
        shutil.copy2(src = file_path, dst=destination)

class FlipMethod(Enum):
    HORIZONTAL = 1
    VERTICAL = 0
    BOTH = -1


def flip_video(video: Path, flip_method: FlipMethod, output_path: Path):
    cap = cv2.VideoCapture(str(video))

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
        fps=round(cap.get(cv2.CAP_PROP_FPS),2),
        frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished reading video")
            break

        flipped_frame = cv2.flip(frame, flip_method.value)

        writer.write(flipped_frame)

    cap.release()
    writer.release()

def move_eyes(
    session_folder_path: Path,
    eye_videos: list[Path],
    timestamps: list[Path],
    output_folder: Path,
):
    """
    Move eye videos to the full recording folder structure, and flip right eye video
    """
    if "757" in str(session_folder_path):
        left_eye = "eye0"
        right_eye = "eye1"
    else:
        left_eye = "eye1"
        right_eye = "eye0"

    for video in eye_videos:
        if left_eye in video.name:
            output_path = output_folder / video.name.replace(left_eye, "left_eye", 1)
            shutil.copy2(src=video, dst=output_path)
        elif right_eye in video.name:
            output_path = output_folder / video.name.replace(right_eye, "right_eye", 1)
            flip_video(video=video, flip_method=FlipMethod.BOTH, output_path=output_path)
        else:
            raise ValueError(f"Video {video} does not contain expected eye identifiers {left_eye} or {right_eye}")

    for timestamp in timestamps:
        if left_eye in timestamp.name:
            new_name = timestamp.name.replace(left_eye, "left_eye", 1)
        elif right_eye in timestamp.name:
            new_name = timestamp.name.replace(right_eye, "right_eye", 1)
        else:
            new_name = timestamp.name
        shutil.copy2(src=timestamp, dst=output_folder / new_name)

def move_to_full_recording(session_folder_path: Path, include_eyes: bool = True):
    full_recording_folder = session_folder_path / "full_recording"
    full_recording_folder.mkdir(parents=False, exist_ok=True)

    base_data_folder = session_folder_path / "base_data"

 
    mocap_data_folder = full_recording_folder / "mocap_data"

    mocap_data_folder.mkdir(parents=False, exist_ok=True)

    if (base_data_folder / "synchronized_corrected_videos").exists():
        synchronized_folder_name = "synchronized_corrected_videos" 
    elif (base_data_folder / "synchronized_videos").exists():
        synchronized_folder_name =  "synchronized_videos"
    else:
        raise RuntimeError(f"No synchronized video folder found in {str(mocap_data_folder)}")
    synchronized_mocap_video_folder = mocap_data_folder / synchronized_folder_name
    synchronized_mocap_video_folder.mkdir(parents=False, exist_ok=True)

    copy_files(files=list((base_data_folder / synchronized_folder_name).glob("*.mp4")), destination=synchronized_mocap_video_folder)
    copy_files(files=list((base_data_folder / synchronized_folder_name).glob("*timestamps_utc.npy")), destination=synchronized_mocap_video_folder)
    copy_files(files=[base_data_folder / synchronized_folder_name / "timestamp_mapping.json"], destination=synchronized_mocap_video_folder)


    if include_eyes:
        eye_data_folder = full_recording_folder / "eye_data"
        eye_data_folder.mkdir(parents=False, exist_ok=True)
        eye_video_folder = eye_data_folder / "eye_videos"
        eye_video_folder.mkdir(parents=False, exist_ok=True)

        move_eyes(
            session_folder_path=session_folder_path,
            eye_videos=list((base_data_folder / "pupil_output").glob("*.mp4")),
            timestamps=list((base_data_folder / "pupil_output").glob("*timestamps_utc.npy")),
            output_folder=eye_video_folder,
        )


def postprocess(session_folder_path: Path, include_eyes: bool = True):
    """
    Postprocess a session folder
    
    Folder should contain Basler videos, timestamps, and timestamp map in a folder titled `raw_videos`
    as well as pupil data in a folder titled `pupil_output`
    It will synchronize the Basler videos, then synchronize them with the pupil videos, and then combine the videos into a single video
    """
    from python_code.video_viewing.combine_basler_videos import combine_videos, create_video_info

    base_data_folder = session_folder_path / "base_data"

    timestamp_synchronize = TimestampSynchronize(base_data_folder, flip_videos=False)
    timestamp_synchronize.synchronize()

    timestamp_converter = TimestampConverter(base_data_folder, include_eyes=include_eyes)
    timestamp_converter.save_basler_utc_timestamps()
    if include_eyes:
        timestamp_converter.save_pupil_utc_timestamps()

    move_to_full_recording(session_folder_path=session_folder_path, include_eyes=include_eyes)

    video_folder = session_folder_path / "full_recording" / "mocap_data" / "synchronized_corrected_videos"

    videos = create_video_info(folder_path=video_folder)

    session_name = session_folder_path.stem
    recording_name = video_folder.parent.stem

    combine_videos(
        videos=videos,
        output_path=video_folder.parent / "combined_mocap.mp4",
        session_name=session_name,
        recording_name=recording_name,
    )

    calibration_path = session_folder_path / "calibration"
    if calibration_path.exists():
        calibration_synchronize = TimestampSynchronize(calibration_path, flip_videos=False)
        calibration_synchronize.synchronize()

def old_postprocess(session_folder_path: Path):
    from synchronization.pupil_synch import PupilSynchronize
    from python_code.video_viewing.combine_videos import combine_videos, create_video_info
    pupil_synchronize = PupilSynchronize(session_folder_path)
    pupil_synchronize.synchronize()

    combined_data_path = session_folder_path / "basler_pupil_synchronized"

    videos = create_video_info(folder_path=combined_data_path)

    combine_videos(videos=videos,
                   output_path=combined_data_path / "combined.mp4",
                   session_name=session_folder_path.parent.stem,
                   recording_name=session_folder_path.stem
    )
    


if __name__ == "__main__":
    session_folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2026-03-08_ferret_407_EO8"
    )

    postprocess(session_folder_path, include_eyes=True)