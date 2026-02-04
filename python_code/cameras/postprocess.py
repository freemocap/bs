from pathlib import Path
import shutil

from python_code.cameras.synchronization.timestamp_converter import TimestampConverter
from python_code.cameras.synchronization.timestamp_synchronize import TimestampSynchronize

def copy_files(files: list[Path], destination: Path):
    if len(files) == 0:
        raise ValueError(f"attemped to copy files to {str(destination)} but no files provided")
    print(f"Copying files {[file.name for file in files]} to {str(destination)}")
    for file_path in files:
        shutil.copy2(src = file_path, dst=destination)

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

        copy_files(files=list((base_data_folder / "pupil_output").glob("*.mp4")), destination=eye_video_folder)
        copy_files(files=list((base_data_folder / "pupil_output").glob("*timestamps_utc.npy")), destination=eye_video_folder)


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
        "/home/scholl-lab/ferret_recordings/session_2025-07-09_ferret_753_EyeCameras_P41_E13/base_data"
    )

    postprocess(session_folder_path, include_eyes=True)