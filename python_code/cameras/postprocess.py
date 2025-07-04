from pathlib import Path

from synchronization.timestamp_synchronize import TimestampSynchronize
from synchronization.pupil_synch import PupilSynchronize
from python_code.video_viewing.combine_videos import combine_videos, create_video_info


def main(session_folder_path: Path):
    """
    Postprocess a session folder
    
    Folder should contain Basler videos, timestamps, and timestamp map in a folder titled `raw_videos`
    as well as pupil data in a folder titled `pupil_output`
    It will synchronize the Basler videos, then synchronize them with the pupil videos, and then combine the videos into a single video
    """

    timestamp_synchronize = TimestampSynchronize(session_folder_path, flip_videos=True)
    timestamp_synchronize.synchronize()

    pupil_synchronize = PupilSynchronize(session_folder_path)
    pupil_synchronize.synchronize()

    combined_data_path = session_folder_path / "basler_pupil_synchronized"

    basler_videos, pupil_videos = create_video_info(folder_path=combined_data_path)

    combine_videos(basler_videos=basler_videos, pupil_videos=pupil_videos)


if __name__ == "__main__":
    session_folder_path = Path(
        "/home/scholl-lab/recordings/session_2025-06-24/testing"
    )

    main(session_folder_path)