from pathlib import Path
from typing import Tuple

from pydantic import BaseModel


class RecordingFolder(BaseModel):
    base_recordings_folder: Path
    recording_name: str
    clip_name: str

    # Derived paths
    recording_folder: Path
    clip_folder: Path

    eye_data_folder: Path
    eye_annotated_videos_folder: Path
    eye_synchronized_videos_folder: Path
    eye_timestamps_folder: Path
    eye_output_data_folder: Path

    eye_data_csv_path: Path
    right_eye_annotated_video_path: Path
    right_eye_video_path: Path
    right_eye_timestamps_npy_path: Path
    left_eye_annotated_video_path: Path
    left_eye_video_path: Path
    left_eye_timestamps_npy_path: Path

    mocap_data_folder: Path
    mocap_annotated_videos_folder: Path
    mocap_synchronized_videos_folder: Path
    mocap_timestamps_folder: Path
    mocap_output_data_folder: Path
    mocap_csv_data_path: Path

    topdown_video_name: str
    topdown_annotated_video_path: Path
    topdown_video_path: Path
    topdown_timestamps_npy_path: Path

    side_0_video_name: str
    side_0_annotated_video_path: Path
    side_0_video_path: Path
    side_0_timestamps_npy_path: Path

    side_1_video_name: str
    side_1_annotated_video_path: Path
    side_1_video_path: Path
    side_1_timestamps_npy_path: Path

    side_2_video_name: str
    side_2_annotated_video_path: Path
    side_2_video_path: Path
    side_2_timestamps_npy_path: Path

    side_3_video_name: str
    side_3_annotated_video_path: Path
    side_3_video_path: Path
    side_3_timestamps_npy_path: Path

    @staticmethod
    def get_video_paths(video_name: str,
        annotated_videos_folder: Path,
        synchronized_videos_folder: Path,
        timestamps_folder: Path
    )-> Tuple[Path, Path, Path]:
        print(f"getting video paths for {video_name} in {synchronized_videos_folder.parent}")
        annotated_video_path = list(annotated_videos_folder.glob(f"{video_name}*.mp4"))[0]
        video_path = list(synchronized_videos_folder.glob(f"{video_name}*.mp4"))[0]
        timestamps_npy_path = list(timestamps_folder.glob(f"{video_name}*utc*.npy"))[0]
        return annotated_video_path, video_path, timestamps_npy_path

    @classmethod
    def create_from_clip(
        cls,
        recording_name: str,
        clip_name: str,
        base_recordings_folder: Path = Path("/Users/philipqueen/"),
    ) -> "RecordingFolder":
        """Create a FerretRecordingPaths instance from recording and clip names."""
        recording_folder = base_recordings_folder / recording_name
        print(f"Parsing recording folder: {recording_folder}")

        clip_folder = recording_folder / "clips" / clip_name
        eye_data_folder = clip_folder / "eye_data"
        eye_annotated_videos_folder = eye_data_folder / "annotated_videos"
        eye_synchronized_videos_folder = eye_data_folder / "eye_videos"
        eye_timestamps_folder = eye_synchronized_videos_folder
        eye_output_data_folder = eye_data_folder / "dlc_output"
        for path in [
            eye_data_folder,
            eye_annotated_videos_folder,
            eye_synchronized_videos_folder,
            eye_timestamps_folder,
            eye_output_data_folder,
        ]:
            if not path.exists():
                raise ValueError(f"Path does not exist: {path}")

        eye_data_csv_path = list(
            eye_output_data_folder.glob("skellyclicker_machine_labels*.csv")
        )[0]

        right_eye_video_name = "eye1"
        (
            right_eye_annotated_video_path,
            right_eye_video_path,
            right_eye_timestamps_npy_path,
        ) = cls.get_video_paths(
            video_name=right_eye_video_name,
            annotated_videos_folder=eye_annotated_videos_folder,
            synchronized_videos_folder=eye_synchronized_videos_folder,
            timestamps_folder=eye_timestamps_folder
        )

        left_eye_video_name = "eye0"
        (
            left_eye_annotated_video_path,
            left_eye_video_path,
            left_eye_timestamps_npy_path
        ) = cls.get_video_paths(
            video_name=left_eye_video_name,
            annotated_videos_folder=eye_annotated_videos_folder,
            synchronized_videos_folder=eye_synchronized_videos_folder,
            timestamps_folder=eye_timestamps_folder
        )

        mocap_data_folder = clip_folder / "mocap_data"
        mocap_annotated_videos_folder = mocap_data_folder / "annotated_videos"
        mocap_synchronized_videos_folder = mocap_data_folder / "synchronized_videos"
        mocap_timestamps_folder = mocap_synchronized_videos_folder
        mocap_output_data_folder = mocap_data_folder / "dlc_output"
        for path in [
            mocap_data_folder,
            mocap_annotated_videos_folder,
            mocap_synchronized_videos_folder,
            mocap_timestamps_folder,
            mocap_output_data_folder,
        ]:
            if not path.exists():
                raise ValueError(f"Path does not exist: {path}")

        mocap_csv_path = list(
            mocap_output_data_folder.glob("skellyclicker_machine_labels*.csv")
        )[0]
        topdown_video_name = "24676894"
        (
            topdown_annotated_video_path,
            topdown_video_path,
            topdown_timestamps_npy_path,
        ) = cls.get_video_paths(
            video_name=topdown_video_name,
            annotated_videos_folder=mocap_annotated_videos_folder,
            synchronized_videos_folder=mocap_synchronized_videos_folder,
            timestamps_folder=mocap_timestamps_folder
        )

        side_0_video_name = "24908831"
        (
            side_0_annotated_video_path,
            side_0_video_path,
            side_0_timestamps_npy_path,
        ) = cls.get_video_paths(
            video_name=side_0_video_name,
            annotated_videos_folder=mocap_annotated_videos_folder,
            synchronized_videos_folder=mocap_synchronized_videos_folder,
            timestamps_folder=mocap_timestamps_folder
        )

        side_1_video_name = "25000609"
        (
            side_1_annotated_video_path,
            side_1_video_path,
            side_1_timestamps_npy_path,
        ) = cls.get_video_paths(
            video_name=side_1_video_name,
            annotated_videos_folder=mocap_annotated_videos_folder,
            synchronized_videos_folder=mocap_synchronized_videos_folder,
            timestamps_folder=mocap_timestamps_folder
        )

        side_2_video_name = "25006505"
        (
            side_2_annotated_video_path,
            side_2_video_path,
            side_2_timestamps_npy_path,
        ) = cls.get_video_paths(
            video_name=side_2_video_name,
            annotated_videos_folder=mocap_annotated_videos_folder,
            synchronized_videos_folder=mocap_synchronized_videos_folder,
            timestamps_folder=mocap_timestamps_folder
        )

        side_3_video_name = "24908832"
        (
            side_3_annotated_video_path,
            side_3_video_path,
            side_3_timestamps_npy_path,
        ) = cls.get_video_paths(
            video_name=side_3_video_name,
            annotated_videos_folder=mocap_annotated_videos_folder,
            synchronized_videos_folder=mocap_synchronized_videos_folder,
            timestamps_folder=mocap_timestamps_folder
        )

        return cls(
            base_recordings_folder=base_recordings_folder,
            recording_name=recording_name,
            clip_name=clip_name,
            recording_folder=recording_folder,
            clip_folder=clip_folder,
            eye_data_folder=eye_data_folder,
            eye_annotated_videos_folder=eye_annotated_videos_folder,
            eye_synchronized_videos_folder=eye_synchronized_videos_folder,
            eye_timestamps_folder=eye_timestamps_folder,
            eye_output_data_folder=eye_output_data_folder,
            eye_data_csv_path=eye_data_csv_path,
            right_eye_annotated_video_path=right_eye_annotated_video_path,
            right_eye_video_path=right_eye_video_path,
            right_eye_timestamps_npy_path=right_eye_timestamps_npy_path,
            left_eye_annotated_video_path=left_eye_annotated_video_path,
            left_eye_video_path=left_eye_video_path,
            left_eye_timestamps_npy_path=left_eye_timestamps_npy_path,
            mocap_data_folder=mocap_data_folder,
            mocap_annotated_videos_folder=mocap_annotated_videos_folder,
            mocap_synchronized_videos_folder=mocap_synchronized_videos_folder,
            mocap_timestamps_folder=mocap_timestamps_folder,
            mocap_output_data_folder=mocap_output_data_folder,
            mocap_csv_data_path=mocap_csv_path,
            topdown_video_name=topdown_video_name,
            topdown_annotated_video_path=topdown_annotated_video_path,
            topdown_video_path=topdown_video_path,
            topdown_timestamps_npy_path=topdown_timestamps_npy_path,
            side_0_video_name=side_0_video_name,
            side_0_annotated_video_path=side_0_annotated_video_path,
            side_0_video_path=side_0_video_path,
            side_0_timestamps_npy_path=side_0_timestamps_npy_path,
            side_1_video_name=side_1_video_name,
            side_1_annotated_video_path=side_1_annotated_video_path,
            side_1_video_path=side_1_video_path,
            side_1_timestamps_npy_path=side_1_timestamps_npy_path,
            side_2_video_name=side_2_video_name,
            side_2_annotated_video_path=side_2_annotated_video_path,
            side_2_video_path=side_2_video_path,
            side_2_timestamps_npy_path=side_2_timestamps_npy_path,
            side_3_video_name=side_3_video_name,
            side_3_annotated_video_path=side_3_annotated_video_path,
            side_3_video_path=side_3_video_path,
            side_3_timestamps_npy_path=side_3_timestamps_npy_path,
        )