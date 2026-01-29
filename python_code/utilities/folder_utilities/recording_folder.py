from pathlib import Path
from typing import Tuple
from enum import Enum

from pydantic import BaseModel


class PipelineSteps(Enum):
    RAW = "raw"
    SYNCHRONIZED = "synchronized"
    DLCED = "DLCED"
    TRIANGULATED = "triangulated"
    POST_PROCESSED = "post_processed"


class BaslerCameras(Enum):
    TOPDOWN = "24676894"
    SIDE_0 = "24908831"
    SIDE_1 = "24908832"
    SIDE_2 = "25000609"
    SIDE_3 = "25006505"


class RecordingFolder(BaseModel):
    folder: Path
    base_recordings_folder: Path
    recording_name: str
    version_name: str
    is_clip: bool

    @classmethod
    def from_folder_path(cls, folder: Path | str) -> "RecordingFolder":
        folder = Path(folder)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")
        if not folder.is_dir():
            raise ValueError(f"Folder is not a directory: {folder}")

        if "clips" in folder.parts:
            base_recordings_folder = folder.parent.parent
            recording_name = base_recordings_folder.stem
            version_name = folder.stem
            is_clip = True
        else:
            base_recordings_folder = folder.parent
            recording_name = base_recordings_folder.stem
            version_name = folder.parent.stem
            is_clip = False

            if version_name != "full_recording":
                raise ValueError(
                    f"Folder must be in 'clips' or be 'full_recording': {folder}"
                )

        if not (folder / "mocap_data").exists():
            raise ValueError(f"Folder does not contain mocap_data: {folder}")
        if not (folder / "eye_data").exists():
            raise ValueError(f"Folder does not contain eye_data: {folder}")

        return cls(
            folder=folder,
            base_recordings_folder=base_recordings_folder,
            recording_name=recording_name,
            version_name=version_name,
            is_clip=is_clip,
        )

    @property
    def mocap_data(self) -> Path:
        return self.folder / "mocap_data"

    @property
    def eye_data(self) -> Path:
        return self.folder / "eye_data"

    @property
    def eye_annotated_videos(self) -> Path | None:
        annotated_videos = (
            self.eye_data / "annotated_videos" / "annotated_videos_eye_model_v3"
        )
        return annotated_videos if annotated_videos.exists() else None

    @property
    def eye_annotated_flipped(self) -> Path | None:
        flipped_annotated_videos = (
            self.eye_data / "annotated_videos" / "annotated_videos_eye_model_v3_flipped"
        )
        return flipped_annotated_videos if flipped_annotated_videos.exists() else None

    @property
    def eye_videos(self) -> Path | None:
        eye_videos = self.eye_data / "videos"
        return eye_videos if eye_videos.exists() else None

    @property
    def eye_dlc_output(self) -> Path | None:
        dlc_output = self.eye_data / "dlc_output" / "eye_model_v3"
        return dlc_output if dlc_output.exists() else None

    @property
    def eye_dlc_output_flipped(self) -> Path | None:
        dlc_output = self.eye_data / "dlc_output" / "eye_model_v3_flipped"
        return dlc_output if dlc_output.exists() else None

    @property
    def eye_data_skellyclicker_labels(self) -> Path | None:
        skellyclicker_labels = (
            self.eye_dlc_output.glob("skellyclicker_machine_labels*.csv")
            if self.eye_dlc_output
            else None
        )
        return next(skellyclicker_labels, None) if skellyclicker_labels else None

    @property
    def eye_data_skellyclicker_labels_flipped(self) -> Path | None:
        skellyclicker_labels = (
            self.eye_dlc_output_flipped.glob("skellyclicker_machine_labels*.csv")
            if self.eye_dlc_output_flipped
            else None
        )
        return next(skellyclicker_labels, None) if skellyclicker_labels else None

    @property
    def eye_output_data(self) -> Path | None:
        output_data = self.eye_data / "output_data"
        return output_data if output_data.exists() else None

    @property
    def eye_data_csv(self) -> Path | None:
        eye_data_csv = self.eye_data / "eye_data.csv"
        return eye_data_csv if eye_data_csv.exists() else None

    @property
    def eye_mean_confidence(self) -> Path | None:
        eye_mean_confidence = self.eye_data / "mean_confidence.csv"
        return eye_mean_confidence if eye_mean_confidence.exists() else None

    @property
    def right_eye_video(self) -> Path | None:
        right_eye_video = self.eye_videos / "eye1.mp4" if self.eye_videos else None
        return right_eye_video if right_eye_video and right_eye_video.exists() else None

    @property
    def right_eye_annotated_video(self) -> Path | None:
        right_eye_annotated_video = (
            self.eye_annotated_videos / "eye1.mp4"
            if self.eye_annotated_videos
            else None
        )
        return (
            right_eye_annotated_video
            if right_eye_annotated_video and right_eye_annotated_video.exists()
            else None
        )

    @property
    def right_eye_annotated_flipped_video(self) -> Path | None:
        right_eye_annotated_video = (
            self.eye_annotated_flipped / "eye1.mp4"
            if self.eye_annotated_flipped
            else None
        )
        if right_eye_annotated_video is None:
            right_eye_annotated_video = (
                self.eye_annotated_flipped / "eye1_flipped.mp4"
                if self.eye_annotated_flipped
                else None
            )
        return (
            right_eye_annotated_video
            if right_eye_annotated_video and right_eye_annotated_video.exists()
            else None
        )

    @property
    def right_eye_timestamps_npy(self) -> Path | None:
        right_eye_timestamps_npy = (
            self.eye_videos / "eye1_timestamps.npy" if self.eye_videos else None
        )
        return (
            right_eye_timestamps_npy
            if right_eye_timestamps_npy and right_eye_timestamps_npy.exists()
            else None
        )

    @property
    def right_eye_plot_points_csv(self) -> Path | None:
        right_eye_plot_points_csv = (
            self.eye_output_data / "right_eye_plot_points.csv"
            if self.eye_output_data
            else None
        )
        return (
            right_eye_plot_points_csv
            if right_eye_plot_points_csv and right_eye_plot_points_csv.exists()
            else None
        )

    @property
    def right_eye_stabilized_canvas(self) -> Path | None:
        right_eye_aligned_canvas = (
            self.eye_output_data / "right_eye_stabilized_canvas.mp4"
            if self.eye_output_data
            else None
        )
        return (
            right_eye_aligned_canvas
            if right_eye_aligned_canvas and right_eye_aligned_canvas.exists()
            else None
        )

    @property
    def left_eye_video(self) -> Path | None:
        left_eye_video = self.eye_videos / "eye0.mp4" if self.eye_videos else None
        return left_eye_video if left_eye_video and left_eye_video.exists() else None

    @property
    def left_eye_annotated_video(self) -> Path | None:
        left_eye_annotated_video = (
            self.eye_annotated_videos / "eye0.mp4"
            if self.eye_annotated_videos
            else None
        )
        return (
            left_eye_annotated_video
            if left_eye_annotated_video and left_eye_annotated_video.exists()
            else None
        )

    @property
    def left_eye_annotated_flipped_video(self) -> Path | None:
        left_eye_annotated_video = (
            self.eye_annotated_flipped / "eye0.mp4"
            if self.eye_annotated_flipped
            else None
        )
        if left_eye_annotated_video is None:
            left_eye_annotated_video = (
                self.eye_annotated_flipped / "eye0_flipped.mp4"
                if self.eye_annotated_flipped
                else None
            )
        return (
            left_eye_annotated_video
            if left_eye_annotated_video and left_eye_annotated_video.exists()
            else None
        )

    @property
    def left_eye_timestamps_npy(self) -> Path | None:
        left_eye_timestamps_npy = (
            self.eye_videos / "eye2_timestamps.npy" if self.eye_videos else None
        )
        return (
            left_eye_timestamps_npy
            if left_eye_timestamps_npy and left_eye_timestamps_npy.exists()
            else None
        )

    @property
    def left_eye_plot_points_csv(self) -> Path | None:
        left_eye_plot_points_csv = (
            self.eye_output_data / "left_eye_plot_points.csv"
            if self.eye_output_data
            else None
        )
        return (
            left_eye_plot_points_csv
            if left_eye_plot_points_csv and left_eye_plot_points_csv.exists()
            else None
        )

    @property
    def left_eye_stabilized_canvas(self) -> Path | None:
        left_eye_aligned_canvas = (
            self.eye_output_data / "left_eye_stabilized_canvas.mp4"
            if self.eye_output_data
            else None
        )
        return (
            left_eye_aligned_canvas
            if left_eye_aligned_canvas and left_eye_aligned_canvas.exists()
            else None
        )

    @property
    def mocap_synchronized_videos(self) -> Path | None:
        mocap_synchronized_videos = self.mocap_data / "synchronized_corrected_videos"
        if not mocap_synchronized_videos.exists():
            mocap_synchronized_videos = self.mocap_data / "synchronized_videos"
        return (
            mocap_synchronized_videos
            if mocap_synchronized_videos and mocap_synchronized_videos.exists()
            else None
        )

    @property
    def head_body_annotated_videos(self) -> Path | None:
        mocap_annotated_videos = (
            self.mocap_data
            / "annotated_videos"
            / "annotated_videos_head_body_eyecam_retrain_test_v2"
        )
        return (
            mocap_annotated_videos
            if mocap_annotated_videos and mocap_annotated_videos.exists()
            else None
        )

    @property
    def head_body_dlc_output(self) -> Path | None:
        mocap_dlc_output = (
            self.mocap_data / "dlc_output" / "head_body_eyecam_retrain_test_v2"
        )
        return (
            mocap_dlc_output if mocap_dlc_output and mocap_dlc_output.exists() else None
        )

    @property
    def head_body_skellyclicker_labels(self) -> Path | None:
        mocap_machine_labels = (
            self.head_body_dlc_output.glob("skellyclicker_machine_labels*.csv")
            if self.head_body_dlc_output
            else None
        )
        return next(mocap_machine_labels, None) if mocap_machine_labels else None

    @property
    def toy_annotated_videos(self) -> Path | None:
        mocap_annotated_videos = (
            self.mocap_data / "annotated_videos" / "annotated_videos_toy_model_v2"
        )
        return (
            mocap_annotated_videos
            if mocap_annotated_videos and mocap_annotated_videos.exists()
            else None
        )

    @property
    def toy_skellyclicker_labels(self) -> Path | None:
        mocap_machine_labels = (
            self.toy_annotated_videos.glob("skellyclicker_machine_labels*.csv")
            if self.toy_annotated_videos
            else None
        )
        return next(mocap_machine_labels, None) if mocap_machine_labels else None

    @property
    def toy_dlc_output(self) -> Path | None:
        mocap_dlc_output = self.mocap_data / "dlc_output" / "toy_model_v2"
        return (
            mocap_dlc_output if mocap_dlc_output and mocap_dlc_output.exists() else None
        )

    @property
    def mocap_output_data(self) -> Path | None:
        mocap_output_data = self.mocap_data / "output_data"
        return (
            mocap_output_data
            if mocap_output_data and mocap_output_data.exists()
            else None
        )

    @property
    def mocap_3d_data(self) -> Path | None:
        mocap_3d_data = (
            self.mocap_output_data / "dlc" if self.mocap_output_data else None
        )
        return mocap_3d_data if mocap_3d_data and mocap_3d_data.exists() else None

    @property
    def mocap_solver_output(self) -> Path | None:
        mocap_solver_output = (
            self.mocap_output_data / "solver_output" if self.mocap_output_data else None
        )
        return (
            mocap_solver_output
            if mocap_solver_output and mocap_solver_output.exists()
            else None
        )

    def get_synchronized_video_by_name(self, video_name: str) -> Path:
        synchronized_video = (
            self.mocap_synchronized_videos.glob(video_name + "*.mp4")
            if self.mocap_synchronized_videos
            else None
        )
        synchronized_video = (
            next(synchronized_video, None) if synchronized_video else None
        )
        if not synchronized_video:
            raise ValueError(
                f"Could not find synchronized video for {video_name} in {self.mocap_synchronized_videos}"
            )
        return synchronized_video
    
    def get_annotated_video_by_name(self, video_name: str) -> Path:
        annotated_video = (
            self.head_body_annotated_videos.glob(video_name + "*.mp4")
            if self.head_body_annotated_videos
            else None
        )
        annotated_video = next(annotated_video, None) if annotated_video else None
        if not annotated_video:
            raise ValueError(
                f"Could not find annotated video for {video_name} in {self.head_body_annotated_videos}"
            )
        return annotated_video
    
    def get_timestamp_by_name(self, video_name: str) -> Path:
        timestamp = (
            self.mocap_synchronized_videos.glob(video_name + "*_utc.npy")
            if self.mocap_synchronized_videos
            else None
        )
        timestamp = next(timestamp, None) if timestamp else None
        if not timestamp:
            raise ValueError(
                f"Could not find timestamp for {video_name} in {self.mocap_synchronized_videos}"
            )
        return timestamp


    def check_synchronization(self):
        for name, path in {
            "eye_videos_folder": self.eye_videos,
            "left_eye_video": self.left_eye_video,
            "left_eye_timestamps_npy": self.left_eye_timestamps_npy,
            "right_eye_video": self.right_eye_video,
            "right_eye_timestamps_npy": self.right_eye_timestamps_npy,
            "mocap_synchronized_videos": self.mocap_synchronized_videos,
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, synchronization failed")
            
        for video in [
            BaslerCameras.TOPDOWN.value,
            BaslerCameras.SIDE_0.value,
            BaslerCameras.SIDE_1.value,
            BaslerCameras.SIDE_2.value,
            BaslerCameras.SIDE_3.value,
        ]:
            try:
                self.get_synchronized_video_by_name(video)
            except ValueError:
                raise ValueError(f"Could not find synchronized video for {video} in {self.mocap_synchronized_videos}")
            try:
                self.get_timestamp_by_name(video)
            except ValueError:
                raise ValueError(f"Could not find timestamp for {video} in {self.mocap_synchronized_videos}")
            try:
                self.get_annotated_video_by_name(video)
            except ValueError:
                raise ValueError(f"Could not find annotated video for {video} in {self.head_body_annotated_videos}")
            

    def check_dlc_output(self, enforce_toy: bool = True, enforce_annotated: bool = True):
        try:
            self.check_synchronization()
        except ValueError:
            raise ValueError("Synchronization failed, dlc output cannot be checked")

        for name, path in {
            "eye_dlc_output_folder": self.eye_dlc_output,
            "eye_dlc_output_flipped_folder": self.eye_dlc_output_flipped,
            "eye_skellyclicker_labels": self.eye_data_skellyclicker_labels,
            "eye_skellyclicker_labels_flipped": self.eye_data_skellyclicker_labels_flipped,
            "mocap_dlc_output_folder": self.head_body_dlc_output,
            "mocap_skellyclicker_labels": self.head_body_skellyclicker_labels
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, dlc output failed")
            
        if enforce_toy:
            for name, path in {
                "toy_dlc_output_folder": self.toy_dlc_output,
                "toy_dlc_output_flipped_folder": self.toy_skellyclicker_labels
            }.items():
                if path is None:
                    raise ValueError(f"{name} does not exist, dlc output failed")
        
        if enforce_annotated:
            for name, path in {
                "left_eye_annotated_video": self.left_eye_annotated_video,
                "left_eye_annotated_flipped_video": self.left_eye_annotated_flipped_video,
                "right_eye_annotated_video": self.right_eye_annotated_video,
                "right_eye_annotated_flipped_video": self.right_eye_annotated_flipped_video,
            }:
                if path is None:
                    raise ValueError(f"{name} does not exist, dlc output failed")
            for video in [
                BaslerCameras.TOPDOWN.value,
                BaslerCameras.SIDE_0.value,
                BaslerCameras.SIDE_1.value,
                BaslerCameras.SIDE_2.value,
                BaslerCameras.SIDE_3.value,
            ]:
                try:
                    self.get_annotated_video_by_name(video)
                except ValueError:
                    raise ValueError(f"Could not find annotated video for {video} in {self.head_body_annotated_videos}")

    def check_triangulation(self):
        try:
            self.check_dlc_output()
        except ValueError as e:
            print(f"DLC output failed with error: {e}")
            raise ValueError("DLC output failed, triangulation cannot be checked")

    def check_postprocessing(self):
        try:
            self.check_triangulation()
        except ValueError as e:
            print(f"Triangulation failed with error: {e}")
            raise ValueError("Triangulation failed, postprocessing cannot be checked")


# class RecordingFolder(BaseModel):
#     base_recordings_folder: Path
#     recording_name: str
#     clip_name: str

#     # Derived paths
#     recording_folder: Path
#     clip_folder: Path

#     eye_data_folder: Path
#     eye_annotated_videos_folder: Path
#     eye_synchronized_videos_folder: Path
#     eye_timestamps_folder: Path
#     eye_dlc_output_folder: Path
#     eye_output_data_folder: Path

#     eye_data_csv_path: Path
#     right_eye_annotated_video_path: Path
#     right_eye_video_path: Path
#     right_eye_aligned_canvas_path: Path
#     right_eye_plot_points_csv_path: Path
#     right_eye_timestamps_npy_path: Path
#     left_eye_annotated_video_path: Path
#     left_eye_video_path: Path
#     left_eye_aligned_canvas_path: Path
#     left_eye_plot_points_csv_path: Path
#     left_eye_timestamps_npy_path: Path

#     mocap_data_folder: Path
#     mocap_annotated_videos_folder: Path
#     mocap_synchronized_videos_folder: Path
#     mocap_timestamps_folder: Path
#     mocap_dlc_output_folder: Path
#     mocap_csv_data_path: Path
#     mocap_output_data_folder: Path

#     topdown_video_name: str
#     topdown_annotated_video_path: Path
#     topdown_video_path: Path
#     topdown_timestamps_npy_path: Path

#     side_0_video_name: str
#     side_0_annotated_video_path: Path
#     side_0_video_path: Path
#     side_0_timestamps_npy_path: Path

#     side_1_video_name: str
#     side_1_annotated_video_path: Path
#     side_1_video_path: Path
#     side_1_timestamps_npy_path: Path

#     side_2_video_name: str
#     side_2_annotated_video_path: Path
#     side_2_video_path: Path
#     side_2_timestamps_npy_path: Path

#     side_3_video_name: str
#     side_3_annotated_video_path: Path
#     side_3_video_path: Path
#     side_3_timestamps_npy_path: Path

#     @staticmethod
#     def get_video_paths(video_name: str,
#         annotated_videos_folder: Path,
#         synchronized_videos_folder: Path,
#         timestamps_folder: Path
#     )-> Tuple[Path, Path, Path]:
#         print(f"getting video paths for {video_name} in {synchronized_videos_folder.parent}")
#         annotated_video_path = list(annotated_videos_folder.glob(f"{video_name}*.mp4"))[0]
#         video_path = list(synchronized_videos_folder.glob(f"{video_name}*.mp4"))[0]
#         timestamps_npy_path = list(timestamps_folder.glob(f"{video_name}*utc*.npy"))[0]
#         return annotated_video_path, video_path, timestamps_npy_path

#     @classmethod
#     def create_from_clip(
#         cls,
#         recording_name: str,
#         clip_name: str,
#         base_recordings_folder: Path = Path("/Users/philipqueen/"),
#     ) -> "RecordingFolder":
#         """Create a FerretRecordingPaths instance from recording and clip names."""
#         clip_folder = Path(base_recordings_folder) / recording_name / "clips" / clip_name
#         return cls.create(recording_name, clip_name, clip_folder, base_recordings_folder)

#     @classmethod
#     def create_full_recording(
#         cls,
#         recording_name: str,
#         base_recordings_folder: Path = Path("/Users/philipqueen/"),
#     ) -> "RecordingFolder":
#         clip_folder = Path(base_recordings_folder) / recording_name / "full_recording"
#         return cls.create(recording_name, "full_recording", clip_folder, base_recordings_folder)

#     @classmethod
#     def create(
#         cls,
#         recording_name: str,
#         clip_name: str,
#         clip_folder: Path,
#         base_recordings_folder: Path = Path("/Users/philipqueen/")
#     ) -> "RecordingFolder":
#         recording_folder = Path(base_recordings_folder) / recording_name
#         print(f"Parsing recording folder: {recording_folder}")
#         eye_data_folder = clip_folder / "eye_data"
#         eye_annotated_videos_folder = eye_data_folder / "annotated_videos" / "annotated_videos_eye_model_v3"
#         eye_synchronized_videos_folder = eye_data_folder / "eye_videos"
#         eye_timestamps_folder = eye_synchronized_videos_folder
#         eye_dlc_output_folder = eye_data_folder / "dlc_output" / "eye_model_v3"
#         eye_output_data_folder = eye_data_folder / "output_data"
#         for path in [
#             eye_data_folder,
#             eye_annotated_videos_folder,
#             eye_synchronized_videos_folder,
#             eye_timestamps_folder,
#             eye_dlc_output_folder,
#         ]:
#             if not path.exists():
#                 raise ValueError(f"Path does not exist: {path}")

#         eye_data_csv_path = list(
#             eye_dlc_output_folder.glob("skellyclicker_machine_labels*.csv")
#         )[0]

#         right_eye_video_name = "eye1"
#         (
#             right_eye_annotated_video_path,
#             right_eye_video_path,
#             right_eye_timestamps_npy_path,
#         ) = cls.get_video_paths(
#             video_name=right_eye_video_name,
#             annotated_videos_folder=eye_annotated_videos_folder,
#             synchronized_videos_folder=eye_synchronized_videos_folder,
#             timestamps_folder=eye_timestamps_folder
#         )
#         right_eye_aligned_canvas_path = eye_data_folder / f"left_eye_stabilized_canvas.mp4"
#         right_eye_plot_points_csv_path = eye_data_folder / f"left_eye_plot_points.csv"

#         left_eye_video_name = "eye0"
#         (
#             left_eye_annotated_video_path,
#             left_eye_video_path,
#             left_eye_timestamps_npy_path
#         ) = cls.get_video_paths(
#             video_name=left_eye_video_name,
#             annotated_videos_folder=eye_annotated_videos_folder,
#             synchronized_videos_folder=eye_synchronized_videos_folder,
#             timestamps_folder=eye_timestamps_folder
#         )
#         left_eye_aligned_canvas_path = eye_data_folder / f"right_eye_stabilized_canvas.mp4"
#         left_eye_plot_points_csv_path = eye_data_folder / f"right_eye_plot_points.csv"

#         mocap_data_folder = clip_folder / "mocap_data"
#         mocap_annotated_videos_folder = mocap_data_folder / "annotated_videos"
#         mocap_head_body_annotated_videos_folder = mocap_annotated_videos_folder / "annotated_videos_head_body_eyecam_retrain_test_v2"
#         mocap_synchronized_videos_folder = mocap_data_folder / "synchronized_videos"
#         if not mocap_synchronized_videos_folder.exists():
#             mocap_synchronized_videos_folder = mocap_data_folder / "synchronized_corrected_videos"
#         mocap_timestamps_folder = mocap_synchronized_videos_folder
#         mocap_dlc_output_folder = mocap_data_folder / "dlc_output"
#         mocap_head_body_dlc_output = mocap_dlc_output_folder / "head_body_eyecam_retrain_test_v2"
#         mocap_output_data_folder = mocap_data_folder / "output_data"
#         for path in [
#             mocap_data_folder,
#             mocap_annotated_videos_folder,
#             mocap_synchronized_videos_folder,
#             mocap_timestamps_folder,
#             mocap_dlc_output_folder,
#         ]:
#             if not path.exists():
#                 raise ValueError(f"Path does not exist: {path}")

#         mocap_csv_path = list(
#             mocap_head_body_dlc_output.glob("skellyclicker_machine_labels*.csv")
#         )[0]
#         topdown_video_name = "24676894"
#         # topdown_video_name = "25006505"
#         (
#             topdown_annotated_video_path,
#             topdown_video_path,
#             topdown_timestamps_npy_path,
#         ) = cls.get_video_paths(
#             video_name=topdown_video_name,
#             annotated_videos_folder=mocap_head_body_annotated_videos_folder,
#             synchronized_videos_folder=mocap_synchronized_videos_folder,
#             timestamps_folder=mocap_timestamps_folder
#         )

#         side_0_video_name = "24908831"
#         (
#             side_0_annotated_video_path,
#             side_0_video_path,
#             side_0_timestamps_npy_path,
#         ) = cls.get_video_paths(
#             video_name=side_0_video_name,
#             annotated_videos_folder=mocap_head_body_annotated_videos_folder,
#             synchronized_videos_folder=mocap_synchronized_videos_folder,
#             timestamps_folder=mocap_timestamps_folder
#         )

#         side_1_video_name = "25000609"
#         (
#             side_1_annotated_video_path,
#             side_1_video_path,
#             side_1_timestamps_npy_path,
#         ) = cls.get_video_paths(
#             video_name=side_1_video_name,
#             annotated_videos_folder=mocap_head_body_annotated_videos_folder,
#             synchronized_videos_folder=mocap_synchronized_videos_folder,
#             timestamps_folder=mocap_timestamps_folder
#         )

#         side_2_video_name = "25006505"
#         (
#             side_2_annotated_video_path,
#             side_2_video_path,
#             side_2_timestamps_npy_path,
#         ) = cls.get_video_paths(
#             video_name=side_2_video_name,
#             annotated_videos_folder=mocap_head_body_annotated_videos_folder,
#             synchronized_videos_folder=mocap_synchronized_videos_folder,
#             timestamps_folder=mocap_timestamps_folder
#         )

#         side_3_video_name = "24908832"
#         (
#             side_3_annotated_video_path,
#             side_3_video_path,
#             side_3_timestamps_npy_path,
#         ) = cls.get_video_paths(
#             video_name=side_3_video_name,
#             annotated_videos_folder=mocap_head_body_annotated_videos_folder,
#             synchronized_videos_folder=mocap_synchronized_videos_folder,
#             timestamps_folder=mocap_timestamps_folder
#         )

#         return cls(
#             base_recordings_folder=base_recordings_folder,
#             recording_name=recording_name,
#             clip_name=clip_name,
#             recording_folder=recording_folder,
#             clip_folder=clip_folder,
#             eye_data_folder=eye_data_folder,
#             eye_annotated_videos_folder=eye_annotated_videos_folder,
#             eye_synchronized_videos_folder=eye_synchronized_videos_folder,
#             eye_timestamps_folder=eye_timestamps_folder,
#             eye_dlc_output_folder=eye_dlc_output_folder,
#             eye_data_csv_path=eye_data_csv_path,
#             eye_output_data_folder=eye_output_data_folder,
#             right_eye_annotated_video_path=right_eye_annotated_video_path,
#             right_eye_video_path=right_eye_video_path,
#             right_eye_aligned_canvas_path=right_eye_aligned_canvas_path,
#             right_eye_plot_points_csv_path=right_eye_plot_points_csv_path,
#             right_eye_timestamps_npy_path=right_eye_timestamps_npy_path,
#             left_eye_annotated_video_path=left_eye_annotated_video_path,
#             left_eye_video_path=left_eye_video_path,
#             left_eye_aligned_canvas_path=left_eye_aligned_canvas_path,
#             left_eye_plot_points_csv_path=left_eye_plot_points_csv_path,
#             left_eye_timestamps_npy_path=left_eye_timestamps_npy_path,
#             mocap_data_folder=mocap_data_folder,
#             mocap_annotated_videos_folder=mocap_annotated_videos_folder,
#             mocap_synchronized_videos_folder=mocap_synchronized_videos_folder,
#             mocap_timestamps_folder=mocap_timestamps_folder,
#             mocap_dlc_output_folder=mocap_dlc_output_folder,
#             mocap_csv_data_path=mocap_csv_path,
#             mocap_output_data_folder=mocap_output_data_folder,
#             topdown_video_name=topdown_video_name,
#             topdown_annotated_video_path=topdown_annotated_video_path,
#             topdown_video_path=topdown_video_path,
#             topdown_timestamps_npy_path=topdown_timestamps_npy_path,
#             side_0_video_name=side_0_video_name,
#             side_0_annotated_video_path=side_0_annotated_video_path,
#             side_0_video_path=side_0_video_path,
#             side_0_timestamps_npy_path=side_0_timestamps_npy_path,
#             side_1_video_name=side_1_video_name,
#             side_1_annotated_video_path=side_1_annotated_video_path,
#             side_1_video_path=side_1_video_path,
#             side_1_timestamps_npy_path=side_1_timestamps_npy_path,
#             side_2_video_name=side_2_video_name,
#             side_2_annotated_video_path=side_2_annotated_video_path,
#             side_2_video_path=side_2_video_path,
#             side_2_timestamps_npy_path=side_2_timestamps_npy_path,
#             side_3_video_name=side_3_video_name,
#             side_3_annotated_video_path=side_3_annotated_video_path,
#             side_3_video_path=side_3_video_path,
#             side_3_timestamps_npy_path=side_3_timestamps_npy_path,
#         )

if __name__ == "__main__":
    RecordingFolder.from_folder_path(
        "/Users/philipqueen/session_2025-07-01_ferret_757_EyeCameras_P33EO5/clips/1m_20s-2m_20s"
    )
