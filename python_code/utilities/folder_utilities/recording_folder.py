from pathlib import Path
from typing import Tuple
from enum import Enum

from pydantic import BaseModel


class PipelineStep(Enum):
    RAW = "raw"
    SYNCHRONIZED = "synchronized"
    DLCED = "DLCED"
    TRIANGULATED = "triangulated"
    EYE_POST_PROCESSED = "eye_post_processed"
    SKULL_POST_PROCESSED = "skull_post_processed"
    GAZE_POST_PROCESSED = "gaze_post_processed"


class BaslerCamera(Enum):
    TOPDOWN = "24676894"
    SIDE_0 = "24908831"
    SIDE_1 = "24908832"
    SIDE_2 = "25000609"
    SIDE_3 = "25006505"


class RecordingFolder(BaseModel):
    folder_path: Path
    base_recordings_folder: Path
    recording_name: str
    version_name: str
    is_clip: bool
    left_eye_name: str
    right_eye_name: str
    processing_step: PipelineStep = PipelineStep.RAW

    @classmethod
    def from_folder_path(cls, folder: Path | str, expected_processing_step: PipelineStep = PipelineStep.RAW) -> "RecordingFolder":
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
            version_name = folder.stem
            is_clip = False

            if version_name != "full_recording":
                print(version_name)
                raise ValueError(
                    f"Folder must be in 'clips' or be 'full_recording': {folder}"
                )

        if not (folder / "mocap_data").exists():
            raise ValueError(f"Folder does not contain mocap_data: {folder}")
        if not (folder / "eye_data").exists():
            raise ValueError(f"Folder does not contain eye_data: {folder}")
        
        if "757" in recording_name:
            left_eye_name = "eye0"
            right_eye_name = "eye1"
        else:
            left_eye_name = "eye1"
            right_eye_name = "eye0"

        recording_folder = cls(
            folder_path=folder,
            base_recordings_folder=base_recordings_folder,
            recording_name=recording_name,
            version_name=version_name,
            is_clip=is_clip,
            left_eye_name=left_eye_name,
            right_eye_name=right_eye_name
        )

        match expected_processing_step:
            case PipelineStep.GAZE_POST_PROCESSED:
                try:
                    recording_folder.check_gaze_postprocessing()
                    recording_folder.processing_step = PipelineStep.GAZE_POST_PROCESSED
                    print(f"Folder is post-processed: {folder}")
                except ValueError as e:
                    print(f"Folder is not post-processed: {e}")
                    raise ValueError(
                        f"Folder is not post-processed: {folder}"
                    )
            case PipelineStep.SKULL_POST_PROCESSED:
                try:
                    recording_folder.check_skull_postprocessing()
                    recording_folder.processing_step = PipelineStep.SKULL_POST_PROCESSED
                    print(f"Folder is skull-post-processed: {folder}")
                except ValueError as e:
                    print(f"Folder is not skull-post-processed: {e}")
                    raise ValueError(
                        f"Folder is not skull-post-processed: {folder}"
                    )
            case PipelineStep.EYE_POST_PROCESSED:
                try:
                    recording_folder.check_eye_postprocessing()
                    recording_folder.processing_step = PipelineStep.EYE_POST_PROCESSED
                    print(f"Folder is eye-post-processed: {folder}")
                except ValueError as e:
                    print(f"Folder is not eye-post-processed: {e}")
                    raise ValueError(
                        f"Folder is not eye-post-processed: {folder}"
                    )
            case PipelineStep.TRIANGULATED:
                try:
                    recording_folder.check_triangulation()
                    recording_folder.processing_step = PipelineStep.TRIANGULATED
                    print(f"Folder is triangulated: {folder}")
                except ValueError as e:
                    print(f"Folder is not triangulated: {e}")
                    raise ValueError(
                        f"Folder is not triangulated: {folder}"
                    )
            case PipelineStep.DLCED:
                try:
                    recording_folder.check_dlc_output()
                    recording_folder.processing_step = PipelineStep.DLCED
                    print(f"Folder is DLCed: {folder}")
                except ValueError as e:
                    print(f"Folder is not DLCed: {e}")
                    raise ValueError(
                        f"Folder is not DLCed: {folder}"
                    )
            case PipelineStep.SYNCHRONIZED:
                try:
                    recording_folder.check_synchronization()
                    recording_folder.processing_step = PipelineStep.SYNCHRONIZED
                except ValueError as e:
                    print(f"Folder is not synchronized: {e}")
                    raise ValueError(
                        f"Folder is not synchronized: {folder}"
                    )
            case PipelineStep.RAW:
                pass
            case _:
                raise ValueError(f"Unknown processing step: {expected_processing_step}")

        return recording_folder

    @property
    def mocap_data(self) -> Path:
        return self.folder_path / "mocap_data"

    @property
    def eye_data(self) -> Path:
        return self.folder_path / "eye_data"

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
        eye_videos = self.eye_data / "eye_videos"
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
        eye_mean_confidence = self.eye_data / "eye_model_v3_mean_confidence.csv"
        return eye_mean_confidence if eye_mean_confidence.exists() else None

    @property
    def right_eye_video(self) -> Path | None:
        right_eye_video = self.eye_videos / f"{self.right_eye_name}.mp4" if self.eye_videos else None
        return right_eye_video if right_eye_video and right_eye_video.exists() else None

    @property
    def right_eye_annotated_video(self) -> Path | None:
        right_eye_annotated_video = (
            self.eye_annotated_videos / f"{self.right_eye_name}.mp4"
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
            self.eye_annotated_flipped / f"{self.right_eye_name}.mp4"
            if self.eye_annotated_flipped
            else None
        )
        if right_eye_annotated_video and not right_eye_annotated_video.exists():
            right_eye_annotated_video = (
                self.eye_annotated_flipped / f"{self.right_eye_name}_flipped.mp4"
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
            self.eye_videos / f"{self.right_eye_name}_timestamps_utc.npy" if self.eye_videos else None
        )
        return (
            right_eye_timestamps_npy
            if right_eye_timestamps_npy and right_eye_timestamps_npy.exists()
            else None
        )

    @property
    def right_eye_plot_points_csv(self) -> Path | None:
        right_eye_plot_points_csv = (
            self.eye_data / "right_eye_plot_points.csv"
            if self.eye_data
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
            self.eye_data / "right_eye_stabilized_canvas.mp4"
            if self.eye_data
            else None
        )
        return (
            right_eye_aligned_canvas
            if right_eye_aligned_canvas and right_eye_aligned_canvas.exists()
            else None
        )

    @property
    def left_eye_video(self) -> Path | None:
        left_eye_video = self.eye_videos / f"{self.left_eye_name}.mp4" if self.eye_videos else None
        return left_eye_video if left_eye_video and left_eye_video.exists() else None

    @property
    def left_eye_annotated_video(self) -> Path | None:
        left_eye_annotated_video = (
            self.eye_annotated_videos / f"{self.left_eye_name}.mp4"
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
            self.eye_annotated_flipped / f"{self.left_eye_name}.mp4"
            if self.eye_annotated_flipped
            else None
        )
        if left_eye_annotated_video and not left_eye_annotated_video.exists():
            left_eye_annotated_video = (
                self.eye_annotated_flipped / f"{self.left_eye_name}_flipped.mp4"
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
            self.eye_videos / f"{self.left_eye_name}_timestamps_utc.npy" if self.eye_videos else None
        )
        return (
            left_eye_timestamps_npy
            if left_eye_timestamps_npy and left_eye_timestamps_npy.exists()
            else None
        )

    @property
    def left_eye_plot_points_csv(self) -> Path | None:
        left_eye_plot_points_csv = (
            self.eye_data / "left_eye_plot_points.csv"
            if self.eye_data
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
            self.eye_data / "left_eye_stabilized_canvas.mp4"
            if self.eye_data
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
        toy_machine_labels = (
            self.toy_dlc_output.glob("skellyclicker_machine_labels*.csv")
            if self.toy_dlc_output
            else None
        )
        return next(toy_machine_labels, None) if toy_machine_labels else None

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
    def toy_3d_data(self) -> Path | None:
        toy_3d_data = (
            self.toy_dlc_output / "dlc" if self.toy_dlc_output else None
        )
        return toy_3d_data if toy_3d_data and toy_3d_data.exists() else None

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
            BaslerCamera.TOPDOWN.value,
            BaslerCamera.SIDE_0.value,
            BaslerCamera.SIDE_1.value,
            BaslerCamera.SIDE_2.value,
            BaslerCamera.SIDE_3.value,
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
                "toy_skellyclicker_labels": self.toy_skellyclicker_labels
            }.items():
                if path is None:
                    raise ValueError(f"{name} does not exist, dlc output failed")
        
        if enforce_annotated:
            for name, path in {
                "left_eye_annotated_video": self.left_eye_annotated_video,
                "left_eye_annotated_flipped_video": self.left_eye_annotated_flipped_video,
                "right_eye_annotated_video": self.right_eye_annotated_video,
                "right_eye_annotated_flipped_video": self.right_eye_annotated_flipped_video,
            }.items():
                if path is None:
                    raise ValueError(f"{name} does not exist, dlc output failed")
            for video in [
                BaslerCamera.TOPDOWN.value,
                BaslerCamera.SIDE_0.value,
                BaslerCamera.SIDE_1.value,
                BaslerCamera.SIDE_2.value,
                BaslerCamera.SIDE_3.value,
            ]:
                try:
                    self.get_annotated_video_by_name(video)
                except ValueError:
                    raise ValueError(f"Could not find annotated video for {video} in {self.head_body_annotated_videos}")

    def check_triangulation(self, enforce_toy: bool = True, enforce_annotated: bool = True):
        try:
            self.check_dlc_output(enforce_toy=enforce_toy)
        except ValueError as e:
            print(f"DLC output failed with error: {e}")
            raise ValueError("DLC output failed, triangulation cannot be checked")

        for name, path in {
            "output_data_folder": self.mocap_output_data,
            "mocap_3d_data": self.mocap_3d_data,
            "head_body_3d_xyz.csv": self.mocap_3d_data / "head_body_3d_xyz.csv" if self.mocap_3d_data else None,
            "head_body_3d_xyz.npy": self.mocap_3d_data / "head_body_3d_xyz.npy" if self.mocap_3d_data else None,
            "head_body_rigid_3d_xyz.csv": self.mocap_3d_data / "head_body_rigid_3d_xyz.csv" if self.mocap_3d_data else None,
            "head_body_rigid_3d_xyz.npy": self.mocap_3d_data / "head_body_rigid_3d_xyz.npy" if self.mocap_3d_data else None,
            "head_freemocap_data_by_frame.csv": self.mocap_3d_data / "head_freemocap_data_by_frame.csv" if self.mocap_3d_data else None,
            "head_freemocap_data_by_frame.parquet": self.mocap_3d_data / "head_freemocap_data_by_frame.parquet" if self.mocap_3d_data else None,
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, triangulation failed")

        if enforce_toy:
            for name, path in {
                "toy_body_3d_xyz.csv": self.mocap_3d_data / "head_body_3d_xyz.csv" if self.mocap_3d_data else None,
                "toy_body_3d_xyz.npy": self.mocap_3d_data / "head_body_3d_xyz.npy" if self.mocap_3d_data else None,
                "toy_body_rigid_3d_xyz.csv": self.mocap_3d_data / "head_body_rigid_3d_xyz.csv" if self.mocap_3d_data else None,
                "toy_body_rigid_3d_xyz.npy": self.mocap_3d_data / "head_body_rigid_3d_xyz.npy" if self.mocap_3d_data else None,
                "toy_freemocap_data_by_frame.csv": self.mocap_3d_data / "head_freemocap_data_by_frame.csv" if self.mocap_3d_data else None,
                "toy_freemocap_data_by_frame.parquet": self.mocap_3d_data / "head_freemocap_data_by_frame.parquet" if self.mocap_3d_data else None,
            }.items():
                if path is None:
                    raise ValueError(f"{name} does not exist, triangulation failed")
                

    def check_eye_postprocessing(self):
        for name, path in {
            "eye_data.csv": self.eye_data_csv,
            "eye_model_v3_mean_confidence.csv": self.eye_mean_confidence,
            "left_eye_plot_points.csv": self.left_eye_plot_points_csv,
            "right_eye_plot_points.csv": self.right_eye_plot_points_csv,
            "left_eye_stabilized_canvas.mp4": self.left_eye_stabilized_canvas,
            "right_eye_stabilized_canvas.mp4": self.right_eye_stabilized_canvas,
            "eye_output_data": self.eye_output_data,
            "eye0_alignment_summary.json": self.eye_output_data / "eye0_alignment_summary.json" if self.eye_output_data else None,
            "eye1_alignment_summary.json": self.eye_output_data / "eye1_alignment_summary.json" if self.eye_output_data else None,
            "eye0_data.csv": self.eye_output_data / "eye0_data.csv" if self.eye_output_data else None,
            "eye1_data.csv": self.eye_output_data / "eye1_data.csv" if self.eye_output_data else None,
            "eye0_correction_comparison.png": self.eye_output_data / "eye0_correction_comparison.png" if self.eye_output_data else None,
            "eye1_correction_comparison.png": self.eye_output_data / "eye1_correction_comparison.png" if self.eye_output_data else None,
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, eye postprocessing failed")
        

    def check_skull_postprocessing(self, enforce_toy: bool = True, enforce_annotated: bool = True):
        try:
            self.check_triangulation(enforce_toy=enforce_toy, enforce_annotated=enforce_annotated)
        except ValueError as e:
            print(f"Triangulation failed with error: {e}")
            raise ValueError("Triangulation failed, postprocessing cannot be checked")

        for name, path in {
            "solver_output_folder": self.mocap_solver_output,
            "reference_geometry": self.mocap_solver_output / "reference_geometry.json" if self.mocap_solver_output else None,
            "reference_geometry_skull": self.mocap_solver_output / "reference_geometry_skull.json" if self.mocap_solver_output else None,
            "rotation_translation_data.csv": self.mocap_solver_output / "rotation_translation_data.csv" if self.mocap_solver_output else None,
            "tidy_trajectory_data.csv": self.mocap_solver_output / "tidy_trajectory_data.csv" if self.mocap_solver_output else None,
            "trajectory_data.csv": self.mocap_solver_output / "trajectory_data.csv" if self.mocap_solver_output else None,
            "topology.json": self.mocap_solver_output / "topology.json" if self.mocap_solver_output else None,
            "metrics.json": self.mocap_solver_output / "metrics.json" if self.mocap_solver_output else None,
        }.items():
            if path is None:
                raise ValueError(f"{name} does not exist, head solver failed")
            
    
    def check_gaze_postprocessing(self, enforce_toy: bool = True, enforce_annotated: bool = True):
        try:
            self.check_eye_postprocessing
        except ValueError as e:
            print(f"eyes not postprocessed: {e}")
            raise ValueError("Eyes are not postprocessed, unable to check gaze postprocessing")
        
        try:
            self.check_skull_postprocessing
        except ValueError as e:
            print(f"skull not postprocessed: {e}")
            raise ValueError("skull not postprocessed, unable to check gaze postprocessing")
        
        # TODO: add analyzable output folder

        # TODO: add properties for each check


    def csv_report(self):
        pass # TODO: implement a csv report that can be passed into a dataframe easily

if __name__ == "__main__":
    RecordingFolder.from_folder_path(
        "/home/scholl-lab/ferret_recordings/session_2025-10-18_ferret_420_E09/full_recording",
        expected_processing_step=PipelineStep.GAZE_POST_PROCESSED
    )
