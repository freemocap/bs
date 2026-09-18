from enum import Enum
from typing import Callable

from pydantic import BaseModel

from python_code.utilities.folder_utilities.recording_folder import PipelineStep, RecordingFolder

from python_code.viz.rerun_viewer.components import (
    data_3d,
    eye_3d,
    eye_data_quality,
    eye_kinematics_traces,
    eye_pupil_pixel_traces,
    eye_video,
    eye_video_landmarks,
    gaze_traces,
    head_rotation,
    mocap_video,
    naive_gaze_traces,
    skull_and_spine_3d,
    skull_and_spine_traces,
    tracked_pupil_points_3d,
    world_video,
)


class ComponentId(Enum):
    EYE_VIDEO = "eye_video"
    EYE_3D = "eye_3d"
    EYE_KINEMATICS_TRACES = "eye_kinematics_traces"
    EYE_PUPIL_PIXEL_TRACES = "eye_pupil_pixel_traces"
    EYE_VIDEO_LANDMARKS = "eye_video_landmarks"
    EYE_DATA_QUALITY = "eye_data_quality"
    SKULL_AND_SPINE_3D = "skull_and_spine_3d"
    SKULL_AND_SPINE_TRACES = "skull_and_spine_traces"
    GAZE_TRACES = "gaze_traces"
    NAIVE_GAZE_TRACES = "naive_gaze_traces"
    TRACKED_PUPIL_POINTS_3D = "tracked_pupil_points_3d"
    HEAD_ROTATION = "head_rotation"
    MOCAP_VIDEO = "mocap_video"
    WORLD_VIDEO = "world_video"
    DATA_3D = "data_3d"


class PanelSlot(Enum):
    LEFT_VIDEO_3D = "left_video_3d"
    RIGHT_TIMESERIES = "right_timeseries"


class ComponentSpec(BaseModel):
    id: ComponentId
    display_name: str
    required_stage: PipelineStep
    slot: PanelSlot = PanelSlot.RIGHT_TIMESERIES
    requires_view: ComponentId | None = None
    per_eye: bool = False


COMPONENT_REGISTRY: dict[ComponentId, ComponentSpec] = {
    ComponentId.EYE_VIDEO: ComponentSpec(
        id=ComponentId.EYE_VIDEO,
        display_name="Eye Video",
        required_stage=PipelineStep.GAZE_POST_PROCESSED,
        slot=PanelSlot.LEFT_VIDEO_3D,
        per_eye=True,
    ),
    ComponentId.EYE_3D: ComponentSpec(
        id=ComponentId.EYE_3D,
        display_name="Eye 3D",
        required_stage=PipelineStep.EYE_POST_PROCESSED,
        slot=PanelSlot.LEFT_VIDEO_3D,
        per_eye=True,
    ),
    ComponentId.EYE_KINEMATICS_TRACES: ComponentSpec(
        id=ComponentId.EYE_KINEMATICS_TRACES,
        display_name="Eye Kinematics Traces",
        required_stage=PipelineStep.EYE_POST_PROCESSED,
        slot=PanelSlot.RIGHT_TIMESERIES,
        per_eye=True,
    ),
    ComponentId.EYE_PUPIL_PIXEL_TRACES: ComponentSpec(
        id=ComponentId.EYE_PUPIL_PIXEL_TRACES,
        display_name="Eye Pupil Pixel Traces",
        required_stage=PipelineStep.TRIANGULATED,
        slot=PanelSlot.RIGHT_TIMESERIES,
    ),
    ComponentId.EYE_VIDEO_LANDMARKS: ComponentSpec(
        id=ComponentId.EYE_VIDEO_LANDMARKS,
        display_name="Eye Video with Landmarks",
        required_stage=PipelineStep.EYE_POST_PROCESSED,
        slot=PanelSlot.LEFT_VIDEO_3D,
        per_eye=True,
    ),
    ComponentId.EYE_DATA_QUALITY: ComponentSpec(
        id=ComponentId.EYE_DATA_QUALITY,
        display_name="Eye Data Quality",
        required_stage=PipelineStep.EYE_POST_PROCESSED,
        slot=PanelSlot.RIGHT_TIMESERIES,
        per_eye=True,
    ),
    ComponentId.SKULL_AND_SPINE_3D: ComponentSpec(
        id=ComponentId.SKULL_AND_SPINE_3D,
        display_name="Skull and Spine 3D",
        required_stage=PipelineStep.SKULL_POST_PROCESSED,
        slot=PanelSlot.LEFT_VIDEO_3D,
    ),
    ComponentId.SKULL_AND_SPINE_TRACES: ComponentSpec(
        id=ComponentId.SKULL_AND_SPINE_TRACES,
        display_name="Skull and Spine Traces",
        required_stage=PipelineStep.SKULL_POST_PROCESSED,
        slot=PanelSlot.RIGHT_TIMESERIES,
    ),
    ComponentId.GAZE_TRACES: ComponentSpec(
        id=ComponentId.GAZE_TRACES,
        display_name="Gaze Traces",
        required_stage=PipelineStep.GAZE_POST_PROCESSED,
        slot=PanelSlot.RIGHT_TIMESERIES,
        per_eye=True,
    ),
    ComponentId.NAIVE_GAZE_TRACES: ComponentSpec(
        id=ComponentId.NAIVE_GAZE_TRACES,
        display_name="Naive Gaze Traces",
        required_stage=PipelineStep.SKULL_POST_PROCESSED,
        slot=PanelSlot.RIGHT_TIMESERIES,
        per_eye=True,
    ),
    ComponentId.TRACKED_PUPIL_POINTS_3D: ComponentSpec(
        id=ComponentId.TRACKED_PUPIL_POINTS_3D,
        display_name="Tracked Pupil Points 3D",
        required_stage=PipelineStep.GAZE_POST_PROCESSED,
        slot=PanelSlot.LEFT_VIDEO_3D,
        requires_view=ComponentId.SKULL_AND_SPINE_3D,
        per_eye=True,
    ),
    ComponentId.HEAD_ROTATION: ComponentSpec(
        id=ComponentId.HEAD_ROTATION,
        display_name="Head Rotation",
        required_stage=PipelineStep.SKULL_POST_PROCESSED,
        slot=PanelSlot.RIGHT_TIMESERIES,
    ),
    ComponentId.MOCAP_VIDEO: ComponentSpec(
        id=ComponentId.MOCAP_VIDEO,
        display_name="Mocap Video",
        required_stage=PipelineStep.TRIANGULATED,
        slot=PanelSlot.LEFT_VIDEO_3D,
    ),
    ComponentId.WORLD_VIDEO: ComponentSpec(
        id=ComponentId.WORLD_VIDEO,
        display_name="World Video",
        required_stage=PipelineStep.SYNCHRONIZED,
        slot=PanelSlot.LEFT_VIDEO_3D,
    ),
    ComponentId.DATA_3D: ComponentSpec(
        id=ComponentId.DATA_3D,
        display_name="3D Data",
        required_stage=PipelineStep.TRIANGULATED,
        slot=PanelSlot.LEFT_VIDEO_3D,
    ),
}

# Function references kept separate from ComponentSpec since pydantic models
# should not hold raw callables. Each entry: (get_view, plot, log_style | None,
# required_data_available). get_view/plot/log_style take positional args matching
# each component's own flat signature (eye_name/entity_path, or recording_folder + context
# for the recording_folder-driven plot function).
COMPONENT_FUNCS: dict[ComponentId, tuple[Callable, Callable, Callable | None, Callable]] = {
    ComponentId.EYE_VIDEO: (eye_video.get_eye_video_view, eye_video.plot_eye_video, None, eye_video.required_data_available),
    ComponentId.EYE_3D: (eye_3d.get_3d_eye_view, eye_3d.plot_3d_eye, None, eye_3d.required_data_available),
    ComponentId.EYE_KINEMATICS_TRACES: (
        eye_kinematics_traces.get_eye_trace_views,
        eye_kinematics_traces.plot_eye_traces,
        eye_kinematics_traces.log_eye_trace_style,
        eye_kinematics_traces.required_data_available,
    ),
    ComponentId.EYE_PUPIL_PIXEL_TRACES: (
        eye_pupil_pixel_traces.get_eye_pupil_pixel_traces_view,
        eye_pupil_pixel_traces.plot_eye_pupil_pixel_traces,
        None,
        eye_pupil_pixel_traces.required_data_available,
    ),
    ComponentId.EYE_VIDEO_LANDMARKS: (
        eye_video_landmarks.get_eye_video_landmarks_view,
        eye_video_landmarks.plot_eye_video_landmarks,
        eye_video_landmarks.log_eye_video_landmarks_style,
        eye_video_landmarks.required_data_available,
    ),
    ComponentId.EYE_DATA_QUALITY: (
        eye_data_quality.get_eye_quality_view,
        eye_data_quality.plot_eye_data_quality,
        eye_data_quality.log_eye_quality_style,
        eye_data_quality.required_data_available,
    ),
    ComponentId.SKULL_AND_SPINE_3D: (
        skull_and_spine_3d.get_ferret_skull_and_spine_3d_view,
        skull_and_spine_3d.plot_ferret_skull_and_spine_3d,
        skull_and_spine_3d.log_ferret_skull_and_spine_3d_style,
        skull_and_spine_3d.required_data_available,
    ),
    ComponentId.SKULL_AND_SPINE_TRACES: (
        skull_and_spine_traces.get_ferret_skull_and_spine_traces_views,
        skull_and_spine_traces.plot_ferret_skull_and_spine_traces,
        skull_and_spine_traces.log_ferret_skull_and_spine_traces_style,
        skull_and_spine_traces.required_data_available,
    ),
    ComponentId.GAZE_TRACES: (
        gaze_traces.get_gaze_trace_views,
        gaze_traces.plot_gaze_traces,
        gaze_traces.log_gaze_trace_style,
        gaze_traces.required_data_available,
    ),
    ComponentId.NAIVE_GAZE_TRACES: (
        naive_gaze_traces.get_naive_gaze_trace_views,
        naive_gaze_traces.plot_naive_gaze_traces,
        naive_gaze_traces.log_naive_gaze_trace_style,
        naive_gaze_traces.required_data_available,
    ),
    ComponentId.TRACKED_PUPIL_POINTS_3D: (
        None,
        tracked_pupil_points_3d.plot_tracked_pupil_points_3d,
        tracked_pupil_points_3d.log_tracked_pupil_points_3d_style,
        tracked_pupil_points_3d.required_data_available,
    ),
    ComponentId.HEAD_ROTATION: (
        head_rotation.get_head_rotation_view,
        head_rotation.plot_head_rotation_for_recording,
        None,
        head_rotation.required_data_available,
    ),
    ComponentId.MOCAP_VIDEO: (
        None,
        mocap_video.plot_mocap_video,
        None,
        mocap_video.required_data_available,
    ),
    ComponentId.WORLD_VIDEO: (
        None,
        world_video.plot_world_video,
        None,
        world_video.required_data_available,
    ),
    ComponentId.DATA_3D: (
        None,
        data_3d.plot_data_3d,
        None,
        data_3d.required_data_available,
    ),
}
