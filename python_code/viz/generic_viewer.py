from datetime import datetime
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from python_code.utilities.folder_utilities.recording_folder import PipelineStep, RecordingFolder
from python_code.viz.rerun_viewer.components.registry import (
    COMPONENT_FUNCS,
    COMPONENT_REGISTRY,
    ComponentId,
    PanelSlot,
)


def _context_kwargs(component_id: ComponentId, eye_name: str | None, entity_path: str) -> dict:
    kwargs = {"entity_path": entity_path}
    if COMPONENT_REGISTRY[component_id].per_eye:
        kwargs["eye_name"] = eye_name
    return kwargs


def _required_data_available(component_id: ComponentId, recording_folder: RecordingFolder, context_kwargs: dict) -> bool:
    """Build each component's own Context model to run its required_data_available check."""
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

    context_classes = {
        ComponentId.EYE_VIDEO: eye_video.EyeVideoContext,
        ComponentId.EYE_3D: eye_3d.Eye3DContext,
        ComponentId.EYE_KINEMATICS_TRACES: eye_kinematics_traces.EyeKinematicsTracesContext,
        ComponentId.EYE_PUPIL_PIXEL_TRACES: eye_pupil_pixel_traces.EyePupilPixelTracesContext,
        ComponentId.EYE_VIDEO_LANDMARKS: eye_video_landmarks.EyeVideoLandmarksContext,
        ComponentId.EYE_DATA_QUALITY: eye_data_quality.EyeDataQualityContext,
        ComponentId.SKULL_AND_SPINE_3D: skull_and_spine_3d.SkullAndSpine3DContext,
        ComponentId.SKULL_AND_SPINE_TRACES: skull_and_spine_traces.SkullAndSpineTracesContext,
        ComponentId.GAZE_TRACES: gaze_traces.GazeTracesContext,
        ComponentId.NAIVE_GAZE_TRACES: naive_gaze_traces.NaiveGazeTracesContext,
        ComponentId.TRACKED_PUPIL_POINTS_3D: tracked_pupil_points_3d.TrackedPupilPoints3DContext,
        ComponentId.HEAD_ROTATION: head_rotation.HeadRotationContext,
        ComponentId.MOCAP_VIDEO: mocap_video.MocapVideoContext,
        ComponentId.WORLD_VIDEO: world_video.WorldVideoContext,
        ComponentId.DATA_3D: data_3d.Data3DContext,
    }
    context = context_classes[component_id](**context_kwargs)
    _, _, _, required_data_available = COMPONENT_FUNCS[component_id]
    return required_data_available(recording_folder, context)


def run_generic_viewer(
    recording_folder: RecordingFolder,
    selected_components: list[ComponentId],
    eyes: list[str] = ["left"],
    left_entity_path: str = "/left_panel",
    right_entity_path: str = "/right_panel",
) -> None:
    """Compose an arbitrary set of components into one Rerun viewer.

    Any component whose required data is missing on `recording_folder` is skipped
    with a warning rather than aborting the whole session.
    """
    instances: list[tuple[ComponentId, str | None, dict]] = []
    for component_id in selected_components:
        spec = COMPONENT_REGISTRY[component_id]
        entity_path = left_entity_path if spec.slot == PanelSlot.LEFT_VIDEO_3D else right_entity_path
        if spec.per_eye:
            for eye_name in eyes:
                instances.append((component_id, eye_name, _context_kwargs(component_id, eye_name, entity_path)))
        else:
            instances.append((component_id, None, _context_kwargs(component_id, None, entity_path)))

    included: list[tuple[ComponentId, str | None, dict]] = []
    skipped: list[str] = []
    for component_id, eye_name, context_kwargs in instances:
        spec = COMPONENT_REGISTRY[component_id]
        label = f"{spec.display_name} ({eye_name})" if eye_name else spec.display_name
        if _required_data_available(component_id, recording_folder, context_kwargs):
            included.append((component_id, eye_name, context_kwargs))
        else:
            skipped.append(f"Skipping {label}: required data not found")
            print(f"WARNING: Skipping {label}: required data not found")

    included_ids = {component_id for component_id, _, _ in included}
    final_included = []
    for component_id, eye_name, context_kwargs in included:
        spec = COMPONENT_REGISTRY[component_id]
        label = f"{spec.display_name} ({eye_name})" if eye_name else spec.display_name
        if spec.requires_view is not None and spec.requires_view not in included_ids:
            skipped.append(f"Skipping {label}: required view component ({spec.requires_view.value}) not included")
            print(f"WARNING: Skipping {label}: required view component ({spec.requires_view.value}) not included")
            continue
        final_included.append((component_id, eye_name, context_kwargs))

    left_views = []
    right_views = []
    for component_id, eye_name, context_kwargs in final_included:
        spec = COMPONENT_REGISTRY[component_id]
        get_view, _, _, _ = COMPONENT_FUNCS[component_id]
        if get_view is None:
            continue
        view = get_view(**context_kwargs)
        target = left_views if spec.slot == PanelSlot.LEFT_VIDEO_3D else right_views
        if isinstance(view, list):
            target.extend(view)
        else:
            target.append(view)

    panels = []
    if left_views:
        panels.append(rrb.Vertical(*left_views))
    if right_views:
        panels.append(rrb.Vertical(*right_views))
    if not panels:
        raise ValueError("No components produced a view - nothing to display")

    blueprint = rrb.Horizontal(*panels)

    recording_string = f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    rr.init(recording_string, spawn=True)
    rr.send_blueprint(blueprint)

    for component_id, eye_name, context_kwargs in final_included:
        _, _, log_style, _ = COMPONENT_FUNCS[component_id]
        if log_style is None:
            continue
        try:
            log_style(**context_kwargs)
        except Exception as e:
            print(f"WARNING: log_style failed for {component_id.value} ({eye_name}): {e}")

    for component_id, eye_name, context_kwargs in final_included:
        _, plot, _, _ = COMPONENT_FUNCS[component_id]
        spec = COMPONENT_REGISTRY[component_id]
        label = f"{spec.display_name} ({eye_name})" if eye_name else spec.display_name
        try:
            plot(recording_folder=recording_folder, **context_kwargs)
        except Exception as e:
            print(f"WARNING: Failed to plot {label}, continuing: {e}")

    print("\n--- Generic Viewer Summary ---")
    for component_id, eye_name, _ in final_included:
        spec = COMPONENT_REGISTRY[component_id]
        label = f"{spec.display_name} ({eye_name})" if eye_name else spec.display_name
        print(f"  INCLUDED: {label}")
    for reason in skipped:
        print(f"  SKIPPED:  {reason}")


if __name__ == "__main__":
    # This is the declarative config a non-technical user edits: pick a recording,
    # a pipeline stage, and the list of components to show. No new code required.
    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s"
    )
    recording_folder = RecordingFolder.from_folder_path(
        folder_path, expected_processing_step=PipelineStep.GAZE_POST_PROCESSED
    )

    run_generic_viewer(
        recording_folder,
        selected_components=[
            ComponentId.EYE_VIDEO,
            ComponentId.EYE_3D,
            ComponentId.SKULL_AND_SPINE_3D,
            ComponentId.EYE_KINEMATICS_TRACES,
            ComponentId.SKULL_AND_SPINE_TRACES,
            ComponentId.GAZE_TRACES,
            ComponentId.NAIVE_GAZE_TRACES,
        ],
        eyes=["left"],
    )
