from pathlib import Path
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D

from python_code.rerun_viewer.rerun_utils.process_videos import process_video
from python_code.rerun_viewer.rerun_utils.video_data import MocapVideoData

def calulcate_yaw_pitch_roll_row(row):
    R = row.iloc[3:12].values.reshape(3, 3)
    
    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]
    
    sy = np.sqrt(r11*r11 + r12*r12)
    
    if sy < 1e-6:  # If close to zero
        x = np.arctan2(r21, r22)
        y = np.arctan2(-r31, sy)
        z = 0
    else:
        x = np.arctan2(r32, r33)
        y = np.arctan2(-r31, sy)
        z = np.arctan2(r12, r11)
        
    series = pd.Series({'yaw': z, 'pitch': y, 'roll': x})
    series = np.degrees(series)
    return series

def rotation_matrix_to_euler_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 3x3 rotation matrices stored in DataFrame columns to Euler angles.
    
    Parameters:
        df: DataFrame with 9 columns representing a 3x3 rotation matrix
        
    Returns:
        DataFrame with added yaw, pitch, roll columns in radians
    """
    euler_df = df.apply(calulcate_yaw_pitch_roll_row, axis=1)

    result = pd.concat([df, euler_df], axis=1)
    return result

def plot_head_rotation(video_data: MocapVideoData, angle_df: pd.DataFrame, entity_path: str):
    yaw_color = (0, 255, 0)
    pitch_color = (0, 0, 255)
    roll_color = (255, 0, 0)

    euler_df = rotation_matrix_to_euler_angles(angle_df)

    for data_type, color, data in [
        ("yaw_lines", yaw_color, euler_df["yaw"]),
        ("yaw_dots", yaw_color, euler_df["yaw"]),
        ("pitch_lines", pitch_color, euler_df["pitch"]),
        ("pitch_dots", pitch_color, euler_df["pitch"]),
        ("roll_lines", roll_color, euler_df["roll"]),
        ("roll_dots", roll_color, euler_df["roll"])
    ]:
        entity_subpath = f"{entity_path}/{data_type}"
        print(f"Logging {entity_subpath}...")
        # Set up visualization properties
        if "line" in data_type:
            rr.log(entity_subpath,
                    rr.SeriesLines(colors=color,
                                    names=data_type,
                                    widths=2),
                    static=True)
        else:  # dots
            rr.log(entity_subpath,
                    rr.SeriesPoints(colors=color,
                                    names=data_type,
                                    markers="circle",
                                    marker_sizes=2),
                    static=True)

        rr.send_columns(
            entity_path=entity_subpath,
            indexes=[rr.TimeColumn("time", duration=video_data.timestamps)],
            columns=rr.Scalars.columns(scalars=data),
        )

def get_head_rotation_view(entity_path: str = ""):
    view = rrb.TimeSeriesView(origin=entity_path, 
                name="head rotation",
                # contents=[f"+ right_eye/pupil_x_line",
                #         f"+ right_eye/pupil_x_dots"],
                axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                )
    return view

if __name__ == "__main__":
    from python_code.utilities.folder_utilities.recording_folder import RecordingFolder, BaslerCamera
    from datetime import datetime

    folder_path = Path(
        "/home/scholl-lab/ferret_recordings/session_2025-10-18_ferret_420_E09/full_recording"
    )
    recording_folder = RecordingFolder.from_folder_path(folder_path)
    recording_folder.check_skull_postprocessing(enforce_toy=False, enforce_annotated=True)

    topdown_synchronized_video = recording_folder.get_synchronized_video_by_name(BaslerCamera.TOPDOWN.value)
    topdown_annotated_video = recording_folder.get_annotated_video_by_name(BaslerCamera.TOPDOWN.value)
    topdown_timestamps_npy = recording_folder.get_timestamp_by_name(BaslerCamera.TOPDOWN.value)

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=topdown_annotated_video,
        raw_video_path=topdown_synchronized_video,
        timestamps_npy_path=topdown_timestamps_npy,
        data_name="TopDown Mocap",
    )

    recording_string = (
        f"{recording_folder.recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    rotation_df_path = recording_folder.mocap_solver_output / "rotation_translation_data.csv"
    rotation_df = pd.read_csv(rotation_df_path)

    rr.init(recording_string, spawn=True)

    head_rotation_entity_path = "/head_rotation"

    view = get_head_rotation_view(entity_path=head_rotation_entity_path)

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)

    plot_head_rotation(video_data=topdown_mocap_video, angle_df=rotation_df, entity_path=head_rotation_entity_path)
