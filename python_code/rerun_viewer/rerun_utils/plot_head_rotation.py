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
    R = row.values.reshape(3, 3)
    
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
    if len(df.columns) != 9:
        raise ValueError("DataFrame must have exactly 9 columns")
    

    euler_df = df.apply(calulcate_yaw_pitch_roll_row, axis=1)

    result = pd.concat([df, euler_df], axis=1)
    return result

def plot_head_rotation(video_data: MocapVideoData, angle_df: pd.DataFrame, entity_path: str):
    yaw_color = (0, 255, 0)
    pitch_color = (0, 0, 255)
    roll_color = (255, 0, 0)

    euler_df = rotation_matrix_to_euler_angles(angle_df)

    for data_type, color, data in [
        ("yaw", yaw_color, euler_df["yaw"]),
        ("pitch", pitch_color, euler_df["pitch"]),
        ("roll", roll_color, euler_df["roll"])
    ]:
        entity_path = f"{entity_path}/{data_type}"
        print(f"Logging {entity_path}...")
        # Set up visualization properties
        if "line" in data_type:
            rr.log(entity_path,
                    rr.SeriesLines(colors=color,
                                    names=data_type,
                                    widths=2),
                    static=True)
        else:  # dots
            rr.log(entity_path,
                    rr.SeriesPoints(colors=color,
                                    names=data_type,
                                    markers="circle",
                                    marker_sizes=2),
                    static=True)

        rr.send_columns(
            entity_path=entity_path,
            indexes=[rr.TimeColumn("time", duration=video_data.timestamps)],
            columns=rr.Scalars.columns(scalars=data),
        )

def get_head_rotation_view(entity_path: str = ""):
    view = rrb.TimeSeriesView(origin=entity_path, 
                name="Right Eye Horizontal Position",
                contents=[f"+ right_eye/pupil_x_line",
                        f"+ right_eye/pupil_x_dots"],
                axis_y=rrb.ScalarAxis(range=(0.0, 400.0))
                )
    return view

if __name__ == "__main__":
    from python_code.rerun_viewer.rerun_utils.recording_folder import RecordingFolder
    from datetime import datetime

    recording_name = "session_2025-07-11_ferret_757_EyeCamera_P43_E15__1"
    clip_name = "0m_37s-1m_37s"
    recording_folder = RecordingFolder.create_from_clip(recording_name, clip_name, base_recordings_folder=Path("/home/scholl-lab/ferret_recordings"))
    # recording_folder = RecordingFolder.create_full_recording(recording_name, base_recordings_folder="/home/scholl-lab/ferret_recordings")

    topdown_mocap_video = MocapVideoData.create(
        annotated_video_path=recording_folder.topdown_annotated_video_path,
        raw_video_path=recording_folder.topdown_video_path,
        timestamps_npy_path=recording_folder.topdown_timestamps_npy_path,
        data_name="TopDown Mocap",
    )

    rotation_df_path = recording_folder.mocap_output_data_folder / "solver_output" / "rotation_translation_data.csv"
    rotation_df = pd.read_csv(rotation_df_path)

    recording_string = (
        f"{recording_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    head_rotation_entity_path = "/head_rotation"

    view = get_head_rotation_view(entity_path=head_rotation_entity_path)

    blueprint = rrb.Horizontal(view)

    rr.send_blueprint(blueprint)

    plot_head_rotation(video_data=topdown_mocap_video, angle_df=rotation_df, entity_path=head_rotation_entity_path)
