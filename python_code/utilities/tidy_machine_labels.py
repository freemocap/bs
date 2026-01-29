import numpy as np
import pandas as pd
from pathlib import Path

def tidy_head_data(labels_path: Path, timestamps_path: Path):
    df_with_mean = pd.read_csv(labels_path)
    print(df_with_mean.head(5))
    print(df_with_mean.columns)

    # calculate mean
    selected_columns_x = ['nose_x','left_cam_tip_x', 'right_cam_tip_x', 'base_x', 'left_eye_x', 'right_eye_x', 'left_ear_x', 'right_ear_x']
    selected_columns_y = ['nose_y','left_cam_tip_y', 'right_cam_tip_y', 'base_y', 'left_eye_y', 'right_eye_y', 'left_ear_y', 'right_ear_y']

    print(selected_columns_y)

    df_with_mean['head_mean_x'] = df_with_mean[selected_columns_x].mean(axis=1)
    df_with_mean['head_mean_y'] = df_with_mean[selected_columns_y].mean(axis=1)

    df_with_mean = df_with_mean.sort_values("video")

    data_length = len(df_with_mean)

    print(f"Loading timestamps from {timestamps_path}")
    timestamp_files = list(timestamps_path.glob("*.npy"))
    timestamp_files.sort(key = lambda path : path.stem)
    timestamp_arrays = [np.load(path) for path in timestamp_files]
    timestamps = np.concatenate(timestamp_arrays, axis=0)
    if timestamps.shape[0] != data_length:
        raise ValueError(f"dataframe size {data_length} does not match timestamp shape {timestamps.shape}")

    tidy_df_list = []
    for marker in ["nose", "left_cam_tip", "right_cam_tip", "base", "left_eye", "right_eye", "left_ear", "right_ear", "head_mean"]:
        df = pd.DataFrame()
        df["frame"] = df_with_mean["frame"]
        df["keypoint"] = np.array([marker] * data_length)
        df["video"] = df_with_mean["video"]
        df["timestamp"] = timestamps / 1e9
        df["x"] = df_with_mean[f"{marker}_x"]
        df["y"] = df_with_mean[f"{marker}_y"]
        df["processing_level"] = "raw"
        tidy_df_list.append(df)
        print(df.head)
    return pd.concat(tidy_df_list).sort_values(['frame','keypoint'])

def tidy_toy_data(labels_path: Path, timestamps_path: Path) -> pd.DataFrame:
    machine_labels_df = pd.read_csv(labels_path)
    print(machine_labels_df.head(5))
    print(machine_labels_df.columns)

    machine_labels_df = machine_labels_df.sort_values("video")

    data_length = len(machine_labels_df)

    print(f"Loading timestamps from {timestamps_path}")
    timestamp_files = list(timestamps_path.glob("*.npy"))
    timestamp_files.sort(key = lambda path : path.stem)
    timestamp_arrays = [np.load(path) for path in timestamp_files]
    timestamps = np.concatenate(timestamp_arrays, axis=0)
    if timestamps.shape[0] != data_length:
        raise ValueError(f"dataframe size {data_length} does not match timestamp shape {timestamps.shape}")

    tidy_df_list = []
    for marker in ["toy_top", "toy_tail_base", "toy_nose"]:
        df = pd.DataFrame()
        df["frame"] = machine_labels_df["frame"]
        df["keypoint"] = np.array([marker] * data_length)
        df["video"] = machine_labels_df["video"]
        df["timestamp"] = timestamps / 1e9
        df["x"] = machine_labels_df[f"{marker}_x"]
        df["y"] = machine_labels_df[f"{marker}_y"]
        df["processing_level"] = "raw"
        tidy_df_list.append(df)
        print(df.head)
    return pd.concat(tidy_df_list).sort_values(['frame','keypoint'])




if __name__=='__main__':
    labels_path = Path("/home/scholl-lab/ferret_recordings/session_2025-10-20_ferret_420_E010/full_recording/mocap_data/dlc_output/toy_model_v2/skellyclicker_machine_labels_iteration_10.csv")
    timestamps_path = Path("/home/scholl-lab/ferret_recordings/session_2025-10-20_ferret_420_E010/full_recording/mocap_data/synchronized_corrected_videos")

    # data=tidy_head_data(labels_path=labels_path, timestamps_path=timestamps_path)
    # data.to_csv("/home/scholl-lab/ferret_recordings/session_2025-06-28_ferret_757_EyeCameras_P30_EO2/full_recording/mocap_data/tidy_head_data.csv", index=False)

    data=tidy_toy_data(labels_path=labels_path, timestamps_path=timestamps_path)
    data.to_csv("/home/scholl-lab/ferret_recordings/session_2025-10-20_ferret_420_E010/full_recording/mocap_data/toy_2d_tidy.csv", index=False)