import pandas as pd
import numpy as np
from pathlib import Path

from python_code.utilities.folder_utilities.recording_folder import RecordingFolder


def get_mean_dlc_confidence(recording_folder: RecordingFolder, save: bool = True):
    camera_names = [recording_folder.left_eye_name, recording_folder.right_eye_name]
    tidy_confidence_dfs = []
    for camera_name in camera_names:
        camera_confidence_df = get_one_camera_confidence(
            path_to_folder_with_dlc_csvs=recording_folder.eye_dlc_output,
            path_to_synchronized_video_folder=recording_folder.eye_videos,
            camera_name=camera_name
        )
        tidy_confidence_dfs.append(camera_confidence_df)

    session_confidence_df = pd.concat(tidy_confidence_dfs, ignore_index=True)

    if save:
        save_path = recording_folder.eye_data / f"{recording_folder.eye_dlc_output.stem}_mean_confidence.csv"
        session_confidence_df.to_csv(save_path, index=False)
        print(f"mean confidence values saved to {save_path}")

    return session_confidence_df

def get_one_camera_confidence(path_to_folder_with_dlc_csvs: Path, path_to_synchronized_video_folder: Path, camera_name: str):
    # Filtered csv list
    csv = list(path_to_folder_with_dlc_csvs.glob(f'{camera_name}*snapshot*.csv'))[0]
    timestamp_path = list(path_to_synchronized_video_folder.glob(f"{camera_name}*timestamps_utc*.npy"))[0]
    timestamps = np.load(timestamp_path) / 1e9

    # Read each csv into a dataframe with a multi-index header
    df = pd.read_csv(csv, header=[1, 2])
    
    # Drop the first column (which just has the headers )
    df = df.iloc[:, 1:]
    
    # Check if data shape is as expected
    if df.shape[1] % 3 != 0:
        print(f"Unexpected number of columns in {csv}: {df.shape[1]}")
        raise ValueError(f"wrong number of columns in csv {csv}")
    
    # Convert the df into a 4D numpy array of shape (num_frames, num_markers, 3) and append to dfs
    as_array = df.values.reshape(df.shape[0], df.shape[1]//3, 3)
    confidence_only_array = as_array[:, :, 2]

    new_df = pd.DataFrame()
    new_df["frames"] = range(0, confidence_only_array.shape[0])
    new_df["camera"] = [camera_name] * confidence_only_array.shape[0]
    new_df["mean_confidence"] = np.mean(confidence_only_array, axis=1)
    new_df["timestamps"] = timestamps

    
    return new_df


if __name__=="__main__":
    recording_folder = RecordingFolder.from_folder_path(
        Path("/home/scholl-lab/ferret_recordings/session_2025-10-20_ferret_420_E010/full_recording")
    )

    get_mean_dlc_confidence(recording_folder=recording_folder)