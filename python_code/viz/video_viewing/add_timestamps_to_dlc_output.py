import pandas as pd
import numpy as np
from pathlib import Path

if __name__=="__main__":
    dlc_path = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/clips/0_37-1_37/mocap_data/output_data/dlc_data/25006505_clipped_3377_8754DLC_Resnet50_headmount_and_spine_shuffle1_snapshot_080.csv")
    timestamp_path = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/clips/0_37-1_37/mocap_data/synchronized_videos/25006505_clipped_3377_8754.npy")

    # dlc_path = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/clips/0_37-1_37/eye_data/output_data/dlc_output/eye0_clipped_4451_11621DLC_Resnet50_pupil_tracking_ferret_757_EyeCameras_P43_E15__1_shuffle1_snapshot_030.csv")
    # timestamp_path = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/clips/0_37-1_37/eye_data/synchronized_videos/eye0_clipped_4451_11621.npy")

    timestamps=np.load(timestamp_path)

    dlc_data = pd.read_csv(dlc_path, header=[1,2], skiprows=0)

    if dlc_data.shape[0] != timestamps.shape[0]:
        raise ValueError(f"shapes of dlc data {dlc_data.shape[0]} and timestamps {timestamps.shape} do not match")
    
    dlc_data["timestamps_utc"] = timestamps

    dlc_data.to_csv(dlc_path)