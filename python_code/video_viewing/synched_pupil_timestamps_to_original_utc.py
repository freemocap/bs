import numpy as np
from pathlib import Path

if __name__=='__main__':
    synched_folder = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/basler_pupil_synchronized")
    pupil_folder = Path("/home/scholl-lab/recordings/session_2025-07-11/ferret_757_EyeCameras_P43_E15__1/pupil_output")

    for eye in ("eye0", "eye1"):
        synched_timestamps = np.load(str(synched_folder / f"cam_{eye}_synchronized_timestamps.npy"), allow_pickle=True)
        original_timestamps = np.load(str(pupil_folder / f"{eye}_timestamps_utc.npy"), allow_pickle=True)

        for original_frame_number in (4504, 11673):
            utc_time = synched_timestamps[original_frame_number]
            new_frame_number = np.where(original_timestamps == utc_time)[0][0]
            print(f"Video: {eye} old frame number: {original_frame_number}, new frame number: {new_frame_number}")
