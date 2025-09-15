import numpy as np
from pathlib import Path

if __name__=='__main__':
    session_folder = Path("/home/scholl-lab/ferret_recordings/session_2025-07-01_ferret_757_EyeCameras_P33_EO5")
    full_recording_folder = session_folder / "full_recording"
    eye_videos_folder = full_recording_folder / "eye_data" / "eye_videos"
    mocap_videos_folder = full_recording_folder / "mocap_data" / "synchronized_corrected_videos"
    # print(str(mocap_videos_folder))
    # print(mocap_videos_folder.exists())
    basler_timestamp_files = list(mocap_videos_folder.glob("*timestamps_utc.npy"))
    # print([file.name for file in basler_timestamp_files])
    basler_timestamp_arrays = [np.load(timestamp_path) for timestamp_path in basler_timestamp_files]

    starting_basler_frame = 7200
    ending_basler_frame = 12600

    basler_median_starting_timestamp = np.median([timestamps[starting_basler_frame] for timestamps in basler_timestamp_arrays])
    basler_median_ending_timestamp = np.median([timestamps[ending_basler_frame] for timestamps in basler_timestamp_arrays])

    print(f"Median basler time at frame {starting_basler_frame} is {int(basler_median_starting_timestamp)} ns")
    print(f"Median basler time at frame {ending_basler_frame} is {int(basler_median_ending_timestamp)} ns")

    print(f"Time difference between Basler start and end is {basler_median_ending_timestamp - basler_median_starting_timestamp} ns")
    for eye in ("eye0", "eye1"):
        pupil_timestamps = np.load(eye_videos_folder / f"{eye}_timestamps_utc.npy")
        print(f"first pupil timestamp from {eye}; {pupil_timestamps[0]}")
        print(f"last pupil timestamp from {eye}; {pupil_timestamps[-1]}")


        for basler_frame_number, basler_time in zip([starting_basler_frame, ending_basler_frame], [basler_median_starting_timestamp, basler_median_ending_timestamp]):
            estimated_match = np.searchsorted(pupil_timestamps, basler_time, side="right")
            if (basler_frame_number - pupil_timestamps[estimated_match-1]) < abs(basler_frame_number - pupil_timestamps[estimated_match]):
                closest_match = estimated_match-1
            else: 
                closest_match = estimated_match
            print(f"Video: {eye} basler frame number: {basler_frame_number}, eye video frame number: {closest_match}")
            print(f"\tTime difference between basler and pupil is {basler_time - pupil_timestamps[closest_match]} ns")
