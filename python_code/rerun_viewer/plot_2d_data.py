from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.signal as signal
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint import VisualBounds2D
from rerun.datatypes import Range2D


def compile_dlc_csvs(path_to_folder_with_dlc_csvs:Path):
    # Filtered csv list
    csv_list = sorted(list(path_to_folder_with_dlc_csvs.glob('*snapshot*.csv')))

    # Initialize an empty list to hold dataframes
    dfs = []

    for csv in csv_list:
        # Read each csv into a dataframe with a multi-index header
        df = pd.read_csv(csv, header=[1, 2])
        
        # Drop the first column (which just has the headers )
        df = df.iloc[:, 1:]
        
        # Check if data shape is as expected
        if df.shape[1] % 3 != 0:
            print(f"Unexpected number of columns in {csv}: {df.shape[1]}")
            continue
        
        try:
            # Convert the df into a 4D numpy array of shape (1, num_frames, num_markers, 3) and append to dfs
            dfs.append(df.values.reshape(1, df.shape[0], df.shape[1]//3, 3))
        except ValueError as e:
            print(f"Reshape failed for {csv} with shape {df.shape}: {e}")


    # Concatenate all the arrays along the first axis (camera axis)
    dlc_2d_array_with_confidence = np.concatenate(dfs, axis=0)

    return dlc_2d_array_with_confidence

def apply_confidence_threshold(array, threshold):
    """
    Set X,Y values to NaN where the corresponding confidence value is below threshold.
    """
    mask = array[..., 2] < threshold  # Shape: (num_cams, num_frames, num_markers)
    array[mask, 0] = np.nan  # Set X to NaN where confidence is low
    array[mask, 1] = np.nan  # Set Y to NaN where confidence is low
    return array

def butter_lowpass_filter(data, cutoff, sampling_rate, order):
    """ Run a low pass butterworth filter on a single column of data"""
    nyquist_freq = 0.5*sampling_rate
    normal_cutoff = cutoff / nyquist_freq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butterworth_filter_2d_data(cams_frames_markers_xy: np.ndarray) -> np.ndarray:
    print("Applying butterworth filter to 2d data")
    output_array = np.zeros_like(cams_frames_markers_xy)
    for camera in range(cams_frames_markers_xy.shape[0]):
        for marker in range(cams_frames_markers_xy.shape[2]):
            for dim in range(cams_frames_markers_xy.shape[3]):
                output_array[camera, :, marker, dim] = butter_lowpass_filter(
                    cams_frames_markers_xy[camera, :, marker, dim],
                    cutoff=4,
                    sampling_rate=90,
                    order=4
                )
    return output_array

def interpolate_2d_data(cam_frame_marker_xy: np.ndarray, method_to_use = 'linear', order = 3) -> np.ndarray:
    """ Takes in a 3d skeleton numpy array from freemocap and interpolates missing NaN values"""
    print(f"number of NaNs to be interpolated: {np.count_nonzero(np.isnan(cam_frame_marker_xy))} out of {np.count_nonzero(cam_frame_marker_xy)} points")
    interpolated_data = np.zeros_like(cam_frame_marker_xy)
    for cam in range(cam_frame_marker_xy.shape[0]):
        skeleton_data=cam_frame_marker_xy[cam, :, :, :]
        num_frames = skeleton_data.shape[0]
        num_markers = skeleton_data.shape[1]

        for marker in range(num_markers):
            this_marker_skel3d_data = skeleton_data[:,marker,:]
            df = pd.DataFrame(this_marker_skel3d_data)
            df2 = df.interpolate(method = method_to_use,axis = 0, order = order) #use pandas interpolation methods to fill in missing data
            # df.interpolate(method=method_to_use, order = 5)
            this_marker_interpolated_skel3d_array = np.array(df2)
            #replace the remaining NaN values (the ones that often happen at the start of the recording)
            this_marker_interpolated_skel3d_array = np.where(np.isfinite(this_marker_interpolated_skel3d_array), this_marker_interpolated_skel3d_array, np.nanmean(this_marker_interpolated_skel3d_array))
            
            interpolated_data[cam,:,marker,:] = this_marker_interpolated_skel3d_array

    return interpolated_data

def plot_2d_x_and_y(camera: int, marker:int, data: np.ndarray, time: list | np.ndarray, entity_path: str):
    x_color = [255, 0, 0]
    y_color = [0, 0, 255]
    rr.log(f"{entity_path}/x",
        rr.SeriesPoints(colors=x_color,
                        markers="circle",
                        marker_sizes=2),
        static=True)
    rr.log(f"{entity_path}/y",
        rr.SeriesPoints(colors=y_color,
                        markers="circle",
                        marker_sizes=2),
        static=True)    
    rr.send_columns(
        entity_path=f"{entity_path}/x",
        indexes=[rr.TimeColumn("time", duration=time)],
        columns=rr.Scalars.columns(scalars=data[camera, :, marker, 0]),
    )
    rr.send_columns(
        entity_path=f"{entity_path}/y",
        indexes=[rr.TimeColumn("time", duration=time)],
        columns=rr.Scalars.columns(scalars=data[camera, :, marker, 1]),
    )

def plot_2d(camera: int, marker: int, csv_path: str):
    csv_path = Path(csv_path)

    unprocessed_2d_data = compile_dlc_csvs(csv_path)
    data_length = unprocessed_2d_data.shape[1]

    thresholded_data = apply_confidence_threshold(unprocessed_2d_data, 0.8)[:,:,:,:2]

    recording_string = (
        f"{csv_path.stem}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    rr.init(recording_string, spawn=True)

    plot_2d_x_and_y(
        camera=camera, 
        marker=marker, 
        data=unprocessed_2d_data, 
        time=range(0, data_length), 
        entity_path="/unprocessed_data"
    )
    plot_2d_x_and_y(
        camera=camera, 
        marker=marker, 
        data=thresholded_data, 
        time=range(0, data_length), 
        entity_path="/with_confidence_threshold"
    )
if __name__=="__main__":
    csv_path = "/home/scholl-lab/ferret_recordings/session_2025-07-11_ferret_757_EyeCamera_P43_E15__1/clips/0m_37s-1m_37s/mocap_data/dlc_output/head_body_eyecam_retrain_test_v2_model_outputs_iteration_1"
    plot_2d(csv_path=csv_path, camera=0, marker=0)