from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Callable, Dict, List, Tuple, Union
import pypylon.pylon as pylon
import cv2
import numpy as np
import matplotlib as mpl
import logging

from diagnostics.timestamp_mapping import TimestampMapping

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(f"/home/scholl-lab/recordings/basler_logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
console_handler = logging.StreamHandler()

file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class MultiCameraRecording:
    def __init__(self, output_path: Path = Path(__file__).parent, fps: float = 30):
        self.tlf = pylon.TlFactory.GetInstance()

        self.all_devices = list(self.tlf.EnumerateDevices())
        self.nir_devices = [device for device in self.all_devices if "NIR" in device.GetModelName()]
        self.rgb_device = [device for device in self.all_devices if "150uc" in device.GetModelName()][0]

        # self.devices = self.all_devices
        self.devices = self.nir_devices
        self.camera_array = self.create_camera_array()

        self.video_writer_dict = None
        self.fps = fps
        self.image_height = 2048
        self.image_width = 2048

        self.rgb_converter = pylon.ImageFormatConverter()  # TODO: do this setup elsewhere
        self.nir_converter = pylon.ImageFormatConverter() 

        # converting to opencv bgr format
        self.rgb_converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.rgb_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.nir_converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.nir_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.validate_output_path(output_path=output_path)

    def validate_output_path(self, output_path: Path):
        output_path = Path(output_path)

        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"Output path {output_path} must be a directory, not a file")

        # if path exists, isn't empty, and isn't the default, make a new directory to avoid overwriting videos
        while output_path != Path(__file__).parent and output_path.exists() and next(output_path.iterdir(), None):
            stem = output_path.stem
            if len(split := stem.split("__")) > 1:
                split[-1] = "__" + str(int(split[-1]) + 1)
            else:
                split.append("__1")
            new_name = ''.join(split)
            output_path = output_path.parent / new_name

        logger.info(f"Will save videos to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path

    def create_camera_array(self) -> pylon.InstantCameraArray:
        # It is possible to do this with devices directly -> instant camera is a utility for making devices easier to work with
        # InstantCameraArray isn't threadsafe -> if we add this to skellycam, each device needs its own InstantCamera
        camera_array = pylon.InstantCameraArray(len(self.devices))

        for index, cam in enumerate(camera_array):
            cam.Attach(self.tlf.CreateDevice(self.devices[index]))

        return camera_array

    def open_camera_array(self):
        if not self.camera_array.IsOpen():
            self.camera_array.Open()

            for index, camera in enumerate(self.camera_array):
                camera_serial = camera.DeviceInfo.GetSerialNumber()
                logger.info(f"set context {index} for camera {camera_serial}")
                camera.SetCameraContext(index) # this gives us an easy to enumerate camera id, but we may prefer using serial number + dictionaries

    def close_camera_array(self):
        self.camera_array.Close()

    def camera_information(self):
        """See list of options for this here: https://docs.baslerweb.com/pylonapi/net/T_Basler_Pylon_PLCamera"""
        for cam in self.camera_array:
            logger.info(f"Camera information for camera {cam.GetCameraContext()}")
            logger.info(f"\tMax number of buffers: {cam.MaxNumBuffer.Value}")
            logger.info(f"\tMax buffer size: {cam.StreamGrabber.MaxBufferSize.Value}")
            logger.info(f"\tExposure time: {cam.ExposureTime.Value}")
            logger.info(f"\tFrame rate: {cam.AcquisitionFrameRate.Value}")
            # logger.info(f"\tShutter mode: {cam.ShutterMode.Value}")
            logger.info(f"\tGain: {cam.Gain.Value}")

    def set_max_num_buffer(self, num: int):
        "The maximum number of buffers that are allocated and used for grabbing."
        for cam in self.camera_array:
            cam.MaxNumBuffer.Value = num

    def set_fps(self, fps: float):
        self.fps = fps

        if self.video_writer_dict:  # Video writers need to match fps
            self.release_video_writers()
            self.create_video_writers()

    def set_exposure_time(self, camera, exposure_time: int):
        camera.ExposureTime.Value = exposure_time
        logger.info(f"Set exposure time for camera {camera.GetCameraContext()} to {exposure_time} {camera.ExposureTime.Unit}")

    def set_gain(self, camera, gain: float):
        camera.Gain.Value = gain
        logger.info(f"Set gain for camera {camera.GetCameraContext()} to {gain}")
    
    def set_image_resolution(self, binning_factor: int):
        if binning_factor not in (1, 2, 3, 4):
            raise RuntimeError(f"Valid binning factors are 1, 2, 3, 4 - you provided {binning_factor}")
        
        self.image_width = int(self.image_width / binning_factor)
        self.image_height = int(self.image_height / binning_factor)
        for cam in self.camera_array:
            cam.BinningHorizontal.Value = binning_factor
            cam.BinningVertical.Value = binning_factor

        if self.video_writer_dict:  # Video writers need to match image height and width
            self.release_video_writers()
            self.create_video_writers()

    def _set_fps_during_grabbing(self):
        for cam in self.camera_array:
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.SetValue(self.fps)
            logger.info(f"Cam {cam.GetCameraContext()} FPS set to {cam.AcquisitionFrameRate.Value}")

    def pylon_internal_statistics(self):
        successful_recording = True
        for cam in self.camera_array:
            logger.info(f"pylon internal statistics for camera {cam.GetCameraContext()}")

            logger.info(f"total buffer count: {cam.StreamGrabber.Statistic_Total_Buffer_Count.GetValue()}")
            logger.info(f"failed buffer count: {cam.StreamGrabber.Statistic_Failed_Buffer_Count.GetValue()}")
            if cam.StreamGrabber.Statistic_Failed_Buffer_Count.GetValue() > 0:
                successful_recording = False
            # logger.info(f"buffer underrun count: {cam.StreamGrabber.Statistic_Buffer_Underrun_Count.GetValue()}") # these below are all allegedly supported but throw errors
            # logger.info(f"total packet count: {cam.StreamGrabber.Statistic_Total_Packet_Count.GetValue()}")
            # logger.info(f"failed packet count: {cam.StreamGrabber.Statistic_Failed_Packet_Count.GetValue()}")
            # logger.info(f"resend request count: {cam.StreamGrabber.Statistic_Resend_Request_Count.GetValue()}")
            # logger.info(f"resend packet count: {cam.StreamGrabber.Statistic_Resend_packet_Count.GetValue()}")

        if successful_recording == False:
            logger.error("FRAMES WERE DROPPED \nYou may need to lower the framerate, reduce the frame size, or increase max number of buffers")
        else:
            logger.info("No frames dropped, recording was successful")
            

    def create_video_writers(self, output_folder: Union[str, Path, None] = None) -> dict:
        if output_folder is None:
            output_folder = self.output_path
        self.video_writer_dict = {}
        for index, camera in enumerate(self.camera_array):
            file_name = f"{camera.DeviceInfo.GetSerialNumber()}.mp4"
            camera_fps = self.fps
            frame_shape = (self.image_width, self.image_height)

            writer = cv2.VideoWriter(
                str(Path(output_folder) / file_name),
                cv2.VideoWriter.fourcc(*'mp4v'),
                camera_fps,
                frame_shape # width, height
            )

            self.video_writer_dict[index] = writer

        return self.video_writer_dict
    
    def release_video_writers(self):
        if self.video_writer_dict:
            for video_writer in self.video_writer_dict.values():
                video_writer.release()

        self.video_writer_dict = None
        logger.info("Video writers released")

    def write_frame(self, frame: np.ndarray, cam_id: int, frame_number: int):
        if self.video_writer_dict is None:
            raise RuntimeError("Attempted to write frame before video writers were created")

        writer = self.video_writer_dict[cam_id]
        if not writer.isOpened():
            raise RuntimeWarning(f"Attmpted to write frame to unopened video writer: cam {cam_id}")
        self.video_writer_dict[cam_id].write(frame)  # Check if pylon's ImageFormatConverter is faster
        if not writer.isOpened():
            raise RuntimeWarning(f"Failed to write frame #{frame_number}")
        
    def get_timestamp_mapping(self) -> TimestampMapping:
        """
        Timestamps are given in ns since camera was powered on. Latching the timestamp gets the current value, which can then be pulled from the latch value.
        This allows us to calculate the timestamp values from (roughly) the same moment in time.
        Synchronization will be off by the lag between first camera latch and last camera latch.
        TODO: find if there is a method for latching an entire camera array at once
        See https://docs.baslerweb.com/timestamp for more info.
        """
        [camera.TimestampLatch.Execute() for camera in self.camera_array]

        # TODO: there's a slight inaccuracy in timing here, as the timestamp mapping is created after the dictionary construction
        starting_timestamps = {camera.GetCameraContext(): camera.TimestampLatchValue.Value for camera in self.camera_array}
        timestamp_mapping = TimestampMapping(camera_timestamps=starting_timestamps)

        return timestamp_mapping

    def _grab_frames(self, condition: Callable, number_of_frames: int):
        frame_counts = [0] * len(self.devices)
        timestamps = np.zeros((len(self.devices), number_of_frames)) # how to handle this if we don't know number of frames in advance?
        starting_timestamps = self.get_timestamp_mapping()
        self.camera_array.StartGrabbing()
        self._set_fps_during_grabbing()
        logger.info("Starting recording...")
        while True:
            with self.camera_array.RetrieveResult(1000) as result:
                if result.GrabSucceeded():
                    image_number = result.ImageNumber 
                    cam_id = result.GetCameraContext()
                    frame_counts[cam_id] = image_number
                    timestamp = result.GetTimeStamp() - starting_timestamps.camera_timestamps[cam_id]
                    logger.debug(f"cam #{cam_id}  image #{image_number} timestamp: {timestamp}")
                    try:
                        timestamps[cam_id, image_number-1] = timestamp
                        if cam_id == 4:
                            frame = self.rgb_converter.Convert(result)
                            image = frame.Array
                        else:
                            frame = result.Array
                            image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        self.write_frame(frame=image, cam_id=cam_id, frame_number=frame_counts[cam_id])
                    except IndexError:
                        # TODO: dynamically resize timestamps array
                        pass  
                        
                    if condition(frame_counts):
                        break
                else:
                    logger.error(f"grab unsuccessful from camera {result.GetCameraContext()}")
                    logger.error(f"error description: {result.GetErrorDescription()}")
                    logger.error(f"failure timestamp: {result.GetTimeStamp()}")

        # TODO: would we get better failure handling if this were in a finally clause on the while loop?
        self.release_video_writers()
        self.camera_array.StopGrabbing()
        final_timestamps = self.get_timestamp_mapping()
        self.save_timestamps(timestamps=timestamps, starting_mapping=starting_timestamps, ending_mapping=final_timestamps)
        logger.info(f"frame counts: {frame_counts}")
        mcr.pylon_internal_statistics()

    def grab_n_frames(self, number_of_frames: int):
        self._grab_frames(condition=lambda frame_counts: min(frame_counts) >= number_of_frames, number_of_frames=number_of_frames)

    def grab_n_seconds(self, number_of_seconds: float):
        number_of_frames = int(number_of_seconds * self.fps)
        self._grab_frames(condition=lambda frame_counts: min(frame_counts) >= number_of_frames, number_of_frames=number_of_frames)

    def grab_until_input(self):
        stop_event = threading.Event()

        def stop_on_input(stop_event):
            input("Press Enter to stop...\n")
            stop_event.set()
        input_thread = threading.Thread(target=stop_on_input, args=(stop_event,))
        input_thread.start()

        self._grab_frames(condition=lambda frame_counts: stop_event.is_set(), number_of_frames=1000000)  # initialize with 1 million frames, or about 1 hour of input at 60 fps with 4 cameras

        input_thread.join()

    def trim_timestamp_zeros(self, timestamps: np.ndarray):
        nonzero = np.nonzero(timestamps)
        return timestamps[:, :nonzero[1].max() + 1]

    def save_timestamps(self, timestamps: np.ndarray, starting_mapping: TimestampMapping, ending_mapping: TimestampMapping):
        timestamps = self.trim_timestamp_zeros(timestamps)
        np.save(self.output_path / "timestamps.npy", timestamps)

        with open(self.output_path / "timestamp_mapping.json", mode="x") as f:
            combined_data = {
                "starting_mapping": starting_mapping.model_dump(),
                "ending_mapping": ending_mapping.model_dump()
            }
            json.dump(combined_data, f, indent=4)

    def grab_until_failure(self):
        self.camera_array.StartGrabbing()
        frame_list = []
        while True:
            with self.camera_array.RetrieveResult(1000) as result:
                if result.GrabSucceeded():
                    image_number = result.ImageNumber
                    cam_id = result.GetCameraContext()
                    logger.debug(f"cam #{cam_id}  image #{image_number} timestamp: {result.GetTimeStamp()}")
                    frame_list.append(result.Array)
                    
                else:
                    logger.error(f"grab unsuccessful from camera {result.GetCameraContext()}")
                    logger.error(f"error description: {result.GetErrorDescription()}")
                    break

        self.camera_array.StopGrabbing()
        logger.info(f"grabbed {len(frame_list)} frames before failure")

def make_session_folder_at_base_path(base_path: Path) -> Path:
    now = datetime.now()
    output_path_name = f"session_{now.year}-{now.month}-{now.day}"

    output_path = base_path / output_path_name

    output_path.mkdir(parents=True, exist_ok=True)

    return output_path

if __name__=="__main__":
    base_path = Path("/home/scholl-lab/recordings")  

    #TURN ON THE IR LIGHTS!
    recording_name = "ferret_0776_P44_E14" #P: postnatal day (age), EO: eyes open day (how long)
    #recording_name = "calibration_moving_charuco" #P: postnatal day (age), EO: eyes open day (how long)

    output_path = make_session_folder_at_base_path(base_path=base_path) / recording_name

    mcr = MultiCameraRecording(output_path=output_path)
    mcr.open_camera_array()
    mcr.set_max_num_buffer(60)
    mcr.set_fps(60)
    mcr.set_image_resolution(binning_factor=2)
    for index, camera in enumerate(mcr.camera_array):
        match mcr.devices[index].GetSerialNumber():
            case "24908831":
                mcr.set_exposure_time(camera, exposure_time=7000)
                mcr.set_gain(camera, gain=0)
            case "24908832": 
                mcr.set_exposure_time(camera, exposure_time=6500)
                mcr.set_gain(camera, gain=0)
            case "25000609": 
                mcr.set_exposure_time(camera, exposure_time=7000)
                mcr.set_gain(camera, gain=0)
            case "25006505": 
                mcr.set_exposure_time(camera, exposure_time=7000)
                mcr.set_gain(camera, gain=0)
            case "40520488":
                mcr.set_exposure_time(camera, exposure_time=7000)
                mcr.set_gain(camera, gain=0)
            case _: 
                raise ValueError("Serial number does not match given values")
        
    mcr.camera_information()

    mcr.create_video_writers()
    # mcr.grab_n_frames(120)  # Divide frames by fps to get time
    # mcr.grab_n_seconds(20*60)
    mcr.grab_until_input()  # press enter to stop recording, will run until enter is pressed


    mcr.close_camera_array()
    # TODO: this segfaults on close every time

    # mcr.grab_until_failure()


