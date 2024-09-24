from pathlib import Path
from typing import List, Tuple, Union
import pypylon.pylon as pylon
import cv2
import numpy as np
import matplotlib as mpl


class MultiCameraRecording:
    def __init__(self):
        self.tlf = pylon.TlFactory.GetInstance()

        self.all_devices = list(self.tlf.EnumerateDevices())
        self.nir_devices = [device for device in self.all_devices if "NIR" in device.GetModelName()]
        self.rgb_device = [device for device in self.all_devices if "150uc" in device.GetModelName()][0]

        self.nir_camera_array = self.create_nir_camera_array()

        self.video_writer_dict = None

    def create_nir_camera_array(self) -> pylon.InstantCameraArray:
        # It is possible to do this with devices directly -> instant camera is a utility for making devices easier to work with
        # InstantCameraArray isn't threadsafe -> if we add this to skellycam, each device needs its own InstantCamera
        nir_camera_array = pylon.InstantCameraArray(len(self.nir_devices))

        for index, cam in enumerate(nir_camera_array):
            cam.Attach(self.tlf.CreateDevice(self.nir_devices[index]))

        return nir_camera_array

    def open_camera_array(self):
        if not self.nir_camera_array.IsOpen():
            self.nir_camera_array.Open()

            for index, camera in enumerate(self.nir_camera_array):
                camera_serial = camera.DeviceInfo.GetSerialNumber()
                print(f"set context {index} for camera {camera_serial}")
                camera.SetCameraContext(index) # this gives us an easy to enumerate camera id, but we may prefer using serial number + dictionaries

    def close_camera_array(self):
        self.nir_camera_array.Close()

    def create_video_writers(self, output_folder: Union[str, Path] = Path(__file__).parent) -> dict:
        self.video_writer_dict = {}
        for index, camera in enumerate(self.nir_camera_array):
            file_name = f"{camera.DeviceInfo.GetSerialNumber()}.mp4"
            camera_fps = 15  # pull this property from device info
            frame_shape = (2048, 2048) # pull this property from device info if possible (may need to grab single frame and query that)

            writer = cv2.VideoWriter(
                str(Path(output_folder) / file_name),
                cv2.VideoWriter.fourcc(*"mp4v"),
                camera_fps,
                frame_shape
            )

            self.video_writer_dict[index] = writer

        return self.video_writer_dict
    
    def release_video_writers(self):
        if self.video_writer_dict:
            for video_writer in self.video_writer_dict:
                video_writer.release()

        self.video_writer_dict = None

    def write_frame(self, frame: np.ndarray, writer: cv2.VideoWriter, cam_id: int, frame_number: int):
        # check if writer is open
        # write to disk
        # check if it's open again -> if not, throw error (failed to write frame #...)

    def grab_n_frames(self, number_of_frames: int):
        frame_counts = [0] * len(self.nir_devices)
        self.nir_camera_array.StartGrabbing()
        while True:
            with self.nir_camera_array.RetrieveResult(1000) as result:
                if result.GrabSucceeded():
                    image_number = result.ImageNumber
                    cam_id = result.GetCameraContext()
                    frame_counts[cam_id] = image_number
                    print(f"cam #{cam_id}  image #{image_number} timestamp: {result.GetTimeStamp()}")
                    
                    if self.video_writer_dict and frame_counts[cam_id] <= number_of_frames:  # naive way of guaranteeing same length
                        self.video_writer_dict[cam_id].write(result.Array)
                    
                    if min(frame_counts) >= number_of_frames:
                        print(f"all cameras have acquired {number_of_frames} frames")
                        self.release_video_writers()
                        break
                else:
                    print(f"grab unsuccessful from camera {result.GetCameraContext()}")
                    print(f"error description: {result.GetErrorDescription()}")

        self.nir_camera_array.StopGrabbing()

if __name__=="__main__":
    mcr = MultiCameraRecording()
    mcr.open_camera_array()

    mcr.create_video_writers()
    mcr.grab_n_frames(15)

# # set the exposure time for each camera
# # for idx, cam in enumerate(cam_array):
# #     camera_serial = cam.DeviceInfo.GetSerialNumber()
# #     print(f"set Exposuretime {idx} for camera {camera_serial}")
# #     cam.ExposureTimeRaw = 10000


