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
            camera_fps = 15.0  # pull this property from device info
            frame_shape = (2048, 2048) # pull this property from device info if possible (may need to grab single frame and query that)

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
        print("Video writers released")

    def write_frame(self, frame: np.ndarray, cam_id: int, frame_number: int):
        writer = self.video_writer_dict[cam_id]
        if not writer.isOpened():
            raise RuntimeWarning(f"Attmpted to write frame to unopened video writer: cam {cam_id}")
        print(f"frame shape is {frame.shape}")
        self.video_writer_dict[cam_id].write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        if not writer.isOpened():
            raise RuntimeWarning(f"Failed to write frame #{frame_number}")

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
                        print(f"writing frame {frame_counts[cam_id]} from camera {cam_id} with width {result.Width} and height {result.Height}")
                        self.write_frame(frame=result.Array, cam_id=cam_id, frame_number=frame_counts[cam_id])
                    
                    if min(frame_counts) >= number_of_frames:
                        print(f"all cameras have acquired {number_of_frames} frames")
                        self.release_video_writers()
                        break
                else:
                    print(f"grab unsuccessful from camera {result.GetCameraContext()}")
                    print(f"error description: {result.GetErrorDescription()}")
                    print(f"failure timestamp: {result.GetTimeStamp()}")

        self.nir_camera_array.StopGrabbing()

    def grab_until_failure(self):
        self.nir_camera_array.StartGrabbing()
        frame_list = []
        while True:
            with self.nir_camera_array.RetrieveResult(1000) as result:
                if result.GrabSucceeded():
                    image_number = result.ImageNumber
                    cam_id = result.GetCameraContext()
                    print(f"cam #{cam_id}  image #{image_number} timestamp: {result.GetTimeStamp()}")
                    frame_list.append(result.Array)
                    
                else:
                    print(f"grab unsuccessful from camera {result.GetCameraContext()}")
                    print(f"error description: {result.GetErrorDescription()}")
                    break

        self.nir_camera_array.StopGrabbing()
        print(f"grabbed {len(frame_list)} frames before failure")


if __name__=="__main__":
    mcr = MultiCameraRecording()
    mcr.open_camera_array()

    mcr.create_video_writers()
    mcr.grab_n_frames(15)

    mcr.close_camera_array()

    # mcr.grab_until_failure()

# # set the exposure time for each camera
# # for idx, cam in enumerate(cam_array):
# #     camera_serial = cam.DeviceInfo.GetSerialNumber()
# #     print(f"set Exposuretime {idx} for camera {camera_serial}")
# #     cam.ExposureTimeRaw = 10000


