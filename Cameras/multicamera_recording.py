from typing import List, Tuple
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

    def create_nir_camera_array(self) ->  pylon.InstantCameraArray:
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
                    
                    # do something with the image ....
                    
                    if min(frame_counts) >= number_of_frames:
                        print( f"all cameras have acquired {number_of_frames} frames")
                        break
                else:
                    print(f"grab unsuccessful from camera {result.GetCameraContext()}")
                    print(f"error description: {result.GetErrorDescription()}")

        self.nir_camera_array.StopGrabbing()

if __name__=="__main__":
    mcr = MultiCameraRecording()
    mcr.open_camera_array()

    mcr.grab_n_frames(15)

# # set the exposure time for each camera
# # for idx, cam in enumerate(cam_array):
# #     camera_serial = cam.DeviceInfo.GetSerialNumber()
# #     print(f"set Exposuretime {idx} for camera {camera_serial}")
# #     cam.ExposureTimeRaw = 10000


