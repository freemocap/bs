from typing import List, Tuple
import pypylon.pylon as pylon
import cv2
import numpy as np
import matplotlib as mpl


class MultiCameraRecording:
    def __init__(self):
        self.tlf = pylon.TlFactory.GetInstance()

        self.all_devices = self.get_devices()

    def get_devices(self) -> List[pylon.DeviceInfo]:
        return list(self.tlf.EnumerateDevices())
    
    def get_nir_devices(self) -> List[pylon.DeviceInfo]:
        return [device for device in self.all_devices if "NIR" in device.GetModelName()]
    
    def get_rgb_device(self) -> pylon.DeviceInfo:
        return [device for device in self.all_devices if "150uc" in device.GetModelName()][0]
    
    def get_nir_camera_array(self)

if __name__=="__main__":
    print(MultiCameraRecording().get_rgb_device().GetModelName())

# for device in devices:
#     print(f"Model name: {device.GetModelName()} Serial Number: {device.GetSerialNumber()}")

# nir_devices = [device for device in devices if "NIR" in device.GetModelName()]

# cam_array = pylon.InstantCameraArray(len(nir_devices))

# for index, cam in enumerate(cam_array):
#     cam.Attach(tlf.CreateDevice(nir_devices[index]))

# cam_array.Open()

# # store a unique number for each camera to identify the incoming images
# for index, cam in enumerate(cam_array):
#     camera_serial = cam.DeviceInfo.GetSerialNumber()
#     print(f"set context {index} for camera {camera_serial}")
#     cam.SetCameraContext(index)

# # set the exposure time for each camera
# # for idx, cam in enumerate(cam_array):
# #     camera_serial = cam.DeviceInfo.GetSerialNumber()
# #     print(f"set Exposuretime {idx} for camera {camera_serial}")
# #     cam.ExposureTimeRaw = 10000

# frames_to_grab = 10

# frame_counts = [0] * len(nir_devices)

# cam_array.StartGrabbing()
# while True:
#     with cam_array.RetrieveResult(1000) as result:
#         if result.GrabSucceeded():
#             image_number = result.ImageNumber
#             cam_id = result.GetCameraContext()
#             frame_counts[cam_id] = image_number
#             print(f"cam #{cam_id}  image #{image_number}")
            
#             # do something with the image ....
            
#             if min(frame_counts) >= frames_to_grab:
#                 print( f"all cameras have acquired {frames_to_grab} frames")
#                 break
                
                
# cam_array.StopGrabbing()

# print(frame_counts)


