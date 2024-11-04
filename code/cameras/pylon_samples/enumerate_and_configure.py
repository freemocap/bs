import pypylon.pylon as pylon
import cv2
import numpy as np
import matplotlib as mpl

tlf = pylon.TlFactory.GetInstance()

devices = tlf.EnumerateDevices()

for device in devices:
    print(f"Model name: {device.GetModelName()} Serial Number: {device.GetSerialNumber()}")

nir_devices = [device for device in devices if "NIR" in device.GetModelName()]

print("NIR devices:")
for device in nir_devices:
    print(f"Model name: {device.GetModelName()} Serial Number: {device.GetSerialNumber()}")

    cam = pylon.InstantCamera(tlf.CreateDevice(device))
    
    cam.Open()

    result = cam.GrabOne(1000)
    image = result.Array

    # cv2.imshow(device.GetSerialNumber(), image)
    # cv2.waitKey(0)

    cam.Close()