# Cameras

Resolution:
- Minimum 1080p, which is 2 MP
- 2k is 5 mp
- 4k is 8mp

## IR Cameras (4)
- **Basler Ace Classic** has "NIR Enhanced"
    - [Product Link](https://www.baslerweb.com/en-us/shop/aca2040-180kmnir/)
    - 4 MP
    - 180 FPS
    - Aspect Ratio: 2048 px × 2048 px
    - Visible + SWIR
    - Camera Link
    - Global Shutter
    - Sensor format: 1"
    - Power: PoCL or 12 VDC
    - Synch: Hardware or Software Synch
    - Lens Mount: C
    - Additional Accessories: Lens, Camera Link Cable, Power Supply (and PC Aquisition Card?)
    - $2,939.00 for camera and sensor (Basler also has [90 FPS version](https://www.baslerweb.com/en-us/shop/aca2040-90umnir/) for $2,119.00)
- **Flir Grasshopper3** has dedicated NIR
    - [Product Link](https://www.flir.com/products/grasshopper3-usb3/?model=GS3-U3-41C6NIR-C&vertical=machine%20vision&segment=iis)
    - 4.1 MP
    - 90 FPS
    - Aspect Ratio: 2048 px × 2048 px
    - Dedicated NIR
    - USB 3.1 interface
    - Globar Shutter
    - Sensor format: 1"
    - Power: External 5V GPIO recommended, but USB is possible
    - Lens Mount: C
    - Additional Accessories: Lens, Locking USB cable, Host Adapter
    - $1,539.00 for camera and sensor

## RGB Cameras (1):
- **Basler Ace** Color, PYTHON 2000 Sensor
    - [Product Link](https://www.baslerweb.com/en-us/shop/aca1920-python-2000/?interface=Y29uZmlndXJhYmxlLzE1MTIvMjE2NTM=&spectrum=Visible&housing_lensmount=C-mount&sort=position&camera_product_line=Y29uZmlndXJhYmxlLzEzMjYvMjA4NzA=&sensor_monocolor=Y29uZmlndXJhYmxlLzE2NDEvMjIyNjM=)
    - 2.3 MP
    - 150 FPS
    - Aspect Ratio: 1920 px x 1200 px
    - Color
    - USB 3. interface
    - Global Shutter
    - Sensor Format: 2/3” 
    - Power: USB 3.0
    - Lens Mount: C
    - Additional Accessories: Lens, Cable (Data Acquisition?)
    - $699.00 for camera and sensor
- **Flir Blackfly S** color
    - [Product Link](https://www.flir.com/products/blackfly-s-usb3/?model=BFS-U3-27S5C-C&vertical=machine%20vision&segment=iis)
    - 2.8 MP
    - 95 FPS
    - Aspect Ratio: 1936 x 1464
    - Color
    - USB 3.1 interface
    - Global Shutter
    - Sensor Format: 2/3” 
    - Power: USB 3.1
    - Lens Mount: C
    - Additional Accessories: Lens, Cable, Host Adapter
    - $715.00 for camera and sensor


## Lenses:
Lens focal length views estimated using Blender mockup of rig. To test in Blender:
1. Set camera "sensor fit" to horizontal, and set width based on sensor size in mm
2. On output tab, set camera resolution in X and Y fields
3. Change focal length until view matches

Flir [lens calculator](https://www.flir.com/iis/machine-vision/lens-calculator/) gives different results than blender FOV

### Side Cameras
For side cameras (square aspect ratio), 7 mm gives full coverage of the space, stretching corner to corner horizontally. 10 mm is not quite corner to corner but covers most of the space. 12mm gives a tighter fit, losing the corners but covering most of the center
- Basler [Kowa Lens LM8HC F1.4 f8mm 1"](https://www.baslerweb.com/en-us/shop/kowa-lens-lm8hc-f1-4-f8mm-1/)
    - 8mm focal length
    - F1.4 - F16
    - C Mount
    - 1" Sensor Compatible
    - $559.00
- Flir [Tamron 8mm 1/1.8inch C mount Lens](https://www.flir.com/products/tamron-8mm-11.8inch-c-mount-lens/?vertical=machine%20vision&segment=iis)
    - 8mm focal length
    - F1.4 - ??
    - C Mount
    - 1/1.8" sensor compatible
    - $263.00

### Top Cameras
For top camera (roughly 1920 x 1200), 5mm gives full coverage top to bottom, with extra space on sides. 6mm is very close to full. Flir calculator says 7mm should have full coverage though
- Basler [Ricoh Lens FL-CC0614A-2M F1.4 f6mm 2/3"](https://www.baslerweb.com/en-us/shop/ricoh-lens-fl-cc0614a-2m-f1-4-f6mm-2-3/)
    - 6mm focal length
    - 2MP Resolution
    - F1.4 - F16
    - C Mount
    - 2/3" Sensor Compatible
    - $249.00
- Flir [Tamron 8mm 1/1.8inch C mount Lens](https://www.flir.com/products/tamron-8mm-11.8inch-c-mount-lens/?vertical=machine%20vision&segment=iis)
    - 8mm focal length
    - F1.4 - ??
    - C Mount
    - 1/1.8" sensor compatible
    - $263.00


## Data Aquisition:
- [Basler microEnable 5 marathon ACL](https://www.baslerweb.com/en-us/shop/microenable-5-marathon-acl/?interface=Camera+Link&sort=position)
    - 2 Ports, need two total (if using USB RGB camera)
    - Can power camera
    - PCIe connection to computer
    - $1,049.00 ea
- [Flir Host Adapter](https://www.flir.com/products/usb-3.1-host-controller-card/?model=ACC-01-1203&vertical=machine+vision&segment=iis)
    - 4 ports, need two total
    - Difference between Host controller and host adapter??
    - PCIe connection to computer
    - $195.00 ea

## Mounts/Attachments
Will need some form of 8020 tripod mounts
- Basler [Tripod Mount Ace](https://www.baslerweb.com/en-us/shop/tripod-mount-ace/)
    - $11.53
- Flir Blackfly S [Tripod Adapter for 39 mm Blackfly S Models](https://www.flir.com/products/tripod-adaptor-for-39mm-blackfly-s-models/?vertical=machine%20vision&segment=iis)
    - $37.50
- Flir Grasshopper [Tripod Adapter for GRAS, GS2, GS3, GX](https://www.flir.com/products/tripod-adapter-for-gras-gs2-gs3-gx/?vertical=machine%20vision&segment=iis)
    - $11.80


## Triggers
Basler accepts software triggers, not sure about Flir
We want hardware triggers, see if this can be done over camera link

## Cables:
- Basler [Camera Link PoCL](https://www.baslerweb.com/en-us/shop/basler-cable-camera-link-pocl-sdr-mdr-p-3-m/)
    - Comes in SDR/SDR and SDR/MDR, Camera Link interface needs MDR
    - 3, 5, 10m
    - $149 - $420 ea.
- Basler [Basler Cable USB 3.0, Micro B 90° A1 sl/A (ace downwards), P, 3 m](https://www.baslerweb.com/en-us/shop/basler-cable-usb-3-0-micro-b-90-a1-sl-a-ace-downwards-p-3-m/)
    - 3m 
    - $45.00
- Flir [Locking USB 3.1](https://www.flir.com/products/usb-3.1-locking-cable/?vertical=machine%20vision&segment=iis)
    - 3 or 5 m
    - $24.50 - 37.50 ea.


## Lighting
Opting for flashing lights for longer service life, unless that's less desireable?
Will IR light work well for NIR? Basler doesn't list any SWIR lights. 
We might not need so much light?
- 3x Basler Standard Light Bar-45x100_PowerIR
    - [Product Link](https://www.baslerweb.com/en-us/shop/basler-standard-light-bar-45x100-powerir/)
    - $709
- 1x Basler Standard Light Bar-45x100-White
    - [Product Link](https://www.baslerweb.com/en-us/shop/basler-standard-light-bar-45x100-white/)
    - $449.00
- Basler Light Controller 4C-1.25A-84W-24V
    - [Product Link](https://www.baslerweb.com/en-us/shop/basler-light-controller-4c-1-25a-84w-24v/)
    - Up to 4 lights
    - 12V or 24V
    - $323
- 1x AC Power Cord USA NEMA 5-15, 1.8m
    - [Poduct Link](https://www.baslerweb.com/en-us/shop/ac-power-cord-usa-nema-5-15-1-8m/)
    - $14
- 4x BLC-BSL connect cable M8, 2m
    - [Product Link](https://www.baslerweb.com/en-us/shop/blc-bsl-connect-cable-m8-2m/)
    - $41
Do we want active IR lighting?
- yes
We want diffuser for visual light
don't worry too much about specifics, we can buy more if needed
Bar lights are fine


## Total Costs:
### Basler:
- 4x IR cameras:
- 1x RGB camera:
- 5x Lens:
- 5x Cable:
- 5x Mount:

**Basler system Total:**

### Flir:
- 4x IR cameras:
- 1x RGB camera:
- 5x Lens:
- 5x Cable:
- 3x Host Adapter (?):
- 5x Mount:

**Flir System Total**:


## Compatability with OpenCV:
### Basler
 - Basler uses Pylon, and has an OpenCV [integration guide](https://www2.baslerweb.com/media/downloads/documents/application_notes/AW00136803000_Getting_Started_with_pylon_6_and_OpenCV.pdf)
 - [PyPylon](https://www.baslerweb.com/en-us/software/pylon/pypylon/) has Basler supported Python integration
 - Has code samples for [multicamera capture](https://github.com/basler/pypylon-samples/blob/main/notebooks/multicamera_handling.ipynb)

### Flir
- Flir uses Spinnaker, and also has an OpenCV [integration guide](https://www.flir.com/support-center/iis/machine-vision/application-note/getting-started-with-opencv/)
- Could not find a Flir-supported python integration. There is an [actively maintained python integration](https://github.com/LJMUAstroecology/flirpy), but don't list the blackfly as supported.
