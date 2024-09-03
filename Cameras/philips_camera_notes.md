# Philip's Notes on Camera Setup 

## Physical Setup Notes
1. There is some glare from the charuco in 2 camera views (cameras ...832 and ...609). Is there something we can put behind the plexiglass to prevent this? It may mess up calibration and tracking.

2. Camera positioning should match the blender file, all be upright, etc.

## Pylon Notes

1. You can adjust exposure with the `exposure time` slider. Take a picture -> adjust the exposure time -> take another picture -> repeat until photo is well exposed.

2. You can test different camera framerates in the `bandwidth manager`, but bandwidth during recording appears to be higher than in the bandwidth manager testing, as frames drop during recording but not testing.

3. The `Recording Settings` appear to reset when the program is closed and FOR EVERY CAMERA. Make sure the settings are correct every time before recording:
    - Set to Video
        - output format: AVI
        - Set fixed playback speed: OFF
    - Record every 1 frame
    - Output folder: `C:\Users\19716\pylon_recordings`


### Bandwidth