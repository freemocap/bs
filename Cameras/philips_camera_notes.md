# Philip's Notes on Camera Setup 

## Physical Setup Notes
1. There is some reflection from the charuco in 2 camera views (cameras ...832 and ...609). Is there something we can put behind the plexiglass to prevent this? It may mess up calibration and tracking.

2. Camera positioning should match the blender file, all be upright, etc.

## Pylon Notes

1. You can adjust exposure with the `exposure time` slider. Take a picture -> adjust the exposure time -> take another picture -> repeat until photo is well exposed.

2. You can test different camera framerates in the `bandwidth manager`, but bandwidth during recording appears to be higher than in the bandwidth manager testing, as frames drop during recording but not testing.

3. The `Recording Settings` need to be set for every camera, but the settings do persist when closing the software. Make sure the settings are correct every time before recording:
    - Set to Video
        - output format: AVI
        - Set fixed playback speed: OFF
    - Record every 1 frame
    - Output folder: `C:\Users\19716\pylon_recordings`


### Bandwidth

I was able to get 4 cameras running at 45 fps by limiting the bandwidth (right click on cameras and select Bandwidth Manager) to 190. This ran a good number of 30 second trials and only errored once. This suggests the current USB bandwidth limit is ~750 MB/s, so we can use that for testing for now.

However, the actual recording drops frames when running at 45 fps. It seems much more bandwidth is needed for the actual recording, or there's another limiting factor in the pipeline from camera -> disk.

Getting 1 dropped frame on a single camera recording at 15 fps:
```
Type	Time	Source	Message
Error	2024-09-10 11:02:43.186	Basler acA2040-90umNIR (24908831)	Image acquisition on "Basler acA2040-90umNIR (24908831)" failed! Error: "Payload data has been discarded. Payload data can be discarded by the camera device if the available bandwidth is insufficient."
```

# Crash Reports

- I tried setting the location to the E drive and it crashed once I clicked record on the 4th camera. Not sure if it's save location specific yet.

