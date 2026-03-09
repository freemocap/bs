# Optical Flow

Calculate optical flow for pupil videos.

## Run

Change the video file path in `__main__.py` and run it. If you haven't processed that video before, it will prompt you to draw a crop around the eye. Press enter to accept the crop, and then it wil run the flow calculation. You can toggle whether the flow displays, and whether it records (saves to video). Thre's also an option for `full_plot`. If this is selected it shows a full matplotlib plot with the flow video and histograms, but runs very slowly.