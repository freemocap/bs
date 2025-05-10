# Cameras

## Multicamera Recording
1. Go to VSCode and it should be open to the repo `bs`
2. In the file menu (top left) navigate to `python_code/cameras/multicamera_recording.py` and click it, then scroll all the way to the bottom where it says `if name==“main”:`
3. Change the recording name variable so match the naming Schema you want. If this is the first recording of the day, or you have touched the cameras at all, they need to be recalibrated.
4. Once you have the name you want, scroll down to where there are multiple grab options. All but one should be commented out. The options are:
    - `grab_n_frames`, which will grab the exact number of frames you input. 
    - `grab_n_seconds` will grab for the amount of time you input. If you want a 20 minutes recording, input `20*60`. 
    - `grab_until_input` will run iuntil you type enter in the terminal at the bottom of backed. This is useful for recording where you start Basler and and pupil, let it run until you’ve collected the data you need, and then stop both manually 
5. Click the run button in the upper right.
6. The recording will be saved to `/home/scholl-lab/recordings` inside a session folder for that day.

## Pupil Recording

1. Open a new terminal
2. Run `conda activate pupil_source`
3. Run `cd Documents/git_repos/pupil/pupil_src`
4. Run `python main.py capture`
5. The pupil gui will open, and if they eye cameras don't automatically open go to settings and click `detect eye 0` and `detect eye 1` 
6. Change the recording name under `Recorder: Recording session name`
7. You can record by pressing the R on screen or hitting `r` on the keyboard
8. Repeat whatever command from `7` to stop recording. 
9. Recordings are saved to `/home/scholl-lab/pupil_recordings`  

## Synchronizing Basler

## Resources
Some more good info on using pypylon: https://pythonforthelab.com/blog/getting-started-with-basler-cameras/