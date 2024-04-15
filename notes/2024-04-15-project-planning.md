# 2024-04-15-ben-scholl-project-planning


## [Milestone] Proof of Concept - Track the Cat
- [ ] Get 2d track using custom DLC pipeline
    - [ ] Clean up DLC script
    - [ ] Label the data 
    - [ ] Train model 
    - [ ] Iterate until 'good enough' 
        - 'Good enough' defined relative to desired research outcomes
        - depending on how this part goes, there will be decisions to make re: 
            - what can be done with the kind of data we can pull out of a custome DLC pipeline
            - what other options exist for building non-DLC tracking methods
- [ ] Reconstruct 3d trajectories
    - [ ] Adapt `freemocap` pipeline
- [ ] Reconstruct Kinematics (i.e. joint angles)
    - [ ] Build new/adapt human-focused `freemocap` pipelines
- [ ] Calculate center-of-mass
    - [ ] Get proportional segment-mass number (or make them up)
- [ ] Visualize in Blender
    - [ ] Adapt `freemocap` human-centered armature methods for quadruped
    - [ ] Ditto for mesh building methods

## [Milestone] Tracking ferrets
- [ ] Adapting `cat` methods to ferrets
- [ ] Ferret eye tracking stuff 
    - [ ] get base eye tracking data
    - [ ] intergrate with mocap
        - [ ] Spatial calibration
        - [ ] Temporal synchronization

## Building mocap rig
### Choosing cameras
- [ ] Choose sensor - resolution/framerate/sensor geometry
- [ ] Choose lens - focal length, fixed/variable etc
- [ ] Choose 'camera -> PC' methods (various)
- [ ] RGB vs IR
- [ ] Model/Brand

### Building motor tracker gantry
- [ ] Figure out how to drive motors
- [ ] Build electro-magnet set up

### Build data management plan
- [ ] Record to disk somehow
- [ ] Backup and share data

### Force plates/direct center-of-pressure analysis ? 
- actual force plates (interferes with magnet)
- some kind of laser-grid (won't interfere, but lower accuracy) 
- some kind of clever rubber-floor (best of both, but requires being clever)

## General tasks
- [ ] Spatial Calibration 
- [ ] Temporal Synchronization
- [ ] Validation
    - [ ] Does it work at all? 
        - i.e. does visualization "look right"
    - [ ] Measure Precision
    - [ ] Measure Accuracy



___


## NOTES
### Levels of "done"
- Feasibility:
    -  Convince yourself its possible
- Proof of concept (PoC): 
    - Technically complete the basic tasks, at least one person one time 
    - e.g. A manuscript could be built on a PoC
- Minimum Viable Product (MVP): 
    - Actually complete the basic taks (at least one person, multiple times)
    - e.g. an R01 style grant could be written on an MVP
- Alpha: 
    - Multiple people can do it multiple times
    - known broken, might require help to set up
- Beta
    - Multiple people, multiple times
    - Possibly working, theoretically could set up without help
- Full 
    - the thing itself.
    - iterative development from there using semantic versioning (MAJOR.MINOR.PATCH)
        - if this is `v01.0`, re-start the process in prep for `v2.0`
