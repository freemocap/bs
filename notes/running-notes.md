# Ben Scholl / Jon Matthis - Running Notes

[![hackmd-github-sync-badge](https://hackmd.io/svBEsisvQ1yc8UMgiw97Wg/badge)](https://hackmd.io/svBEsisvQ1yc8UMgiw97Wg)


## 2024-03-22
- Planning getting ben up to speed on the whole GH workflow
- Ben went to Germany and has ferret eye tracking stuff (hypothetically) 
    - still a whole bunch of little things to handle on the hard/software thing
    - lots of cool tricks around calibration/cameras IR vs visible light etc
- Ferret protocol approved
- got access to the room 
    - can start building arena 
    - its a 1m x 1m x 1m  cube
        - arena floor ~.25"
    - got an XY gantry to control things
        - canadian company called Zaber
        - SDK - https://software.zaber.com/motion-library/docs/tutorials/code
        - this Gantry is XXY (two X motors that are yoked together)
            - way to drive a *motor group* vs a simple motor
        - https://www.zaber.com/products/xy-xyz-gantry-systems
        - https://www.zaber.com/products/controllers-joysticks/X-MCC

- Trying to get the germany folks to send the info and hardware and whatnot
- End of the day - ben wants to:
    - build an arena that does something similar to Kate's tracking task
    - a tracking task where the animal is chasing the toy around the arena
    - Building an experimental paradigm
        - design arbitrary trajectories for the toy 
        - want to be able vary things like
            - velocity
            - trajecotry
                - etc - arbitrary parameter space 
    - EVENTUALLY
        - realtime tracking and `closed loop` methods to interact with the animal
- Thinking about data management
    - Building a NAS serve
    - processing locally vs on cloud
- No immedediate fixed points or deadlines for BS in the immediate future
    - Nothing specific in th next 3-6 months
    - next planned grants sorta flexible, no big grant plans in the next year or two 
- wants two more meetings
    - intro to GH 
    - show more specifically the pieces
        - the 'tangibles' around i.e. cat/animal tracking
- Jon will arrange next meeting to share progress and updates on
    - project plan/timeline etc
    - rig mockup
    - progress on kinematic tracking pipeline
    - progress on motor control stuff 