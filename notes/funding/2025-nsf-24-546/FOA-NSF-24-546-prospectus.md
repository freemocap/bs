# Recording the full perception-action-environment cycle during real-world natural behavior across the lifespan   

## Overview
- How does behavioral and environmental experience shape the development of the musculoskeletal and nervous systems across the lifespan? 

- We know that environment drives body/neural development (hubel and wiesel), but the fidelity of our understanding of that connection is constrained by our ability to accurately record the relevant empirical measurements of the relevant environmental, behavioral, and neural data across both behavioral and developmental timescales. 

- We present a proposed research program in service of furthering our understanding of the neurally mediated relationship between the organism and its environment, and the role of the relationship in development across the lifespan. 

- We co-develop our research plan in parallel to a novel experimental apparatus representing our best approximation of the impossible aspirational goal of recording every relevant empirical aspect of the organism's environment, sensation, neural activation, and behavior. 

## Apparatus
- This apparatus represents a 1m^2 behavioral arena outfitted with the following spatially calibrated and temporally synchronized empirical systems: 
  - Full-body kinematic markerless motion capture (~90Hz)
  - 6-axis head-mounted Inertial Measurement Unit (IMU)
  - Binocular eye/gaze tracking (200x200px, ~200Hz in each eye)
  - Head-mounted (first person view) camera (400x400px, 120Hz)
  - Neuropixel recordings (??? Numbers, Hz, etc)

- We have the following mechanisms of environment manipulation:
  - Automatically controllable mouse target 
    - Controlled via 2-axis magnetic gantry 
    - Houses a tasty treat that the animal gets to eat if they catch the target
    - Can be automated or manually controlled
  - 360-degree virtual reality display 
    - controllable via closed loop connection to the head sensor

## Planned activities
### Lifespan experiences
- Gather longitudinal recordings from ferrets across lifespan in either `Control` or `Manipulated` condition (specific manipulation TBD)

- Record every day from birth until XXX weeks
- At end of lifespan:
  - Chronic anesthetized measurements to get:
    - retinal sensitivities to light, motion, color, etc
    
  - Full histological assay to record: 
    - **Musculoskeletal aspects**:
      - muscle volume/cross sectional area
      - bone density and functional morphology
      - bone/tendon junctions
      - etc
      
    - **Neural aspects:**
      - (??? Neurosceincey stuff of relevant cortical and subcortical areas) 
      
    - **Ocular aspects**: 
      - IOR of cornea
      - location/size of lens
      - Pupil size extents (at max/min iris constriction)
      
    
#### Control condition 
- Place animals in **`Control`** condition
  - Raised in "optimal" lighting (nice and bright, full spectrum, day/night cycle)
  - Rewarding standard interactions with the target mouse (with a tasty treat and a happy BEEP on successful capture)
  - Normal/Veridical relationship between movement and virtual environment (1:1 optic flow in response to head movement)

#### Manipulated condition 
- Animals in **`Manipulated`** condition (wherein we alter the developmental environment in some way),
  - Potential manipulations include: 
    - Raised in the dark (or in a particular color of light)
    - Single eye suture (no binocular info)
    - Slippery surface (place low-friction surface on ground of arena)
    - Manipulated rewarding interactions with target mouse (always turns left, always retreats linearly, etc)
    - Non-rewarding interactions with the mouse (same as Control, but no treat!!)
    - Manipulated/non-veridical relationship between movement and virtual environment (manipulate optic flow gain, direction, etc relative to animal movement)

### Empirical outcome data 
## Areas of interest: 
  - Central and peripheral nervous systems
    - Visual perception 
    - Oculomotor control
    - Action selection
    - Motor control
      
  - Biomechanical/Musculoskeletal systems 
    - Coordinated movement (limb movement kinematics)
    - Directed locomotion (full-body translation/energetics)
    - Orienting behavior (head-direction pointing)

  - Development
    - Body size development (bones)
    - Peripheral systems (eyeballs)
    
  - Behavior
    - Orienting behavior
    - Prey pursuit and capture
    - Environmental exploration/visual search
