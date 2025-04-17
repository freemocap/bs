//
// Formatting
//
#set text(
    font: "New Computer Modern",
    size: 10pt
)

// Title 
#align(center, text(14pt)[
  *Understanding Neural Development Through Environment-Animal Perceptuomotor Experiences*
])
#align(center, text(12pt)[#smallcaps[research prospectus]])

// Author affiliations

#grid(
  columns: (1fr, 1fr),
  align(center)[
    *Jonathan Samir Matthis* \
    FreeMoCap Foundation / Northeastern University
  ],
  align(center)[
    *Benjamin Scholl* \
    Colorado University Anschutz  \
  ]
)

//
// MAIN BODY CONTENT 
//

== Overview
- How does behavioral and environmental experience shape the development of the musculoskeletal and nervous systems across the lifespan? 

- We know that environment drives body/neural development (hubel and wiesel), but the fidelity of our understanding of that connection is constrained by our ability to accurately record the relevant empirical measurements of the relevant environmental, behavioral, and neural data across both behavioral and developmental timescales. 

- We present a proposed research program in service of furthering our understanding of the neurally mediated relationship between the organism and its environment, and the role of the relationship in development across the lifespan. 

- We co-develop our research plan in parallel to a novel experimental apparatus representing our best approximation of the impossible aspirational goal of recording every relevant empirical aspect of the organism's environment, sensation, neural activation, and behavior. 
- Want to understand neural basis of visual/motor developement, specifically the way that developmental process is shaped by *actual experience*
    - The MOTOR activity associated with moving around the world
    - The PERCEPTUAL activity associated with the sensory consequences of the motor activity 
        - i.e. moving the body through space produces visual motion/optic flow
    - Agent (animal) and environment are linked via perception/action coupling
    - Presumably the activities that lead to reward (success) will result in strengthening of the sensory-motor pathways that were active in the time prior to the rewarding stimulus via dopaminergic reinforcement [Citation something]

== Measurements: 
== Direct measurements
    - *BODY:* Full 6 DoF kinematics  all body segments (esp the skull) in world-centered coordinates
    - *EYE:* Binocular horizontal, vertical, [torsional] position of each eye, in HEAD-CENTERED coordinates
    - *ENVIRONMENT:* Create accurate 3d models and representations of the enclosure space where the activities take place
    - *NEURAL* activity data (precisely time-synchronized to the BODY and GAZE data streams) 

=== Derived Measurements 
- With BODY and EYE data, we can compute: 
    - *GAZE:* Binocular horizontal, vertical, [torsional] position of each eye in WORLD-CENTERED coordinates (e.g. Matthis, Yates, Hayhoe 2018)
- With GAZE+ENVIRONMENT, we can compute: 
    - *RETINAL OPTIC FLOW* (e.g. Matthis et al 2022): 
        - Simple spherical pinhole camera model of the eye combined with gaze estimates gives us 6 DoF (technically 5 DoF because we don't have torsion) of each eyeball trajectory as the animal moves through its environment. 
        - Projecting the ENVIRONMENT onto the back of the eyeball model and tracking changes over time provides an estimate of retinal motion associated with the real-world recorded behavior of the animals over the course of their development. 
        - Coupling this estimated retinal motion with the oculumotor signals associated with the eye movements, the motor signals associated with the body movements, and the neural activity associated with all of the above will provide an unprecidented picture of the neurodevelopmental process


== Apparatus
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

== Planned activities
=== Lifespan experiences
- Gather longitudinal recordings from ferrets across lifespan in either `Control` or `Manipulated` condition (specific manipulation TBD)

- Record every day from birth until XXX weeks
- At end of lifespan:
  - Chronic anesthetized measurements to get:
    - retinal sensitivities to light, motion, color, etc
    
  - Full histological assay to record: 
    - *Musculoskeletal aspects*:
      - muscle volume/cross sectional area
      - bone density and functional morphology
      - bone/tendon junctions
      - etc
      
    - *Neural aspects:*
      - (??? Neurosceincey stuff of relevant cortical and subcortical areas) 
      
    - *Ocular aspects*: 
      - IOR of cornea
      - location/size of lens
      - Pupil size extents (at max/min iris constriction)
      
    
==== Control condition 
- Place animals in *`Control`* condition
  - Raised in "optimal" lighting (nice and bright, full spectrum, day/night cycle)
  - Rewarding standard interactions with the target mouse (with a tasty treat and a happy BEEP on successful capture)
  - Normal/Veridical relationship between movement and virtual environment (1:1 optic flow in response to head movement)

==== Manipulated condition 
- Animals in *`Manipulated`* condition (wherein we alter the developmental environment in some way),
  - Potential manipulations include: 
    - Raised in the dark (or in a particular color of light)
    - Single eye suture (no binocular info)
    - Slippery surface (place low-friction surface on ground of arena)
    - Manipulated rewarding interactions with target mouse (always turns left, always retreats linearly, etc)
    - Non-rewarding interactions with the mouse (same as Control, but no treat!!)
    - Manipulated/non-veridical relationship between movement and virtual environment (manipulate optic flow gain, direction, etc relative to animal movement)

=== Empirical outcome data 
- See https://freemocap.github.io/bs/deriveddata.html
- See @matthis2022
  
== Areas of interest: 
  - *Central and peripheral nervous systems*
    - Visual perception 
    - Oculomotor control
    - Action selection
    - Motor control
      
  - *Biomechanical/Musculoskeletal systems* 
    - Coordinated movement (limb movement kinematics)
    - Directed locomotion (full-body translation/energetics)
    - Orienting behavior (head-direction pointing)

  - *Development*
    - Body size development (bones)
    - Peripheral systems (eyeballs)
    
  - *Behavior*
    - Orienting behavior
    - Prey pursuit and capture
    - Environmental exploration/visual search


//
// END MAIN BODY CONTENT
// 


#bibliography("laser-ferrets.bib", style: "american-psychological-association")