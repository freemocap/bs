notes/funding/2025-nsf-24-546/matthis-scholl-FOA-NSF-24-546-prospectus.typ
#import "nsf-prospectus-template.typ": prospectus

#show: prospectus.with(
  title: "Understanding function and development of the perceptuomotor system through environment-animal interactions",
  authors: (
    (name: "Benjamin Scholl", affiliations: (1), email: "benjamin.scholl@cuanschutz.edu"),
    (name: "Jonathan Samir Matthis", affiliations: (2,3), email: "jonmatthis@gmail.com"),

  ),
  affiliations: (
    (number: 1, name: "University of Colorado Anshutz"),
    (number: 2, name: "FreeMoCap Foundation"),
    (number: 3, name: "Northeastern University"),
  ),
  program: "NSF-24-546",
  date: "April 2025",
  bibfile: "laser-ferrets.bib",
)

= Research Overview

- Environmental energy elicits patterns of perceptual neural activity which form the basis of goal-directed motor behavior.


- The coupling between an organisms moment-to-moment perceptual input and their motor output defines the basic agent/environment Perception-Action Loop (@Warren2006) which underpins goal directed behavior (such as chasing a moving target)
    - Animal moves, generate retinal optic flow
    - Retinal optic flow specifies egocentric movement relative to external objects (@matthis2022)
    - Self-motion estimates drive locomotor state and goal (@Fajen2003)
    - Behavioral goals/Task (@hayhoe2005) dictates information needed for successful behavior (which drives oculomotor behavior, e.g. gaze targeting and stabilization (@matthis2018))

- This Perception/Action cycle has been studied extensively in humans, but research is lacks neural grounding
- Similarly, the neural basis of oculomotor control and low level visual perception well mapped in various animal models, but behavior often extremely impoverished and so lack ecological validity.

- This project seeks to under stand the functional mappings between gaze, locomotion, and behavior

- We co-develop our research plan in parallel to a novel experimental apparatus representing our best approximation of the impossible aspirational goal of recording every relevant empirical aspect of the organism's environment, sensation, neural activation, and behavior.

- Specifically, we will record full-body kinematics and binocular gaze data in order to directly simulate the #smallcaps[binocular retinal optic flow] patterns associated with the animal's real-world environment interactions (@matthis2018). 

- These patterns capture the geometric aspects of the Animal/Environment interaction, and provide estimates of the task-relevant environmental illumination patterns that the ocular and visual systems evolved to detect. 

- These estimates will guide [THE WAY WE DO #smallcaps[neuropixel] STUFF] and [THE WAY WE DO THE END OF LIFE RETINA STUFF], with the goal of understanding the neural structures and functional pathways that define the animal/environment behavioral coupling. 

== Measurements:
== Direct measurements
    - #smallcaps[body:] Full 6 DoF kinematics  all body segments (esp the skull) in world-centered coordinates
    - #smallcaps[eye:] Binocular horizontal, vertical, [torsional] position of each eye, in HEAD-CENTERED coordinates
    - #smallcaps[environment:] Create accurate 3d models and representations of the enclosure space where the activities take place
    - #smallcaps[neural] activity data (precisely time-synchronized to the BODY and GAZE data streams)

=== Derived Measurements
- From #smallcaps[body] data, we can compute:
    - #smallcaps[limb coherence] - Measuing coherence between movement patterns of Left/Right/Fore/Hind limb pairs
    - #smallcaps[locomotion] - e.g. locomotor state, direction, speed, efficiency, etc
- With #smallcaps[body] + #smallcaps[eye] data, we can compute:
    - #smallcaps[gaze:] Binocular horizontal, vertical, [torsional] position of each eye in WORLD-CENTERED coordinates (@matthis2018)
- With #smallcaps[gaze]+#smallcaps[environment], we can compute:
    - #smallcaps[gaze target] (e.g. @wallace2025)
        - Projecting binocular gaze vectors into the world to identify when #smallcaps[target] falls onto Area Centralis
    - #smallcaps[retinal optic flow] (@matthis2022):
        - Simple spherical pinhole camera model of the eye combined with gaze estimates gives us 6 DoF (technically 5 DoF because we don't have torsion) of each eyeball trajectory as the animal moves through its environment.
        - Projecting the #smallcaps[environment] onto the back of the eyeball model and tracking changes over time provides an estimate of retinal motion associated with the real-world recorded behavior of the animals over the course of their development.


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
    - #smallcaps[Musculoskeletal aspects]:
      - muscle volume/cross sectional area
      - bone density and functional morphology
      - bone/tendon junctions
      - etc

    - #smallcaps[Neural aspects]:
      - (??? Neurosceincey stuff of relevant cortical and subcortical areas)

    - #smallcaps[Ocular aspects]:
      - IOR of cornea
      - location/size of lens
      - Pupil size extents (at max/min iris constriction??)
      - Oculomotor muscle max/min length(??)


==== Control condition
- Place animals in #smallcaps[`Control`] condition
  - Raised in "optimal" lighting (nice and bright, full spectrum, day/night cycle)
  - Rewarding standard interactions with the target mouse (with a tasty treat and a happy BEEP on successful capture)
  - Normal/Veridical relationship between movement and virtual environment (1:1 optic flow in response to head movement)

==== Manipulated condition
- Animals in #smallcaps[`Manipulated`] condition (wherein we alter the developmental environment in some way),
  - Manipulations:
    - Perceptual Input:
      - Raised in the dark (or in a particular color of light)
      - Single eye suture (no binocular info)
    - Environment: 
      - Slippery surface (place low-friction surface on ground of arena)
      - Manipulated/non-veridical relationship between movement and virtual environment (manipulate optic flow gain, direction, etc relative to animal movement)
    - Task: 
      - Manipulated rewarding interactions with target mouse (always turns left, always retreats linearly, etc) - I think we should do this one!
      - Non-rewarding interactions with the mouse (same as Control, but no treat!!)