# Real-world developmental laser ferrets 

## Scientific angle
- Want to understand neural basis of visual/motor developement, specifically the way that developmental process is shaped by **actual experience**
    - The MOTOR activity associated with moving around the world
    - The PERCEPTUAL activity associated with the sensory consequences of the motor activity 
        - i.e. moving the body through space produces visual motion/optic flow
    - Agent (animal) and environment are linked via perception/action coupling
    - Presumably the activities that lead to reward (success) will result in strengthening of the sensory-motor pathways that were active in the time prior to the rewarding stimulus via dopaminergic reinforcement [Citation something]
- In order to understand this sensory-motor-world coupling, we endeavor to record as MUCH of the relevant data as we possibly can, up to the capacities of the current state of technology
    - Want to record: 
        - **BODY:** Full 6 DoF kinematics  all body segments (esp the skull) in world-centered coordinates
        - **EYE:** Binocular horizontal, vertical, [torsional] position of each eye, in HEAD-CENTERED coordinates
        - **ENVIRONMENT:** Create accurate 3d models and representations of the enclosure space where the activities take place
        - **NEURAL** activity data (precisely time-synchronised to the BODY and GAZE data streams) 

    - With BODY and EYE data, we can compute: 
        - **GAZE:** Binocular horizontal, vertical, [torsional] position of each eye in WORLD-CENTERED coordinates (e.g. Matthis, Yates, Hayhoe 2018)
    - With GAZE+ENVIRONMENT, we can compute: 
        - **RETINAL OPTIC FLOW** (e.g. Matthis et al 2022): 
            - Simple spherical pinhole camera model of the eye combined with gaze estimates gives us 6 DoF (technically 5 DoF because we don't have torsion) of each eyeball trajectory as the animal moves through its environment. 
            - Projecting the ENVIRONMENT onto the back of the eyeball model and tracking changes over time provides an estimate of retinal motion associated with the real-world recorded behavior of the animals over the course of their development. 
            - Coupling this estimated retinal motion with the oculumotor signals associated with the eye movements, the motor signals associated with the body movements, and the neural activity associated with all of the above will provide an unprecidented picture of the neurodevelopmental process

- Technical challenges
    - Because this project relies on numerous data streams recording from disparate systems and requiring distinct processing pipelines, it is essential to develop a sophisticated research infrastructure to manage the temporal synchronization between all systems, as well as the spatial calibration between each raw data stream and its associated reference frame (world-centered, head-centered, eye-centered) and manage transformations and projections from one reference frame to another (e.g. projection world-centered poitns from the environment onto an eye-centered simulated retina for optic flow)
    - This need is compounded by the fact that this is a **developmental** study, so we must be able to record from all of these streams repeatedly and reliably across the different timepoints of the animals' lives
    - The level of technical sophistication required to build this system is beyond what is possible to produce through standard academic means (i.e. grad student or post-doc derived spaghetti code)
    - If we do this right, we will be able to not only build THIS research poject on the back of the resulting technical tool, but we will be able to adapt these methods other research programs and species (rodents, humans, etc)

        