#let prospectus(
  title: "Research Prospectus",
  author: "Researcher Name",
  date: "May 2024",
  institution: "University Name",
  department: "Department Name",
  email: "researcher@university.edu",
  body
) = {
  // Page setup
  set page(
    paper: "a4",
    margin: (x: 2.5cm, y: 2.5cm),
  )

  // Font setup
  set text(
    font: "New Computer Modern",
    size: 11pt,
    hyphenate: false,
  )

  // Title
  align(center)[
    #block(text(weight: "bold", size: 16pt)[#title])
    #v(0.5cm)
    #block(text(size: 12pt)[#author])
    #block(text(size: 11pt)[#institution, #department])
    #block(text(style: "italic", size: 10pt)[#email])
    #block(text(size: 10pt)[#date])
    #v(0.8cm)
  ]

  // Main content with subtle headings
  set heading(
    numbering: none,
    outlined: false,
  )

  // Body content
  body
}

#show: prospectus.with(
  title: "Investigating Neural Network Approaches to Climate Prediction",
  author: "Jane A. Researcher",
  institution: "University of Science",
  department: "Department of Computer Science",
  email: "jresearcher@science.edu",
  date: "May 15, 2024",
)

= Research Context
#v(-0.2cm)
This research aims to address a critical gap in climate prediction models by leveraging recent advances in deep learning architectures. Current approaches face limitations in capturing complex non-linear relationships between atmospheric variables and accurately predicting extreme weather events.

= Research Objectives
#v(-0.2cm)
The primary objectives of this research are to:
- Develop a novel neural network architecture that integrates temporal and spatial climate data
- Improve prediction accuracy for extreme weather events by at least 15% over existing models
- Create more computationally efficient models that can run on standard research hardware
- Validate models against historical climate data from multiple geographical regions

= Methodology
#v(-0.2cm)
This study will employ a mixed-methods approach combining:
1. Transformer-based architecture with attention mechanisms specialized for climate data
2. Integration of satellite imagery with traditional meteorological measurements
3. Comparative analysis against existing statistical and machine learning models
4. Rigorous validation using holdout test sets from diverse climate zones

= Anticipated Outcomes
#v(-0.2cm)
This research is expected to produce: (1) a novel neural network architecture specifically optimized for climate prediction; (2) open-source implementation accessible to the research community; (3) improved forecasting capabilities for extreme weather events; and (4) insights into the most significant predictive features for different climate phenomena.

= Resources and Timeline
#v(-0.2cm)
The research will require access to high-performance computing resources, climate datasets from NOAA and the IPCC, and collaboration with the Climate Modeling Group. I anticipate completing this work over 12 months, with preliminary results available after 6 months.