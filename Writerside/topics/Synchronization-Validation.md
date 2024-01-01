# Synchronization Validation

Ensure that:
- Each camera records at a consistent framerate
  - Outcome measures:
    - Dropped Frames
      - total
      - percentage
      - per second
    - Frame Duration
      - Mean
      - Median
      - Std Dev
- MultiCamera Frames are within expected temporal envelope
  - Outcome measures:
    - within-multiframe
      - Interframe interval 
    - between-multiframe
      - inter-multiframe-duration
        - as above, but relative to either:
          - Idealized timestamps
          - Camera0 timestamp
          - Mean/Median timestamp
