FERRET_SKELETON = [
    ["nose", "left_eye"],
    ["nose", "right_eye"],
    ["left_eye", "left_ear"],
    ["right_eye", "right_ear"],
    ["left_ear", "base"],
    ["right_ear", "base"],
    ["base", "left_cam_tip"],
    ["base", "right_cam_tip"],
    ["base", "spine_t1"],
    ["spine_t1", "sacrum"],
]

# Full eye outline (closed ring) plus pupil-center spokes, for kpms visualisation.
# Top arc (p1 at apex):   tear_duct — p3 — p2 — p1 — p8 — p7 — outer_eye
# Bottom arc:             tear_duct — p4 — p5 — p6 — p7
# Pupil spokes to cardinal points: top (p1), medial (p3), bottom (p5), lateral (p7)
EYE_SKELETON_3D = [
    # top arc
    ["tear_duct", "p3"],
    ["p3", "p2"],
    ["p2", "p1"],
    ["p1", "p8"],
    ["p8", "p7"],
    ["p7", "outer_eye"],
    # bottom arc
    ["tear_duct", "p4"],
    ["p4", "p5"],
    ["p5", "p6"],
    ["p6", "p7"],
    # pupil spokes
    ["pupil_center", "p1"],
    ["pupil_center", "p3"],
    ["pupil_center", "p5"],
    ["pupil_center", "p7"],
]

SKULL_AND_GAZE_SKELETON = [
    *FERRET_SKELETON,
    ["left_eye", "left_gaze_target"],
    ["right_eye", "right_gaze_target"],
]

BOTH_EYES_SKELETON = [
    *[["left_" + a, "left_" + b] for a, b in EYE_SKELETON_3D],
    *[["right_" + a, "right_" + b] for a, b in EYE_SKELETON_3D],
    ["left_tear_duct", "right_tear_duct"],  # medial bridge
]

# The skull_kinematics rigid body only carries its own 8 markers (no
# spine_t1/sacrum -- see bodyparts.HEAD_BODYPARTS), so this is FERRET_SKELETON
# minus the two edges that reach past "base" into the spine.
HEAD_SKELETON = [
    ["nose", "left_eye"],
    ["nose", "right_eye"],
    ["left_eye", "left_ear"],
    ["right_eye", "right_ear"],
    ["left_ear", "base"],
    ["right_ear", "base"],
    ["base", "left_cam_tip"],
    ["base", "right_cam_tip"],
]

# Tracked pupil boundary ring (p1..p8, closed octagon) plus cardinal spokes
# to pupil_center (top p1, medial p3, bottom p5, lateral p7 — same convention
# as EYE_SKELETON_3D's pupil spokes), for each eye, anchored to the skull's
# eye keypoint.
_PUPIL_RING = [[f"p{i}", f"p{i % 8 + 1}"] for i in range(1, 9)]
_PUPIL_SPOKES = [["pupil_center", f"p{i}"] for i in (1, 3, 5, 7)]

HEAD_WITH_PUPIL_POINTS_SKELETON = [
    *HEAD_SKELETON,
    ["left_eye", "left_pupil_center"],
    ["right_eye", "right_pupil_center"],
    *[["left_" + a, "left_" + b] for a, b in [*_PUPIL_RING, *_PUPIL_SPOKES]],
    *[["right_" + a, "right_" + b] for a, b in [*_PUPIL_RING, *_PUPIL_SPOKES]],
]