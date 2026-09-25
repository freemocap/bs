FERRET_BODYPARTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_cam_tip",
    "right_cam_tip",
    "base",
    "spine_t1",
    "sacrum",
]

FERRET_ANTERIOR_BODYPARTS = ["nose"]
FERRET_POSTERIOR_BODYPARTS = ["sacrum"]

EYE_BODYPARTS = [
    "tear_duct",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "outer_eye",
    "pupil_center",
]

EYE_ANTERIOR_BODYPARTS = ["tear_duct"]
EYE_POSTERIOR_BODYPARTS = ["outer_eye"]

BOTH_EYES_BODYPARTS = [
    "left_tear_duct",  "left_p1",  "left_p2",  "left_p3",  "left_p4",
    "left_p5",         "left_p6",  "left_p7",  "left_p8",
    "left_outer_eye",  "left_pupil_center",
    "right_tear_duct", "right_p1", "right_p2", "right_p3", "right_p4",
    "right_p5",        "right_p6", "right_p7", "right_p8",
    "right_outer_eye", "right_pupil_center",
]

BOTH_EYES_ANTERIOR_BODYPARTS = ["left_tear_duct", "right_tear_duct"]
BOTH_EYES_POSTERIOR_BODYPARTS = ["left_outer_eye", "right_outer_eye"]

SKULL_AND_GAZE_BODYPARTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_cam_tip",
    "right_cam_tip",
    "base",
    "spine_t1",
    "sacrum",
    "left_gaze_target",
    "right_gaze_target",
]

SKULL_AND_GAZE_ANTERIOR_BODYPARTS = ["nose"]
SKULL_AND_GAZE_POSTERIOR_BODYPARTS = ["sacrum"]

# The RigidBodyKinematics for the skull (skull_kinematics.parquet) only
# carries its own 8 reference-geometry markers -- unlike FERRET_BODYPARTS,
# it has no spine_t1/sacrum (those are solver-only markers, not part of the
# skull rigid body's keypoint set).
HEAD_BODYPARTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_cam_tip",
    "right_cam_tip",
    "base",
]

HEAD_WITH_PUPIL_POINTS_LEFT_EYE_BODYPARTS = [
    "left_pupil_center", "left_p1", "left_p2", "left_p3", "left_p4",
    "left_p5",           "left_p6", "left_p7", "left_p8",
]
HEAD_WITH_PUPIL_POINTS_RIGHT_EYE_BODYPARTS = [
    "right_pupil_center", "right_p1", "right_p2", "right_p3", "right_p4",
    "right_p5",           "right_p6", "right_p7", "right_p8",
]

HEAD_WITH_PUPIL_POINTS_BODYPARTS = [
    *HEAD_BODYPARTS,
    *HEAD_WITH_PUPIL_POINTS_LEFT_EYE_BODYPARTS,
    *HEAD_WITH_PUPIL_POINTS_RIGHT_EYE_BODYPARTS,
]

HEAD_WITH_PUPIL_POINTS_ANTERIOR_BODYPARTS = ["nose"]
# No spine_t1/sacrum available on this keypoint set (see HEAD_BODYPARTS); use
# "base" (skull base, opposite the nose) as the posterior anchor instead.
HEAD_WITH_PUPIL_POINTS_POSTERIOR_BODYPARTS = ["base"]
