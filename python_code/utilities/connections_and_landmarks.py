ferret_head_spine_landmarks = {
    "nose": 0,
    "left_cam_tip": 1,
    "right_cam_tip": 2,
    "base": 3,
    "left_eye": 4,
    "right_eye": 5,
    "left_ear": 6,
    "right_ear": 7,
    "spine_t1": 8,
    "sacrum": 9,
    "tail_tip": 10,
    "center": 11,
}

ferret_head_spine_connections = (
    (0, 5),
    (0, 4),
    (5, 7),
    (4, 6),
    (3, 1),
    (3, 2),
    (3, 8),
    (8, 9),
    (9, 10),
)

toy_landmarks = {
    "front": 0,
    "top": 1,
    "back": 2,
}

toy_connections = (
    (0, 1),
    (1, 2),
)