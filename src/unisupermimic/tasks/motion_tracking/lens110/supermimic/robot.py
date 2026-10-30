"""Lens110 grouped joint order and nominal pose from Project 10's training asset."""

JOINT_NAMES = (
    [
        f"{side}_{part}_joint"
        for side in ("left", "right")
        for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + ["torso_yaw_joint"]
    + [
        f"{side}_{part}_joint"
        for side in ("left", "right")
        for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
    ]
)

TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_hand_tip",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_hand_tip",
]

# Exact nominal pose in the existing training URDF's lens110_dance.py.
DEFAULT_Q = [
    -0.14,
    -0.01,
    -0.1,
    0.36,
    -0.2575,
    0.0,
    -0.14,
    -0.01,
    0.1,
    0.36,
    -0.2575,
    0.0,
    0.0,
    0.4,
    0.2,
    0.0,
    -0.8,
    0.4,
    -0.2,
    0.0,
    -0.8,
]

OBS_DIM = 75
GOAL_FRAME_DIM = 57
STATE_DIM = 55
CONTEXT_DIM = 25
