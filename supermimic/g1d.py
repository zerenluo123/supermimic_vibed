"""G1D joint layout used by the trainer. The plant stays the two-slide MJCF; the network state collapses them."""

from __future__ import annotations

# MuJoCo order after the three planar root joints. dof_pos in the retarget pkl is this order.
LIFT_JOINTS = ("LZ_mt_Joint", "LZ_it_Joint")
UPPER_JOINTS = [
    "Yaw_Joint",
    "torso_Joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# Motor class for the G1 action-scale formula. Yaw_Joint is the waist pitch (axis y).
UPPER_MOTORS = [
    "waist_pitch",
    "waist_yaw",
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
]
# State q: one column height, then the 16 position-controlled joints.
STATE_JOINTS = ["column", *UPPER_JOINTS]
TRACKED_BODIES = [
    "AGV_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
# Each slide joint carries half of the column coordinate.
COUPLED = {"LZ_mt_Joint": (0, 0.5), "LZ_it_Joint": (0, 0.5)}
COLUMN_LIMIT = (0.0, 0.42)
VX_MAX, WZ_MAX, VZ_MAX = 1.5, 0.6, 0.0765
TAU = 0.15
BASE_JOINTS = ("base_x_joint:0", "base_x_joint:1", "base_x_joint:2")

# Observation: 16 + 16 + 1 + 1 + 3 + 3 + 19. Velocity scale matches G1 (0.1) on the arm
# joints; column speed stays in m/s because its full scale is 0.0765.
OBS_NOISE_WIDTHS = (
    [0.01] * 16
    + [0.5] * 16
    + [0.01, 0.01]
    + [0.2] * 3
    + [0.05] * 3
    + [0.0] * 19
)
OBS_SCALES = (
    [1.0] * 16
    + [0.1] * 16
    + [1.0, 1.0]
    + [0.25] * 3
    + [1.0] * 3
    + [1.0] * 19
)
