"""One-env check that TrackingSim(kind='g1d') reads and writes the AGV the way training will.

Headless. Builds a synthetic clip, resets onto it, and commands full vx and column speed.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import add_launcher_args, launch_simulation

import argparse

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()

import joblib
import numpy as np
import torch

from supermimic.env import TrackingSim, make_sim
from supermimic.g1d import COLUMN_LIMIT, COUPLED, STATE_JOINTS, TRACKED_BODIES
from supermimic.kinematics import G1DKinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.robot import g1d_default_q

dev = "cuda:0"
xml = os.path.join(REPO, "data", "g1d_description", "g1d.xml")
kin = G1DKinematics(dev, xml, STATE_JOINTS, TRACKED_BODIES, COUPLED, COLUMN_LIMIT)
yaw = 0.5
half = 0.5 * yaw
root_rot = np.tile([0.0, 0.0, math.sin(half), math.cos(half)], (12, 1)).astype(np.float32)
root_pos = np.tile([0.3, -0.2, 0.1113], (12, 1)).astype(np.float32)
dof = np.zeros((12, 18), np.float32)
dof[:, 0] = 0.1
dof[:, 1] = 0.1
dof[:, 2] = 0.25
clip_dir = tempfile.mkdtemp()
joblib.dump({"fps": 50.0, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof}, os.path.join(clip_dir, "clip.pkl"))
motion = MotionLib.from_g1d_dir(clip_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(g1d_default_q(), device=dev), obs_kind="g1d")

sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, 1, motion, mdp, eval_mode=True, kind="g1d")
    env.reset(torch.zeros(1, dtype=torch.long, device=dev), motion.clip_start.clone())
    env.refresh()
    p, R, v, w, q, qd = env.state()
    got_yaw = math.atan2(float(R[0, 1, 0]), float(R[0, 0, 0]))
    print(f"reset p={p[0].tolist()} yaw={got_yaw:.4f} height={float(q[0, 0]):.4f} waist={float(q[0, 1]):.4f}", flush=True)
    assert abs(float(p[0, 2]) - 0.1113) < 0.01, p
    assert abs(float(p[0, 0]) - 0.3) < 0.02, p
    assert abs(float(p[0, 1]) + 0.2) < 0.02, p
    assert abs(got_yaw - yaw) < 0.05, got_yaw
    assert abs(float(q[0, 0]) - 0.2) < 0.01, q[0, 0]
    assert abs(float(q[0, 1]) - 0.25) < 0.02, q[0, 1]
    a = torch.zeros(1, 19, device=dev)
    a[0, 16] = 1.0
    a[0, 18] = 1.0
    for _ in range(10):
        env.step(a)
    p, R, v, w, q, qd = env.state()
    print(f"drive v={v[0].tolist()} w={w[0].tolist()} height={float(q[0, 0]):.4f} qd0={float(qd[0, 0]):.4f}", flush=True)
    expect_x = 1.5 * math.cos(yaw)
    assert float(v[0, 0]) > 0.4 * expect_x, v
    assert float(q[0, 0]) > 0.205, q[0, 0]
    assert float(qd[0, 0]) > 0.02, qd[0, 0]
    print("g1d tracking env ok", flush=True)
