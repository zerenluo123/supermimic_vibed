"""CPU checks for the G1D trainer layout. Does not launch Isaac."""

from __future__ import annotations

import os
import sys
import tempfile

import joblib
import mujoco
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from supermimic.buffer import ReplayBuffer
from supermimic.g1d import (
    COLUMN_LIMIT,
    COUPLED,
    LIFT_JOINTS,
    OBS_SCALES,
    STATE_JOINTS,
    TRACKED_BODIES,
    UPPER_JOINTS,
)
from supermimic.kinematics import G1DKinematics, G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, NUM_J, OBS_DIM, STATE_DIM, XI_DIM
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import g1d_action_scale, g1d_default_q

XML = os.path.join(REPO, "data", "g1d_description", "g1d.xml")


def main():
    assert (STATE_DIM, OBS_DIM, NUM_J, XI_DIM, INV_DIM, CRITIC_EXTRA) == (71, 99, 29, 33, 71, 113)
    g1 = MDP(G1Kinematics("cpu"), None, torch.zeros(29))
    assert (g1.state_dim, g1.obs_dim, g1.nj, g1.xi_dim, g1.inv_dim, g1.goal_frame, g1.action_dim, g1.critic_extra, g1.n_tracked) == (
        71, 99, 29, 33, 71, 73, 29, 113, 14,
    )
    kin = G1DKinematics("cpu", XML, STATE_JOINTS, TRACKED_BODIES, COUPLED, COLUMN_LIMIT)
    mdp = MDP(kin, None, torch.tensor(g1d_default_q()), obs_kind="g1d")
    assert (mdp.state_dim, mdp.obs_dim, mdp.nj, mdp.xi_dim, mdp.inv_dim, mdp.goal_frame, mdp.action_dim, mdp.critic_extra, mdp.n_tracked) == (
        47, 65, 17, 21, 47, 49, 19, 71, 8,
    )
    assert list(kin.body_names[:3]) == ["AGV_link", "LZ_mt_Link", "LZ_it_Link"]
    assert abs(float(kin.q_hi[0]) - 0.42) < 1e-6 and abs(float(kin.q_lo[0])) < 1e-8

    def pose(height, waist):
        q = torch.tensor(g1d_default_q()).view(1, -1).clone()
        q[0, 0] = height
        q[0, 1] = waist
        p = torch.zeros(1, 3)
        R = torch.eye(3).view(1, 3, 3)
        v = torch.zeros(1, 3)
        w = torch.zeros(1, 3)
        qd = torch.zeros(1, 17)
        return kin.forward(p, R, v, w, q, qd, all_bodies=True), q

    (bp0, *_), _ = pose(0.0, 0.0)
    (bp1, *_), _ = pose(0.2, 0.0)
    torso = kin.body_names.index("torso_link")
    assert abs(float(bp1[0, torso, 2] - bp0[0, torso, 2]) - 0.2) < 1e-4, (bp0[0, torso], bp1[0, torso])
    assert abs(float(bp1[0, 0, 2] - bp0[0, 0, 2])) < 1e-6

    mj = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(mj)
    names = [mj.joint(i).name for i in range(mj.njnt)]
    assert names[3:5] == list(LIFT_JOINTS)
    assert names[5:] == UPPER_JOINTS
    q = torch.zeros(1, 17)
    q[0, 0] = 0.2
    q[0, 1] = 0.4
    data.qpos[3] = data.qpos[4] = 0.1
    data.qpos[5] = 0.4
    mujoco.mj_kinematics(mj, data)
    agv = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "AGV_link")
    p = torch.tensor(data.xpos[agv], dtype=torch.float32).view(1, 3)
    R = torch.tensor(data.xmat[agv], dtype=torch.float32).view(1, 3, 3)
    bp = kin.forward(p, R, torch.zeros(1, 3), torch.zeros(1, 3), q, torch.zeros(1, 17), all_bodies=True)[0][0]
    for name in ("LZ_it_Link", "Yaw_Link", "torso_link", "left_wrist_yaw_link"):
        bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, name)
        err = np.linalg.norm(bp[kin.body_names.index(name)].numpy() - data.xpos[bid])
        assert err < 1e-4, (name, err)

    xi = torch.zeros(1, 21)
    xi[0, 1] = 0.01
    a = torch.zeros(1, 19)
    S = (p, R, torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 0.3]]), q, torch.zeros(1, 17))
    with tempfile.TemporaryDirectory() as d:
        # Reference AGV turned 90 deg left of the robot, so rel is that yaw in 6D.
        dof = np.zeros((8, 18), np.float32)
        yaw = float(torch.atan2(R[0, 1, 0], R[0, 0, 0])) + np.pi / 2
        rot = np.tile([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)], (8, 1)).astype(np.float32)
        joblib.dump({"fps": 50.0, "root_pos": np.tile(p.numpy(), (8, 1)), "root_rot": rot, "dof_pos": dof}, os.path.join(d, "clip.pkl"))
        mdp.motion = MotionLib.from_g1d_dir(d, kin, "cpu")
    o = mdp.obs(S, a, xi, torch.zeros(1, dtype=torch.long))
    assert o.shape == (1, 65)
    assert torch.allclose(o[0, 59:], torch.tensor([0.0, -1.0, 1.0, 0.0, 0.0, 0.0]), atol=1e-5), o[0, 59:]
    assert not bool(mdp.termination(S, torch.zeros(1, dtype=torch.long)).any())
    assert float(mdp.soft_termination(S, torch.zeros(1, dtype=torch.long)).abs().max()) == 0.0
    assert abs(float(o[0, 32]) - 0.19) < 1e-6  # height minus the column offset
    assert abs(float(o[0, 16 + 16 + 1 + 1 + 2]) - 0.3) < 1e-6  # body wz
    assert torch.allclose(o[0, 16 + 16 + 2 + 3 : 16 + 16 + 2 + 6], torch.tensor([0.0, 0.0, -1.0]))
    assert OBS_SCALES[33] == 1.0

    alpha = g1d_action_scale()
    assert len(alpha) == 19 and alpha[16:] == [1.5, 0.6, 0.0765]
    assert alpha[0] == 0.5  # waist pitch is set by hand, not the 5020 formula
    flat = mdp.unflatten(torch.cat((p, torch.tensor([[1.0, 0, 0, 0]]), torch.zeros(1, 6), q, torch.zeros(1, 17)), -1))
    assert flat[4].shape[-1] == 17

    with tempfile.TemporaryDirectory() as d:
        dof = np.zeros((8, 18), np.float32)
        dof[:, 0] = 0.05
        dof[:, 1] = 0.07
        dof[:, 2] = 0.2
        joblib.dump({"fps": 50.0, "root_pos": np.zeros((8, 3), np.float32), "root_rot": np.tile([0, 0, 0, 1], (8, 1)).astype(np.float32), "dof_pos": dof}, os.path.join(d, "clip.pkl"))
        lib = MotionLib.from_g1d_dir(d, kin, "cpu")
        assert lib.q.shape[1] == 17 and abs(float(lib.q[0, 0]) - 0.12) < 1e-5
        assert abs(float(lib.q[0, 1]) - 0.2) < 1e-5
        g = lib.goal(torch.zeros(1, dtype=torch.long), 10)
        assert g.shape == (1, 490)

    wm = WorldModel(num_j=17, inv_dim=47, xi_dim=21, action_dim=19)
    pi = Policy(65, 490, action_dim=19)
    assert wm.in_dim == 47 + 19 + 21 and wm.out_dim == 46
    assert pi.dec[-1].out_features == 19
    buf = ReplayBuffer(4, 2, "cpu", 47, 19, 21)
    assert buf.data["s"].shape[-1] == 47 and buf.data["a"].shape[-1] == 19
    print("g1d spec ok")


if __name__ == "__main__":
    main()
