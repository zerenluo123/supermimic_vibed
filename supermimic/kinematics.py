"""Differentiable forward kinematics of the G1 in torch, built from the compiled MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import G1_JOINT_NAMES, MJCF_PATH, TRACKED_BODIES
from .rotations import axis_angle_to_mat, quat_to_mat


class G1Kinematics:
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES):
        m = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.device = device
        jname_to_idx = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        # moving bodies in tree order (mujoco orders parents before children)
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] == 0:
                continue  # fixed links are merged visually only; none are tracked
            jid = m.body_jntadr[b]
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                while pb not in mj_to_local:  # skip fixed parents
                    raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[pb])
                joint_idx.append(jname_to_idx[m.joint(jid).name])
            axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits (in G1_JOINT_NAMES order)
        lo, hi = np.zeros(29), np.zeros(29)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root (pelvis) position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,29) in G1_JOINT_NAMES order.
        """
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j])
            axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
            wi = W[par] + axis_w * qd[..., j : j + 1]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _fold_offset(model: mujoco.MjModel, body: int, ancestor: int) -> tuple[np.ndarray, np.ndarray]:
    """Body frame of ``body`` expressed in ``ancestor``, folding fixed links in between."""
    chain = []
    b = body
    while b != ancestor:
        chain.append(b)
        b = int(model.body_parentid[b])
    off_p = np.zeros(3)
    off_q = np.array([1.0, 0.0, 0.0, 0.0])
    for b in reversed(chain):
        R = quat_to_mat(torch.tensor(off_q, dtype=torch.float32)).numpy()
        off_p = off_p + R @ model.body_pos[b]
        off_q = _quat_mul(off_q, model.body_quat[b])
    return off_p, off_q


class G1DKinematics:
    """FK whose q is the 17-D trainer state. The two slide joints each take half the column height.

    The root arguments are the AGV_link pose. The three planar joints are that pose, not entries of q.
    """

    def __init__(self, device: str, mjcf: str, state_joints: list[str], tracked: list[str], coupled: dict, column_limit: tuple[float, float], root_body: str = "AGV_link"):
        m = mujoco.MjModel.from_xml_path(mjcf)
        self.device = device
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_body)
        bodies, parent, joint_idx, scales, axes, offs_p, offs_R, slides = [], [], [], [], [], [], [], []
        mj_to_local = {root: 0}
        bodies.append(root_body)
        parent.append(-1)
        joint_idx.append(-1)
        scales.append(1.0)
        axes.append(np.zeros(3))
        offs_p.append(np.zeros(3))
        offs_R.append(np.array([1.0, 0.0, 0.0, 0.0]))
        slides.append(False)
        for b in range(1, m.nbody):
            if b == root or m.body_jntnum[b] == 0:
                continue
            if m.body_jntnum[b] != 1:
                raise RuntimeError(f"{m.body(b).name} has {m.body_jntnum[b]} joints")
            jname = m.joint(m.body_jntadr[b]).name
            if jname in coupled:
                idx, scale = coupled[jname]
            else:
                idx, scale = state_joints.index(jname), 1.0
            ancestor = int(m.body_parentid[b])
            while ancestor not in mj_to_local:
                ancestor = int(m.body_parentid[ancestor])
            off_p, off_q = _fold_offset(m, b, ancestor)
            mj_to_local[b] = len(bodies)
            bodies.append(m.body(b).name)
            parent.append(mj_to_local[ancestor])
            joint_idx.append(idx)
            scales.append(scale)
            axes.append(m.jnt_axis[m.body_jntadr[b]].copy())
            offs_p.append(off_p)
            offs_R.append(off_q)
            slides.append(int(m.jnt_type[m.body_jntadr[b]]) == int(mujoco.mjtJoint.mjJNT_SLIDE))
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.scale = torch.tensor(scales, dtype=torch.float32, device=device)
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.slide = slides
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        lo = np.zeros(len(state_joints))
        hi = np.zeros(len(state_joints))
        lo[0], hi[0] = column_limit
        named = {m.joint(j).name: j for j in range(m.njnt)}
        for i, name in enumerate(state_joints):
            if name in named:
                lo[i], hi[i] = m.jnt_range[named[name]]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            ang = q[..., j] * self.scale[i]
            rate = qd[..., j] * self.scale[i]
            Rp = Rs[par]
            R_off = Rp @ self.off_R[i]
            axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
            if self.slide[i]:
                pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1) + axis_w * ang.unsqueeze(-1)
                Ri = R_off
                wi = W[par]
                vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1) + axis_w * rate.unsqueeze(-1)
            else:
                pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
                Ri = R_off @ axis_angle_to_mat(self.axes[i], ang)
                wi = W[par] + axis_w * rate.unsqueeze(-1)
                vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
