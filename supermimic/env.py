"""Motion-tracking simulator on Isaac Lab 3 (kit-less, Newton physics with the MuJoCo-Warp solver).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`supermimic.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1).
"""

from __future__ import annotations

import math
import os

import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate, replace

from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .reset_sampling import update_bin_returns
from .robot import DATA_DIR, G1_CFG, G1_JOINT_NAMES, action_scale
from .rotations import mat_to_quat, quat_to_mat, so3_exp


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    # Use the official asset cached by download_data.sh, avoiding OmniClient's S3 download at startup.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(usd_path=f"{DATA_DIR}/ground/default_ground_plane.usda"),
    )
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def make_sim(device: str, args) -> SimulationContext:
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True),
        num_substeps=4,
    )
    return sim_utils.SimulationCfg(dt=CONTROL_DT, device=device, physics=physics)


class TrackingSim:
    """Batched G1 motion-tracking simulator.

    Must be constructed inside ``launch_simulation(sim_cfg, ...)`` with ``sim_cfg`` from :func:`make_sim`.
    """

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True, kind: str = "g1"):
        self.device = sim_cfg.device
        self.kind = kind
        self.sim = SimulationContext(sim_cfg)
        scene_cfg = TrackingSceneCfg(num_envs=num_envs, env_spacing=0.0, replicate_physics=True)
        if kind == "g1d":
            from .robot import g1d_articulation_cfg

            scene_cfg = replace(scene_cfg, robot=g1d_articulation_cfg())
        self.scene = instantiate(scene_cfg)
        self.sim.reset()
        self.robot = self.scene["robot"]
        if kind == "g1":
            assert list(self.robot.joint_names) == G1_JOINT_NAMES, self.robot.joint_names
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos.torch[0].clone()
        if kind == "g1":
            self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        else:
            from .robot import g1d_action_scale

            self.alpha = torch.tensor(g1d_action_scale(), device=dev)
            self._bind_g1d()
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]

        # ---- sim context xi = [friction, joint offsets, torso com offset(3)] ----
        nj = self.mdp.nj
        self.xi = torch.zeros(self.N, self.mdp.xi_dim, device=dev)
        self._setup_friction()
        if eval_mode:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, 1 + nj : 1 + nj + 3] = com
            default_com = self.robot.data.body_com_pose_b.torch[:, self.torso_id, :3].clone()
            self.robot.set_coms_index(coms=(default_com + com).unsqueeze(1), body_ids=[self.torso_id])
        self._apply_friction()
        if kind == "g1d":
            self._prepare_g1d_drives()

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, self.mdp.action_dim, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        if kind == "g1d":
            from .g1d import BASE_CMD_DELAY

            # (vx, wz, vz) commands, oldest first; the plant sees row 0.
            self.base_queue = torch.zeros(BASE_CMD_DELAY + 1, self.N, 3, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ g1d plant
    def _bind_g1d(self):
        from .g1d import BASE_JOINTS, LIFT_JOINTS, UPPER_JOINTS

        names = list(self.robot.joint_names)

        def j(name: str) -> int:
            if name not in names:
                raise RuntimeError(f"USD is missing joint {name}; have {names}")
            return names.index(name)

        self.base_ids = [j(n) for n in BASE_JOINTS]
        self.lift_ids = [j(n) for n in LIFT_JOINTS]
        self.upper_ids = [j(n) for n in UPPER_JOINTS]
        self.agv_id = int(self.robot.find_bodies("AGV_link")[0][0])

    def _prepare_g1d_drives(self):
        """Velocity-drive gains from the chassis test: kv = m/τ, lift split, yaw measured once."""
        import math

        import mujoco

        from .g1d import TAU

        mj = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "g1d_description", "g1d.xml"))
        mass = float(mj.body_mass.sum())

        def subtree(body_name: str) -> float:
            root = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, body_name)
            ids = [root]
            changed = True
            while changed:
                changed = False
                for i in range(mj.nbody):
                    if int(mj.body_parentid[i]) in ids and i not in ids:
                        ids.append(i)
                        changed = True
            return float(mj.body_mass[ids].sum())

        m_mt, m_it = subtree("LZ_mt_Link"), subtree("LZ_it_Link")
        dev = self.device
        n = self.robot.num_joints
        self.ff = torch.zeros(self.N, n, device=dev)
        g = 9.81
        self.ff[:, self.lift_ids[0]] = m_mt * g
        self.ff[:, self.lift_ids[1]] = m_it * g
        self.robot.write_joint_damping_to_sim_index(
            damping=torch.full((self.N, 1), mass / TAU, device=dev), joint_ids=[self.base_ids[0]]
        )
        self.robot.write_joint_damping_to_sim_index(
            damping=torch.full((self.N, 1), mass / TAU, device=dev), joint_ids=[self.base_ids[1]]
        )
        m_stage = (m_mt - m_it) + 4.0 * m_it
        kv = 0.5 * m_stage / TAU
        for i in self.lift_ids:
            self.robot.write_joint_damping_to_sim_index(damping=torch.full((self.N, 1), kv, device=dev), joint_ids=[i])
        iyaw = self.base_ids[2]
        kd0 = 8.0
        self.robot.write_joint_damping_to_sim_index(damping=torch.full((self.N, 1), kd0, device=dev), joint_ids=[iyaw])
        hold = torch.zeros(self.N, n, device=dev)
        self.robot.write_joint_position_to_sim_index(position=hold)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(hold))
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(0.0)
        probe, duration = 0.25, 0.2
        vel = torch.zeros(self.N, n, device=dev)
        vel[:, iyaw] = probe
        cmd = self.robot.actuators.target_command
        for _ in range(int(round(duration / CONTROL_DT))):
            cmd.set_position_index(value=hold, full_data=True)
            cmd.set_velocity_index(value=vel, full_data=True)
            cmd.set_effort_index(value=self.ff, full_data=True)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(CONTROL_DT)
        wz = float(self.robot.data.joint_vel.torch[0, iyaw])
        frac = min(max(wz / probe, 1e-4), 0.999)
        tau_yaw = -duration / math.log(1.0 - frac)
        self.robot.write_joint_damping_to_sim_index(
            damping=torch.full((self.N, 1), kd0 * tau_yaw / TAU, device=dev), joint_ids=[iyaw]
        )

    def _yaw(self, R: torch.Tensor) -> torch.Tensor:
        return torch.atan2(R[..., 1, 0], R[..., 0, 0])

    def _write_g1d(self, env_ids, p, R, v, w, q, qd):
        n = len(env_ids)
        nj = self.robot.num_joints
        dev = self.device
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :2] = self.scene.env_origins[env_ids, :2]
        pose[:, 2] = p[:, 2] + self.scene.env_origins[env_ids, 2]
        pose[:, 6] = 1.0
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        pos = torch.zeros(n, nj, device=dev)
        vel = torch.zeros(n, nj, device=dev)
        yaw = self._yaw(R)
        pos[:, self.base_ids[0]] = p[:, 0]
        pos[:, self.base_ids[1]] = p[:, 1]
        pos[:, self.base_ids[2]] = yaw
        pos[:, self.lift_ids] = (0.5 * q[:, :1])
        pos[:, self.upper_ids] = q[:, 1:]
        vel[:, self.base_ids[0]] = v[:, 0]
        vel[:, self.base_ids[1]] = v[:, 1]
        vel[:, self.base_ids[2]] = w[:, 2]
        vel[:, self.lift_ids] = (0.5 * qd[:, :1])
        vel[:, self.upper_ids] = qd[:, 1:]
        self.robot.write_joint_position_to_sim_index(position=pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=vel, env_ids=env_ids)
        hold = pos.clone()
        cmd = self.robot.actuators.target_command
        cmd.set_position_index(value=hold, env_ids=env_ids)
        cmd.set_velocity_index(value=vel, env_ids=env_ids)

    def _hold_action_g1d(self, env_ids, R, v, w, q, qd):
        """Action whose targets are the reset state: upper joints at q, chassis and column at their reset speeds."""
        bias = self.xi[env_ids, 2 : 2 + 16]
        upper = (q[:, 1:] - self.mdp.default_q[1:] - bias) / self.alpha[:16]
        yaw = self._yaw(R)
        forward = v[:, 0] * torch.cos(yaw) + v[:, 1] * torch.sin(yaw)
        base = torch.stack((forward, w[:, 2], qd[:, 0]), -1) / self.alpha[16:19]
        return torch.cat((upper, base.clamp(-1.0, 1.0)), -1)

    def _command_g1d(self, a: torch.Tensor, xi: torch.Tensor):
        S = self.state()
        R = S[1]
        yaw = self._yaw(R)
        c, s = torch.cos(yaw), torch.sin(yaw)
        # Saturate at VX_MAX / WZ_MAX / VZ_MAX like the real chassis and column; the world model still sees the raw action.
        new = self.alpha[16:19] * a[:, 16:19].clamp(-1.0, 1.0)
        self.base_queue = torch.cat((self.base_queue[1:], new.unsqueeze(0)), 0)
        vx, wz, vz = self.base_queue[0].unbind(-1)
        q = self.robot.data.joint_pos.torch
        vel = torch.zeros_like(q)
        hold = q.clone()
        bias = xi[:, 2 : 2 + 16]
        hold[:, self.upper_ids] = self.mdp.default_q[1:] + bias + self.alpha[:16] * a[:, :16]
        vel[:, self.base_ids[0]] = vx * c
        vel[:, self.base_ids[1]] = vx * s
        vel[:, self.base_ids[2]] = wz
        vel[:, self.lift_ids] = (0.5 * vz).unsqueeze(-1)
        cmd = self.robot.actuators.target_command
        cmd.set_position_index(value=hold, full_data=True)
        cmd.set_velocity_index(value=vel, full_data=True)
        cmd.set_effort_index(value=self.ff, full_data=True)

    # ------------------------------------------------------------------ friction
    def _setup_friction(self):
        mgr = self.sim.backend_manager
        model = mgr.get_model()
        self._mu_view = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        # ground shapes (body == -1): lowest friction of the range, since MuJoCo takes the max of a pair
        shape_body = wp.to_torch(model.shape_body)
        mu_all = wp.to_torch(model.shape_material_mu)
        mu_all[shape_body < 0] = 0.3 if not self.eval_mode else 1.0

    def _apply_friction(self):
        self._mu_view[:] = self.xi[:, :1]
        self.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    # ------------------------------------------------------------------ state
    def state(self):
        d = self.robot.data
        if self.kind == "g1d":
            pose = d.body_link_pose_w.torch[:, self.agv_id].clone()
            vel = d.body_link_vel_w.torch[:, self.agv_id].clone()
        else:
            pose = d.root_link_pose_w.torch.clone()
            vel = d.root_link_vel_w.torch.clone()
        p = pose[:, :3] - self.scene.env_origins
        quat = pose[:, 3:7]  # Isaac Lab 3 uses xyzw
        R = quat_to_mat(torch.cat((quat[:, 3:4], quat[:, :3]), -1))
        q = d.joint_pos.torch.clone()
        qd = d.joint_vel.torch.clone()
        if self.kind == "g1d":
            height = q[:, self.lift_ids].sum(-1, keepdim=True)
            speed = qd[:, self.lift_ids].sum(-1, keepdim=True)
            q = torch.cat((height, q[:, self.upper_ids]), -1)
            qd = torch.cat((speed, qd[:, self.upper_ids]), -1)
        return (p, R, vel[:, :3], vel[:, 3:6], q, qd)

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise and self.kind == "g1d":
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.zeros(n, device=dev)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1
            )
            R = so3_exp(rv) @ R
            # Differential drive: perturb speed along the heading only, never sideways.
            yaw = self._yaw(R)
            dv = torch.empty(n, device=dev).uniform_(-0.5, 0.5)
            v = v + torch.stack((dv * torch.cos(yaw), dv * torch.sin(yaw), torch.zeros(n, device=dev)), -1)
            w = w + torch.stack(
                (torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1
            )
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        elif self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        if self.kind == "g1d":
            self._write_g1d(env_ids, p, R, v, w, q, qd)
        else:
            quat = mat_to_quat(R)
            pose = torch.cat((p + self.scene.env_origins[env_ids], quat[:, 1:4], quat[:, 0:1]), -1)
            self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            self.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1), env_ids=env_ids)
            self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
            self.robot.set_joint_position_target_index(target=q, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            nj = self.mdp.nj
            self.xi[env_ids, 1 : 1 + nj] = torch.empty(n, nj, device=dev).uniform_(-0.01, 0.01)
        if self.kind == "g1d":
            # The previous action is the one that holds the reset state, so the first step's action-rate
            # penalty and a_prev in the observation see a real change of command, not a jump from 0.
            hold = self._hold_action_g1d(env_ids, R, v, w, q, qd)
            self.a_prev[env_ids] = hold
            self.base_queue[:, env_ids] = (self.alpha[16:19] * hold[:, 16:19]).unsqueeze(0)
        self.scene.reset(env_ids)

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward() if hasattr(self.sim, "forward") else None
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        if self.kind == "g1d":
            self._command_g1d(a, xi)
        else:
            target = self.default_q + xi[:, 1:30] + self.alpha * a
            self.robot.set_joint_position_target_index(target=target)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(CONTROL_DT)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= self.mdp.eval_failure(S1, f1) & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            update_bin_returns(self.bin_G, b, G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out
