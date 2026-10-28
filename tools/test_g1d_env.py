"""Single-env g1d chassis and lift check inside the SuperMimic trainer.

Same launch and physics as ``supermimic/train.py``: ``make_sim`` (Newton + MuJoCo Warp,
dt 0.02 s, 4 substeps, implicitfast, pyramidal cone, MuJoCo contacts) and the same
write / step / update sequence as ``TrackingSim.step``. The asset is the retarget MJCF
converted from the official G1D URDF (real meshes and inertias), imported to USD the
same way as ``tools/mjcf_to_usd.py``, and spawned with ``Physics=mujoco`` like ``G1_CFG``.

Upper-body joints are implicit position-velocity drives, which is the G1 training actuator.
``base_x``, ``base_y``, ``base_yaw`` and the two lift slides are implicit velocity drives:
the solver force is ``kv * (qdot_target - qdot)``, plus a gravity feedforward on the slides.
The two slides are the MJCF equality (they extend together). A stick command is body-frame;
``base_x`` / ``base_y`` targets are that command rotated into the world frame.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="g1d chassis/lift check in the SuperMimic trainer")
parser.add_argument("--pad", action="store_true", help="read /dev/input/js* and open the Newton viewer")
parser.add_argument("--js", default="", help="joystick device, default the first /dev/input/js*")
parser.add_argument("--layout", choices=("auto", "x", "d"), default="auto")
add_launcher_args(parser)
args = parser.parse_args()
if args.pad and not args.visualizer:
    args.visualizer = ["newton_gl"]

import mujoco
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, instantiate

from supermimic.env import make_sim
from supermimic.motion import CONTROL_DT
from test_g1d_base import LinuxPad, _find_pad

XML = os.path.join(REPO, "data", "g1d_description", "g1d.xml")
USD = os.path.join(REPO, "data", "g1d_usd", "g1d", "g1d.usda")
TAU = 0.15
VX_MAX = 1.5
WZ_MAX = 0.6
COLUMN_V_MAX = 0.0765
G = 9.81

# Three joints share one body pair, so Newton exposes them as one joint's DOFs.
# USD order is base_x, base_y, base_yaw.
BASE_X, BASE_Y, BASE_YAW = "base_x_joint:0", "base_x_joint:1", "base_x_joint:2"
CHASSIS = (BASE_X, BASE_Y, BASE_YAW)
LIFTS = ("LZ_mt_Joint", "LZ_it_Joint")


def subtree_mass(model: mujoco.MjModel, body_name: str) -> float:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    ids = [root]
    changed = True
    while changed:
        changed = False
        for i in range(model.nbody):
            if int(model.body_parentid[i]) in ids and i not in ids:
                ids.append(i)
                changed = True
    return float(model.body_mass[ids].sum())


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD,
            variants={"Physics": "mujoco"},
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"LZ_mt_Joint": 0.05, "LZ_it_Joint": 0.05},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        # ImplicitActuatorCfg is the actuator G1 training uses. Non-zero damping and
        # zero stiffness selects Newton's velocity drive: f = kd * (qd_target - qd).
        actuators={
            "upper": ImplicitActuatorCfg(
                joint_names_expr=["Yaw_Joint", "torso_Joint", "left_.*", "right_.*"],
                stiffness=400.0,
                damping=40.0,
                effort_limit_sim=300.0,
            ),
            "chassis": ImplicitActuatorCfg(
                joint_names_expr=[BASE_X, BASE_Y, BASE_YAW],
                stiffness=0.0,
                damping=1.0,
                effort_limit_sim=1.0e5,
            ),
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["LZ_mt_Joint", "LZ_it_Joint"],
                stiffness=0.0,
                damping=1.0,
                effort_limit_sim=1.0e5,
            ),
        },
    )


def jid(robot, name: str) -> int:
    return list(robot.joint_names).index(name)


def command_and_step(sim, scene, robot, hold, velocity, effort, render: bool) -> None:
    """Same physics calls as TrackingSim.step, plus velocity and effort commands."""
    cmd = robot.actuators.target_command
    cmd.set_position_index(value=hold, full_data=True)
    cmd.set_velocity_index(value=velocity, full_data=True)
    cmd.set_effort_index(value=effort, full_data=True)
    scene.write_data_to_sim()
    sim.step(render=render)
    scene.update(CONTROL_DT)


def write_damping(robot, index: int, value: float) -> None:
    robot.write_joint_damping_to_sim_index(damping=torch.tensor([[value]], device=robot.device), joint_ids=[index])


def reset_pose(sim, scene, robot, hold) -> None:
    zeros = torch.zeros_like(hold)
    robot.write_joint_position_to_sim_index(position=hold)
    robot.write_joint_velocity_to_sim_index(velocity=zeros)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)


def fit_tau(reached: float, target: float, duration: float) -> float:
    frac = min(max(reached / target, 1e-4), 0.999)
    return -duration / math.log(1.0 - frac)


def stick_targets(robot, left_x: float, left_y: float, right_y: float, yaw: float) -> torch.Tensor:
    vx = float(max(-1.0, min(1.0, left_y))) * VX_MAX
    # Stick right is a right turn. SDK positive vyaw is counter-clockwise, so right is negative.
    wz = -float(max(-1.0, min(1.0, left_x))) * WZ_MAX
    column = float(max(-1.0, min(1.0, right_y))) * COLUMN_V_MAX
    out = torch.zeros(1, robot.num_joints, device=robot.device)
    out[0, jid(robot, BASE_X)] = vx * math.cos(yaw)
    out[0, jid(robot, BASE_Y)] = vx * math.sin(yaw)
    out[0, jid(robot, BASE_YAW)] = wz
    half = 0.5 * column
    out[0, jid(robot, "LZ_mt_Joint")] = half
    out[0, jid(robot, "LZ_it_Joint")] = half
    return out


def read_state(robot):
    q = robot.data.joint_pos.torch[0]
    qd = robot.data.joint_vel.torch[0]
    yaw = float(q[jid(robot, BASE_YAW)])
    xd = float(qd[jid(robot, BASE_X)])
    yd = float(qd[jid(robot, BASE_Y)])
    c, s = math.cos(yaw), math.sin(yaw)
    vx, vy = c * xd + s * yd, -s * xd + c * yd
    wz = float(qd[jid(robot, BASE_YAW)])
    z = float(q[jid(robot, "LZ_mt_Joint")] + q[jid(robot, "LZ_it_Joint")])
    zd = float(qd[jid(robot, "LZ_mt_Joint")] + qd[jid(robot, "LZ_it_Joint")])
    gap = abs(float(q[jid(robot, "LZ_mt_Joint")] - q[jid(robot, "LZ_it_Joint")]))
    return vx, vy, wz, z, zd, gap, yaw


def main() -> int:
    if not os.path.isfile(USD):
        raise SystemExit(f"missing {USD}; convert data/g1d_description/g1d.xml with the MJCF importer first")
    mj = mujoco.MjModel.from_xml_path(XML)
    mass = float(mj.body_mass.sum())
    m_mt = subtree_mass(mj, "LZ_mt_Link")
    m_it = subtree_mass(mj, "LZ_it_Link")
    print(f"mjcf mass {mass:.3f} kg  lift subtree mt {m_mt:.3f} kg  it {m_it:.3f} kg  joints {mj.njnt}")

    sim_cfg = make_sim("cuda:0", args)
    with launch_simulation(sim_cfg, args):
        sim = SimulationContext(sim_cfg)
        scene = instantiate(SceneCfg(num_envs=1, env_spacing=0.0, replicate_physics=True))
        sim.reset()
        robot = scene["robot"]
        print("JOINTS", list(robot.joint_names))
        print("BODIES", list(robot.body_names))
        newton_mass = float(robot.data.body_mass.torch[0].sum())
        print(f"newton mass {newton_mass:.3f} kg  root z {float(robot.data.root_link_pos_w.torch[0, 2]):.3f}")
        for name in (*CHASSIS, *LIFTS):
            if name not in robot.joint_names:
                raise SystemExit(f"USD is missing joint {name}")

        dev = robot.device
        hold = robot.data.joint_pos.torch.clone()
        # Velocity drives ignore the position target, so a reset step drops the column
        # onto the lower stop before this pose is captured. Put it back off the stop.
        hold[0, jid(robot, "LZ_mt_Joint")] = 0.05
        hold[0, jid(robot, "LZ_it_Joint")] = 0.05
        effort = torch.zeros(1, robot.num_joints, device=dev)
        effort[0, jid(robot, "LZ_mt_Joint")] = m_mt * G
        effort[0, jid(robot, "LZ_it_Joint")] = m_it * G
        write_damping(robot, jid(robot, BASE_X), mass / TAU)
        write_damping(robot, jid(robot, BASE_Y), mass / TAU)
        # Stages are locked together, so the inner stage moves at twice the joint rate.
        # KE = 0.5*(m_mt_link + 4*m_it)*qdot^2. Split that inertia across the two drives.
        m_stage = (m_mt - m_it) + 4.0 * m_it
        kv_stage = 0.5 * m_stage / TAU
        write_damping(robot, jid(robot, "LZ_mt_Joint"), kv_stage)
        write_damping(robot, jid(robot, "LZ_it_Joint"), kv_stage)
        print(f"lift reduced mass {m_stage:.3f} kg  kv each {kv_stage:.2f}")

        # Yaw inertia is not a single body. One step of the velocity drive measures it,
        # then the gain is set so the closed-loop time constant is TAU.
        iyaw = jid(robot, BASE_YAW)
        kd0 = 8.0
        write_damping(robot, iyaw, kd0)
        reset_pose(sim, scene, robot, hold)
        probe = 0.25
        duration = 0.2
        vel = torch.zeros(1, robot.num_joints, device=dev)
        vel[0, iyaw] = probe
        for _ in range(int(round(duration / CONTROL_DT))):
            command_and_step(sim, scene, robot, hold, vel, effort, render=False)
        wz = float(robot.data.joint_vel.torch[0, iyaw])
        tau_yaw = fit_tau(wz, probe, duration)
        inertia = kd0 * tau_yaw
        write_damping(robot, iyaw, inertia / TAU)
        print(f"yaw probe wz {wz:.3f}  tau {tau_yaw:.3f}s at kd {kd0:.1f}  I {inertia:.3f}  kd {inertia / TAU:.2f}")
        reset_pose(sim, scene, robot, hold)

        if args.pad:
            return run_pad(sim, scene, robot, hold, effort)
        return run_checks(sim, scene, robot, hold, effort)


def run_checks(sim, scene, robot, hold, effort) -> int:
    phases = [
        (1.0, 0.0, 0.0, 0.0, "hold"),
        (1.2, 0.0, 0.5, 0.0, "forward"),
        (2.8, 0.5, 0.0, 0.0, "spin"),
        (1.2, 0.0, 0.4, 0.0, "forward_turned"),
        (1.5, 0.0, 0.0, 0.5, "lift"),
        (1.0, 0.0, 0.0, 0.0, "hold_again"),
    ]
    samples = {name: [] for *_, name in phases}
    max_gap = 0.0
    for duration, left_x, left_y, right_y, name in phases:
        t_phase = 0.0
        for _ in range(int(round(duration / CONTROL_DT))):
            vx, vy, wz, z, zd, gap, yaw = read_state(robot)
            # command from the yaw at the start of the control step, same as a 50 Hz policy
            vel = stick_targets(robot, left_x, left_y, right_y, yaw)
            command_and_step(sim, scene, robot, hold, vel, effort, render=False)
            t_phase += CONTROL_DT
            vx, vy, wz, z, zd, gap, yaw = read_state(robot)
            max_gap = max(max_gap, gap)
            samples[name].append((t_phase, vx, vy, wz, zd, z, yaw))

    failures = []

    def check(ok: bool, msg: str) -> None:
        print(("ok  " if ok else "FAIL") + "  " + msg)
        if not ok:
            failures.append(msg)

    z0 = samples["hold"][0][5]
    check(
        abs(z0 - 0.10) < 0.01 and abs(samples["hold"][-1][5] - z0) < 2e-3,
        f"vz=0 holds column at {z0:.3f} m, drift {samples['hold'][-1][5] - z0:+.4f} m",
    )
    check(max_gap < 2e-3, f"stages stay together, max |q_mt-q_it| {max_gap:.3e} m")

    fwd = samples["forward"]
    near = min(fwd, key=lambda s: abs(s[0] - TAU))
    vx_cmd = 0.5 * VX_MAX
    expect = vx_cmd * (1.0 - math.exp(-near[0] / TAU))
    check(abs(near[1] - expect) < 0.05, f"forward at t={near[0]:.2f}s body vx {near[1]:.3f} vs {expect:.3f}")
    check(abs(fwd[-1][1] - vx_cmd) < 0.05 and abs(fwd[-1][2]) < 0.05, f"forward settles vx {fwd[-1][1]:.3f} vy {fwd[-1][2]:.3f}")

    spin = samples["spin"]
    wz_cmd = -0.5 * WZ_MAX
    check(abs(spin[-1][3] - wz_cmd) < 0.05, f"spin settles wz {spin[-1][3]:.3f}, command {wz_cmd:.3f}")

    turned = samples["forward_turned"]
    vx_cmd2 = 0.4 * VX_MAX
    vx_b, vy_b, yaw = turned[-1][1], turned[-1][2], turned[-1][6]
    world = (vx_b * math.cos(yaw) - vy_b * math.sin(yaw), vx_b * math.sin(yaw) + vy_b * math.cos(yaw))
    heading = (math.cos(yaw), math.sin(yaw))
    norm = math.hypot(*world) + 1e-9
    align = (world[0] * heading[0] + world[1] * heading[1]) / norm
    check(abs(vx_b - vx_cmd2) < 0.06 and abs(vy_b) < 0.06, f"after turning, body vx {vx_b:.3f} vy {vy_b:.3f}")
    check(align > 0.98, f"world velocity follows heading, cos {align:.4f}, yaw {yaw:.2f}")

    lift = samples["lift"]
    col_cmd = 0.5 * COLUMN_V_MAX
    near_l = min(lift, key=lambda s: abs(s[0] - TAU))
    expect_l = col_cmd * (1.0 - math.exp(-near_l[0] / TAU))
    check(abs(near_l[4] - expect_l) < 0.004, f"column at t={near_l[0]:.2f}s speed {near_l[4]:.4f} vs {expect_l:.4f}")
    check(abs(lift[-1][4] - col_cmd) < 0.008, f"column settles {lift[-1][4]:.4f}, command {col_cmd:.4f}")

    again = samples["hold_again"]
    check(abs(again[-1][4]) < 0.005, f"stick centered, column speed {again[-1][4]:.4f} m/s")
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


def run_pad(sim, scene, robot, hold, effort) -> int:
    import time

    pad = LinuxPad(_find_pad(args.js), args.layout)
    print(f"pad {pad.name} layout {pad.layout}")
    print("左摇杆上下 vx，左右 wz。右摇杆上下升降柱。关掉 Newton 窗口结束。")
    shown = 0.0
    next_tick = time.perf_counter()
    try:
        while True:
            left_x, left_y, right_y = pad.sticks()
            _vx, _vy, _wz, _z, _zd, _gap, yaw = read_state(robot)
            vel = stick_targets(robot, left_x, left_y, right_y, yaw)
            command_and_step(sim, scene, robot, hold, vel, effort, render=True)
            now = time.perf_counter()
            if now - shown > 0.5:
                vx, vy, wz, z, zd, gap, yaw = read_state(robot)
                print(
                    f"stick lx {left_x:+.2f} ly {left_y:+.2f} ry {right_y:+.2f}  "
                    f"vx {vx:+.2f} (cmd {left_y * VX_MAX:+.2f})  "
                    f"wz {wz:+.2f} (cmd {-left_x * WZ_MAX:+.2f})  "
                    f"lift {zd * 1000:+.1f} mm/s (cmd {right_y * COLUMN_V_MAX * 1000:+.1f})  "
                    f"height {z:.3f}"
                )
                shown = now
            next_tick += CONTROL_DT
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    finally:
        pad.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
