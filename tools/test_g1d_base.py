"""Standalone MuJoCo sketch of the g1d velocity servo. Not the training environment.

The single-env test that spawns the real G1D meshes inside the SuperMimic trainer
(``make_sim`` + Newton / MuJoCo Warp, same step as ``TrackingSim.step``) is
``tools/test_g1d_env.py``.

This file steps MuJoCo 3.15 with primitive boxes and the same timing, and applies:

    f = kv * (qdot_target - qdot)

``qdot_target`` for ``base_x`` / ``base_y`` / ``base_yaw`` is the world-frame joint velocity
obtained from the body-frame stick command. The two lift stages are constrained to the same
position and each receives half of the column velocity.

Stick map from the real G1-D demo. Stick axes are in [-1, 1]:
  left stick y   -> body vx,  full scale ±1.5 m/s, positive y is forward
  left stick x   -> body wz,  full scale ±0.6 rad/s, positive x is CCW (SDK positive vyaw)
  right stick y  -> column,   full scale ±76.5 mm/s, positive y is up

``TAU`` is the placeholder time constant, not a measurement. kv is set from the simulated
mass matrix as kv = M / TAU so a step response can be compared with 1 - exp(-t/TAU).
"""

from __future__ import annotations

import glob
import os
import struct
import sys
import time

import mujoco
import numpy as np

# Same numbers as supermimic.env.make_sim and supermimic.motion.CONTROL_DT.
CONTROL_DT = 0.02
SUBSTEPS = 4
TIMESTEP = CONTROL_DT / SUBSTEPS

VX_MAX = 1.5
WZ_MAX = 0.6
COLUMN_V_MAX = 0.0765
LIFT_RANGE = 0.21
TAU = 0.15

# Masses taken from ScaleRetarget/assets/unitree_g1d/g1d_mocap.xml. The upper body is one
# rigid lump (arms are not articulated in this check). 20.723 kg is everything above LZ_it.
MASS_AGV = 37.0
MASS_WHEEL = 1.5
MASS_LZ_OT = 3.3
MASS_LZ_MT = 3.3
MASS_LZ_IT = 3.4
MASS_UPPER = 20.723


def build_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec()
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.gravity = [0.0, 0.0, -9.81]

    def inertial(body, mass: float, inertia: list[float], ipos=None):
        body.mass = mass
        body.inertia = inertia
        body.explicitinertial = True
        if ipos is not None:
            body.ipos = ipos

    def box(body, size, pos, rgba):
        # density 0: explicit inertials above already own the mass.
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=rgba,
            contype=0, conaffinity=0, density=0,
        )

    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.85, 0.85, 0.82, 1],
        contype=0, conaffinity=0,
    )
    agv = spec.worldbody.add_body(name="AGV_link", pos=[0.0, 0.0, 0.1113])
    agv.add_joint(name="base_x_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1.0, 0.0, 0.0])
    agv.add_joint(name="base_y_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, 1.0, 0.0])
    agv.add_joint(name="base_yaw_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 0.0, 1.0])
    inertial(agv, MASS_AGV, [0.0810683, 0.0911694, 0.158051])
    box(agv, [0.27, 0.26, 0.08], [0, 0, 0], [0.25, 0.27, 0.30, 1])
    box(agv, [0.08, 0.02, 0.02], [0.35, 0, 0.06], [0.75, 0.15, 0.12, 1])
    for name, y in (("Left_Wheel_Link", 0.2031), ("RIght_Wheel_Link", -0.2031)):
        wheel = agv.add_body(name=name, pos=[0.0, y, -0.0265])
        inertial(wheel, MASS_WHEEL, [0.00154, 0.00154, 0.00276])
        wheel.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.09, 0.02], rgba=[0.1, 0.1, 0.1, 1],
            quat=[0.707, 0.707, 0, 0], contype=0, conaffinity=0, density=0,
        )
    outer = agv.add_body(name="LZ_ot_Link", pos=[0.0, 0.0, 0.5275])
    inertial(outer, MASS_LZ_OT, [0.0149, 0.0718, 0.0718])
    box(outer, [0.06, 0.06, 0.22], [0, 0, -0.16], [0.55, 0.56, 0.58, 1])

    mid = outer.add_body(name="LZ_mt_Link", pos=[0.0, 0.0, 0.01])
    mid.add_joint(
        name="LZ_mt_Joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, 0.0, 1.0],
        range=[0.0, LIFT_RANGE], limited=True,
    )
    inertial(mid, MASS_LZ_MT, [0.0574, 0.0574, 0.0128], ipos=[0.0, 0.0, 0.216])
    box(mid, [0.05, 0.05, 0.12], [0, 0, 0.12], [0.25, 0.45, 0.75, 1])

    inner = mid.add_body(name="LZ_it_Link", pos=[0.0, 0.0, 0.018])
    inner.add_joint(
        name="LZ_it_Joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, 0.0, 1.0],
        range=[0.0, LIFT_RANGE], limited=True,
    )
    inertial(inner, MASS_LZ_IT, [0.0106, 0.0682, 0.0682])
    box(inner, [0.045, 0.045, 0.10], [0, 0, 0.08], [0.20, 0.55, 0.40, 1])
    upper = inner.add_body(name="upper_body", pos=[0.0, 0.0, 0.15])
    inertial(upper, MASS_UPPER, [0.40, 0.35, 0.20], ipos=[0.0, 0.0, 0.15])
    box(upper, [0.12, 0.16, 0.18], [0, 0, 0.20], [0.72, 0.55, 0.35, 1])

    # q_mt = q_it. Both stages extend by the same amount.
    spec.add_equality(
        name="lifts_together", type=mujoco.mjtEq.mjEQ_JOINT,
        name1="LZ_mt_Joint", name2="LZ_it_Joint",
        data=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], active=True,
        solref=[0.002, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    qadr = lambda n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    data.qpos[qadr("LZ_mt_Joint")] = 0.05
    data.qpos[qadr("LZ_it_Joint")] = 0.05
    mujoco.mj_forward(model, data)
    return model, data


def dof(model: mujoco.MjModel, joint: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return int(model.jnt_dofadr[jid])


def reduced_mass(model: mujoco.MjModel, data: mujoco.MjData, dofs: list[int]) -> float:
    mujoco.mj_forward(model, data)
    dense = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, dense)
    e = np.zeros(model.nv)
    for i in dofs:
        e[i] = 1.0
    return float(e @ dense @ e)


def gravity_force(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    saved = data.qvel.copy()
    data.qvel[:] = 0.0
    grav = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, grav)
    data.qvel[:] = saved
    return grav


def body_planar(xd: float, yd: float, psi: float) -> tuple[float, float]:
    c, s = np.cos(psi), np.sin(psi)
    return c * xd + s * yd, -s * xd + c * yd


def stick_to_command(left_x: float, left_y: float, right_y: float, psi: float) -> np.ndarray:
    vx = float(np.clip(left_y, -1.0, 1.0)) * VX_MAX
    wz = float(np.clip(left_x, -1.0, 1.0)) * WZ_MAX
    column = float(np.clip(right_y, -1.0, 1.0)) * COLUMN_V_MAX
    return np.array([vx * np.cos(psi), vx * np.sin(psi), wz, 0.5 * column, 0.5 * column])


def _apply(model, data, ids, kv, target) -> None:
    ix, iy, iyaw, imt, iit = ids
    del ix, iy, iyaw
    grav = gravity_force(model, data)
    force = kv * (target - data.qvel[ids])
    force[3] += grav[imt]
    force[4] += grav[iit]
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[ids] = force
    mujoco.mj_step(model, data)


class LinuxPad:
    """Linux joystick (/dev/input/js*). One event is 8 bytes: time, value, type, number."""

    _EVENT = struct.Struct("IhBB")
    _AXIS = 0x02

    def __init__(self, path: str, layout: str):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes: dict[int, float] = {}
        self.name = _js_name(self.fd)
        self.layout = _resolve_layout(layout, self.name)
        self._drain()

    def close(self) -> None:
        os.close(self.fd)

    def _drain(self) -> None:
        while True:
            try:
                buf = os.read(self.fd, self._EVENT.size * 64)
            except BlockingIOError:
                return
            if not buf:
                return
            for off in range(0, len(buf) - self._EVENT.size + 1, self._EVENT.size):
                _time, value, kind, number = self._EVENT.unpack_from(buf, off)
                # The kernel's initial report on this pad is a full-scale deflection, not the
                # stick at rest. Wait for a real movement before commanding anything.
                if kind & 0x80:
                    continue
                if kind & 0x7F == self._AXIS:
                    self.axes[number] = float(np.clip(value / 32767.0, -1.0, 1.0))

    def sticks(self) -> tuple[float, float, float]:
        self._drain()
        # Stick up is a negative axis on these pads. Commands use up as positive.
        left_x = _deadzone(self.axes.get(0, 0.0))
        left_y = _deadzone(-self.axes.get(1, 0.0))
        right_axis = 4 if self.layout == "x" else 3
        right_y = _deadzone(-self.axes.get(right_axis, 0.0))
        return left_x, left_y, right_y


def _deadzone(value: float, width: float = 0.12) -> float:
    if abs(value) < width:
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def _js_name(fd: int) -> str:
    # JSIOCGNAME(128) = _IOC(_IOC_READ, 'j', 0x13, 128)
    buf = bytearray(128)
    import fcntl

    fcntl.ioctl(fd, 0x80806A13, buf)
    return buf.split(b"\x00", 1)[0].decode(errors="replace") or "unknown"


def _resolve_layout(layout: str, name: str) -> str:
    if layout in ("x", "d"):
        return layout
    lowered = name.lower()
    if "xbox" in lowered or "x-box" in lowered or "xinput" in lowered:
        return "x"
    # Cordless RumblePad 2, and Logitech pads with the back switch on D: right stick Y is axis 3.
    if "rumblepad" in lowered or "cordless" in lowered:
        return "d"
    return "x"


def _find_pad(explicit: str) -> str:
    if explicit:
        return explicit
    found = sorted(glob.glob("/dev/input/js*"))
    if not found:
        raise SystemExit(
            "没有找到 /dev/input/js*。把罗技接收器插上，手柄背后的开关拨到 X，等灯亮了再运行。\n"
            "插上之后如果提示权限不够，执行一次：sudo chmod a+rw /dev/input/js0"
        )
    return found[0]


def run_pad(device: str, layout: str) -> int:
    import mujoco.viewer

    path = _find_pad(device)
    try:
        pad = LinuxPad(path, layout)
    except PermissionError:
        print(f"打不开 {path}：当前用户不在 input 组。")
        print(f"先执行：sudo chmod a+rw {path}")
        print("想以后免去这一步：sudo usermod -aG input $USER，然后重新登录。")
        return 1
    model, data = build_model()
    names = ["base_x_joint", "base_y_joint", "base_yaw_joint", "LZ_mt_Joint", "LZ_it_Joint"]
    ids = [dof(model, n) for n in names]
    ix, iy, iyaw, imt, iit = ids
    m_xy = reduced_mass(model, data, [ix])
    i_yaw = reduced_mass(model, data, [iyaw])
    m_lift = reduced_mass(model, data, [imt, iit])
    kv = np.array([m_xy / TAU, m_xy / TAU, i_yaw / TAU, m_lift / (2.0 * TAU), m_lift / (2.0 * TAU)])
    agv = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "AGV_link")
    right_axis = 4 if pad.layout == "x" else 3
    print(f"pad {path}  {pad.name}  layout {pad.layout}  right stick Y = axis {right_axis}")
    print("左摇杆上下 vx ±1.5 m/s，左右 wz ±0.6 rad/s（向右为逆时针）。右摇杆上下升降柱 ±76.5 mm/s。")
    print("红条是机头。关掉窗口结束。")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -18
            viewer.cam.azimuth = 140
            next_tick = time.perf_counter()
            shown = 0.0
            while viewer.is_running():
                left_x, left_y, right_y = pad.sticks()
                psi = float(data.qpos[iyaw])
                target = stick_to_command(left_x, left_y, right_y, psi)
                for _sub in range(SUBSTEPS):
                    _apply(model, data, ids, kv, target)
                viewer.cam.lookat[:] = data.xpos[agv]
                viewer.sync()
                now = time.perf_counter()
                if now - shown > 0.5:
                    vx, vy = body_planar(float(data.qvel[ix]), float(data.qvel[iy]), psi)
                    height = float(data.qpos[imt] + data.qpos[iit])
                    print(
                        f"stick lx {left_x:+.2f} ly {left_y:+.2f} ry {right_y:+.2f}  "
                        f"vx {vx:+.2f} wz {float(data.qvel[iyaw]):+.2f}  height {height:.3f} m"
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


def main() -> int:
    model, data = build_model()
    assert model.opt.timestep == TIMESTEP
    assert int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    names = ["base_x_joint", "base_y_joint", "base_yaw_joint", "LZ_mt_Joint", "LZ_it_Joint"]
    ids = [dof(model, n) for n in names]
    ix, iy, iyaw, imt, iit = ids
    # One reduced coordinate z = q_mt = q_it. Two actuators share Mz, so each kv is half.
    m_xy = reduced_mass(model, data, [ix])
    i_yaw = reduced_mass(model, data, [iyaw])
    m_lift = reduced_mass(model, data, [imt, iit])
    kv = np.array([m_xy / TAU, m_xy / TAU, i_yaw / TAU, m_lift / (2.0 * TAU), m_lift / (2.0 * TAU)])

    print(f"solver: MuJoCo {mujoco.__version__}  timestep {TIMESTEP} s  substeps {SUBSTEPS}  integrator implicitfast")
    # Lift inertia is for the shared joint rate zdot, not the column mass. With q_mt = q_it,
    # the inner stage's world speed is 2*zdot, so Mz = m_mt + 4*(m_it + m_upper).
    print(f"mass xy {m_xy:.3f} kg   yaw inertia {i_yaw:.4f} kg m^2   lift reduced mass {m_lift:.3f} kg")
    print(f"kv N/(m/s) or N*m/(rad/s): {np.array2string(kv, precision=2)}")

    # (duration s, left_x, left_y, right_y, check name)
    phases = [
        (1.0, 0.0, 0.0, 0.0, "hold"),
        (1.2, 0.0, 0.5, 0.0, "forward"),
        (2.8, 0.5, 0.0, 0.0, "spin"),
        (1.2, 0.0, 0.4, 0.0, "forward_turned"),
        (1.5, 0.0, 0.0, 0.5, "lift"),
        (1.0, 0.0, 0.0, 0.0, "hold_again"),
    ]
    max_lift_gap = 0.0
    samples = {name: [] for *_, name in phases}
    t = 0.0
    for duration, left_x, left_y, right_y, name in phases:
        t_phase = 0.0
        n_control = int(round(duration / CONTROL_DT))
        for _ in range(n_control):
            psi = float(data.qpos[iyaw])
            target = stick_to_command(left_x, left_y, right_y, psi)
            for _sub in range(SUBSTEPS):
                _apply(model, data, ids, kv, target)
            t += CONTROL_DT
            t_phase += CONTROL_DT
            gap = abs(float(data.qpos[imt] - data.qpos[iit]))
            max_lift_gap = max(max_lift_gap, gap)
            vx, vy = body_planar(float(data.qvel[ix]), float(data.qvel[iy]), float(data.qpos[iyaw]))
            samples[name].append((t_phase, vx, vy, float(data.qvel[iyaw]), float(data.qvel[imt] + data.qvel[iit]), float(data.qpos[imt] + data.qpos[iit]), float(data.qpos[iyaw])))

    failures = []

    def check(ok: bool, msg: str):
        print(("ok  " if ok else "FAIL") + "  " + msg)
        if not ok:
            failures.append(msg)

    hold = samples["hold"]
    z0 = hold[0][5]
    check(abs(hold[-1][5] - z0) < 1e-3, f"vz=0 holds column height, drift {hold[-1][5] - z0:+.4f} m")
    check(max_lift_gap < 1e-3, f"two stages stay together, max |q_mt-q_it| {max_lift_gap:.3e} m")

    fwd = samples["forward"]
    # t = TAU is the 63% point of a first-order step from rest.
    near = min(fwd, key=lambda s: abs(s[0] - TAU))
    vx_cmd = 0.5 * VX_MAX
    expect = vx_cmd * (1.0 - np.exp(-near[0] / TAU))
    check(abs(near[1] - expect) < 0.03, f"forward at t={near[0]:.2f}s body vx {near[1]:.3f} vs {expect:.3f} m/s")
    check(abs(fwd[-1][1] - vx_cmd) < 0.03 and abs(fwd[-1][2]) < 0.03, f"forward settles vx {fwd[-1][1]:.3f} vy {fwd[-1][2]:.3f}, command {vx_cmd:.3f}")

    spin = samples["spin"]
    wz_cmd = 0.5 * WZ_MAX
    check(abs(spin[-1][3] - wz_cmd) < 0.03 and abs(spin[-1][1]) < 0.05, f"spin settles wz {spin[-1][3]:.3f} (command {wz_cmd:.3f}), body vx {spin[-1][1]:.3f}")

    turned = samples["forward_turned"]
    vx_cmd2 = 0.4 * VX_MAX
    psi = turned[-1][6]
    vx_b, vy_b = turned[-1][1], turned[-1][2]
    # World velocity should lie along the heading, which is what the rotated qdot_target asks for.
    heading = np.array([np.cos(psi), np.sin(psi)])
    world = np.array([
        vx_b * np.cos(psi) - vy_b * np.sin(psi),
        vx_b * np.sin(psi) + vy_b * np.cos(psi),
    ])
    align = float(np.dot(world, heading) / (np.linalg.norm(world) + 1e-9))
    check(abs(vx_b - vx_cmd2) < 0.04 and abs(vy_b) < 0.04, f"after turning, body vx {vx_b:.3f} vy {vy_b:.3f}, command {vx_cmd2:.3f}")
    check(align > 0.99, f"world velocity follows heading, cos {align:.4f}, psi {psi:.2f} rad")

    lift = samples["lift"]
    col_cmd = 0.5 * COLUMN_V_MAX
    near_l = min(lift, key=lambda s: abs(s[0] - TAU))
    expect_l = col_cmd * (1.0 - np.exp(-near_l[0] / TAU))
    check(abs(near_l[4] - expect_l) < 0.005, f"column at t={near_l[0]:.2f}s speed {near_l[4]:.4f} vs {expect_l:.4f} m/s")
    check(abs(lift[-1][4] - col_cmd) < 0.004, f"column settles {lift[-1][4]:.4f} m/s, command {col_cmd:.4f}")
    check(0.0 <= lift[-1][5] / 2.0 <= LIFT_RANGE, f"each stage stays inside 0–{LIFT_RANGE} m, q {lift[-1][5] / 2.0:.3f}")

    again = samples["hold_again"]
    # Releasing the stick does not freeze the height. Speed decays as exp(-t/TAU), and the
    # extra rise during that decay is about v*TAU. again[0] is already one control step in.
    v0 = lift[-1][4]
    coast = v0 * TAU * (np.exp(-CONTROL_DT / TAU) - np.exp(-1.0 / TAU))
    measured = again[-1][5] - again[0][5]
    check(abs(again[-1][4]) < 0.001, f"stick centered brings column speed to {again[-1][4]:.5f} m/s")
    check(abs(measured - coast) < 0.001, f"coast after release {measured:+.4f} m vs first-order {coast:.4f} m")

    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="g1d chassis and lift velocity model")
    parser.add_argument("--pad", action="store_true", help="drive with /dev/input/js* and open a window")
    parser.add_argument("--device", default="", help="joystick path, default the first /dev/input/js*")
    parser.add_argument("--layout", choices=("auto", "x", "d"), default="auto", help="x: right stick Y is axis 4; d: axis 3")
    args = parser.parse_args()
    if args.pad:
        sys.exit(run_pad(args.device, args.layout))
    sys.exit(main())
