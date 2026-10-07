"""Scripted ground-truth-aware driver, used ONLY to validate scenes, physics and evaluators.

It reads simulator ground truth (object poses) to plan top-down grasps, so it is a
test tool, not a policy and not part of any ReKep evaluation.  Pose -> joint
targets use a damped-least-squares IK on the Robotiq ``pinch`` site; the joint
targets are sent to the same position actuators every other controller uses.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .env import TaskEnv
from .mathutil import rot_z, wrap_angle


class IKFailure(RuntimeError):
    pass


class Oracle:
    def __init__(self, env: TaskEnv, speed: float = 0.25, yaw_rate: float = 1.0):
        self.env, self.speed, self.yaw_rate = env, speed, yaw_rate
        m = env.model
        self.scratch = mujoco.MjData(m)
        self.lo = np.array([m.jnt_range[j][0] for j in env.arm_joint_ids])
        self.hi = np.array([m.jnt_range[j][1] for j in env.arm_joint_ids])
        mujoco.mj_forward(m, env.data)
        self.R_home = env.data.site_xmat[env.pinch_site].reshape(3, 3).copy()
        left = env.data.geom_xpos[m.geom("left_pad1").id]
        right = env.data.geom_xpos[m.geom("right_pad1").id]
        axis = (right - left)[:2]
        self.closing_phi_home = math.atan2(axis[1], axis[0])
        self.q_cmd = env.arm_qpos()
        self.yaw = 0.0
        self.hz = env.common["control_hz"]

    # -- kinematics -------------------------------------------------------------
    def pinch_pos(self) -> np.ndarray:
        return self.env.data.site_xpos[self.env.pinch_site].copy()

    def ik(self, pos, yaw: float, q0, iters: int = 30, damping: float = 0.05) -> tuple[np.ndarray, float]:
        env, m, d = self.env, self.env.model, self.scratch
        target_R = rot_z(yaw) @ self.R_home
        q = np.array(q0, float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err_norm = math.inf
        for _ in range(iters):
            for adr, v in zip(env.arm_qpos_adr, q):
                d.qpos[adr] = v
            mujoco.mj_fwdPosition(m, d)
            R = d.site_xmat[env.pinch_site].reshape(3, 3)
            p_err = np.asarray(pos) - d.site_xpos[env.pinch_site]
            r_err = 0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))
            err = np.concatenate([p_err, r_err])
            err_norm = max(np.linalg.norm(p_err), np.linalg.norm(r_err) * 0.1)
            if np.linalg.norm(p_err) < 5e-4 and np.linalg.norm(r_err) < 5e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, env.pinch_site)
            J = np.vstack([jacp[:, env.arm_dof_adr], jacr[:, env.arm_dof_adr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
            norm = np.linalg.norm(dq)
            if norm > 0.3:
                dq *= 0.3 / norm
            q = np.clip(q + dq, self.lo, self.hi)
        return q, err_norm

    def reachable(self, pos, yaw: float) -> bool:
        q, err = self.ik(pos, yaw, self.env.arm_qpos(), iters=80)
        return err < 1e-3

    # -- motion ------------------------------------------------------------------
    def goto(self, pos, yaw: float | None = None, speed: float | None = None, grip: float | None = None) -> None:
        env = self.env
        yaw = self.yaw if yaw is None else yaw
        start, p1 = self.pinch_pos(), np.asarray(pos, float)
        speed = speed or self.speed
        steps = max(2, int(math.ceil(max(np.linalg.norm(p1 - start) / speed,
                                         abs(yaw - self.yaw) / self.yaw_rate) * self.hz)))
        y0 = self.yaw
        for i in range(1, steps + 1):
            a = i / steps
            q, err = self.ik(start + a * (p1 - start), y0 + a * (yaw - y0), self.q_cmd, iters=10)
            if err > 5e-3:
                raise IKFailure(f"IK failed at step {i}/{steps}: err {err:.4f} for {p1} yaw {yaw:.2f}")
            self.q_cmd = q
            env.step(q, grip, ticks=1)
        self.yaw = yaw
        self.hold(10, grip)

    def hold(self, ticks: int, grip: float | None = None) -> None:
        self.env.step(self.q_cmd, grip, ticks=ticks)

    def gripper(self, value: float, ticks: int = 60) -> None:
        self.env.step(self.q_cmd, value, ticks=ticks)

    # -- primitives ----------------------------------------------------------------
    def grasp_yaw(self, object_yaw: float) -> float:
        """Gripper yaw whose closing axis is along the object's local x axis (mod 180 deg)."""
        y = wrap_angle(object_yaw - self.closing_phi_home)
        while y > math.pi / 2:
            y -= math.pi
        while y < -math.pi / 2:
            y += math.pi
        return y

    def pick(self, xy, z: float, yaw: float, hover_z: float = 0.22) -> None:
        open_, closed = self.env.common["gripper_open_ctrl"], self.env.common["gripper_closed_ctrl"]
        self.goto([xy[0], xy[1], hover_z], yaw, grip=open_)
        self.goto([xy[0], xy[1], z], yaw, speed=0.12, grip=open_)
        self.gripper(closed, ticks=60)

    def place(self, xy, z: float, yaw: float, hover_z: float = 0.30, release: bool = True) -> None:
        open_, closed = self.env.common["gripper_open_ctrl"], self.env.common["gripper_closed_ctrl"]
        self.goto([xy[0], xy[1], hover_z], yaw, grip=closed)
        self.goto([xy[0], xy[1], z], yaw, speed=0.06, grip=closed)
        self.hold(20, closed)
        if release:
            self.gripper(open_, ticks=50)
            self.goto([xy[0], xy[1], hover_z], yaw, speed=0.15, grip=open_)
