"""Ground-truth-aware scripted demonstrations (test tools; see oracle.py)."""
from __future__ import annotations

import math

import numpy as np

from .env import TaskEnv
from .oracle import Oracle


def _pinch_z_for(env: TaskEnv, center_z: float) -> float:
    g = env.common["gripper"]
    return max(env.common["table"]["top_z"] + g["pad_bottom_below_pinch_m"] + 0.003,
               center_z - g["pad_center_above_pinch_m"])


def oracle_pickup(env: TaskEnv, name: str | None = None, lift: float = 0.14, hold_s: float = 1.5) -> Oracle:
    """Top-down grasp of ``name`` (default: the instructed target) and lift it."""
    o = Oracle(env)
    name = name or env.plan["target"]
    obj = next(x for x in env.plan["objects"] if x["name"] == name)
    pos = env.snapshot().bodies[name].pos
    yaw = 0.0 if obj["shape"] == "cylinder" else o.grasp_yaw(obj["yaw"])
    z = _pinch_z_for(env, pos[2])
    closed = env.common["gripper_closed_ctrl"]
    o.pick(pos[:2], z, yaw)
    o.goto([pos[0], pos[1], z + lift], yaw, speed=0.10, grip=closed)
    o.hold(int(hold_s * env.common["control_hz"]), closed)
    return o


def oracle_stack(env: TaskEnv, release_clearance: float = 0.003) -> Oracle:
    """Move the bottom block onto the zone, then stack the others in the requested order."""
    o = Oracle(env)
    order, size = env.plan["order"], env.plan["block_size"]
    top = env.common["table"]["top_z"]
    blocks = {b["name"]: b for b in env.plan["blocks"]}
    for level, name in enumerate(order):
        state = env.snapshot()
        pos = state.bodies[name].pos
        from .mathutil import yaw_from_quat
        yaw = o.grasp_yaw(yaw_from_quat(state.bodies[name].quat))
        target_xy = env.plan["zone"]["center"] if level == 0 else state.bodies[order[level - 1]].pos[:2]
        center_z = top + size / 2 + level * size + release_clearance
        o.pick(pos[:2], _pinch_z_for(env, pos[2]), yaw)
        o.place(target_xy, _pinch_z_for(env, center_z), yaw, hover_z=_pinch_z_for(env, top + size * (level + 2.2)))
    o.hold(25, env.common["gripper_open_ctrl"])
    return o


def oracle_push_t(env: TaskEnv, tol: float = 0.004, max_s: float = 30.0, v: float = 0.04) -> Oracle:
    """Push the T from the back of its bar along its symmetry axis (stem first) with the closed gripper.

    A wide flat face (the 120 mm bar back) pushed through the centre of mass keeps
    the T translating.  Intended for scenarios where the goal lies on that line
    (translation only; yaw already equals the goal yaw).
    """
    from .mathutil import yaw_from_quat
    o = Oracle(env)
    closed = env.common["gripper_closed_ctrl"]
    goal = np.array(env.plan["goal"]["center"])
    body = env.snapshot().bodies["push_t"]
    com, t_yaw = body.pos[:2].copy(), yaw_from_quat(body.quat)
    d = -np.array([math.cos(t_yaw), math.sin(t_yaw)])           # stem direction = push direction
    z = env.common["table"]["top_z"] + 0.0177 + 0.008          # closed pads hang 17.7 mm below the site
    start = com - d * 0.11
    yaw = o.grasp_yaw(math.atan2(d[1], d[0]))                   # closing axis along the push direction
    o.goto([start[0], start[1], 0.16], yaw, grip=closed)
    o.gripper(closed, 40)
    o.goto([start[0], start[1], z], yaw, speed=0.12, grip=closed)
    push = 0.0
    for _ in range(int(max_s * env.common["control_hz"])):
        pos = env.snapshot().bodies["push_t"].pos[:2]
        if np.linalg.norm(goal - pos) < tol:
            break
        push += v / env.common["control_hz"]
        target = start + d * push
        q, err = o.ik([target[0], target[1], z], yaw, o.q_cmd, iters=10)
        o.q_cmd = q
        env.step(q, closed, ticks=1)
    o.hold(25, closed)
    o.goto([*(o.pinch_pos()[:2] - d * 0.05), 0.16], yaw, speed=0.15, grip=closed)    # back off, then up
    return o


def push_t_scenario(config: dict, offset: float = 0.14, direction=(0.962, 0.274)) -> dict:
    """Reset overrides for the one situation ``oracle_push_t`` can solve: the goal lies ``offset``
    metres ahead of the T along the T's own symmetry axis, with the same yaw.  The push direction
    is tangential to the robot base so the pusher start stays inside the arm's reach."""
    goal = np.array(config["push_t"]["goal"]["center_xy"])
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    yaw = math.atan2(-d[1], -d[0])                      # stem (local -x) points along the push direction
    start = goal - d * offset
    return {"goal_yaw": yaw, "t_init": [float(start[0]), float(start[1]), yaw]}
