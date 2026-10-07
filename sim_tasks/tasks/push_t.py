"""Push T: push a T-shaped block (no lifting) onto a green T-shaped goal outline."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..env import TaskEnv
from ..evaluator import BaseEvaluator, FailureReason, probe_stats, result, upright_error, yaw_error
from ..mathutil import quat_from_yaw, rot_z, yaw_from_quat, wrap_angle
from ..scene import free_body, geom
from ..state import SceneState

GOAL_RGBA = (0.10, 0.85, 0.20, 1.0)
T_RGBA = (0.15, 0.30, 0.90, 1.0)


def t_layout(cfg: dict) -> dict[str, float]:
    """Local T geometry.  Body frame origin = centre of mass; bar along local y at +x, stem toward -x."""
    bx, by = cfg["bar_half"]
    sx, sy = cfg["stem_half"]
    area_bar, area_stem = 4 * bx * by, 4 * sx * sy
    offset = bx + sx                                   # bar centre -> stem centre
    bar_x = offset * area_stem / (area_bar + area_stem)
    return {"bar_x": bar_x, "stem_x": bar_x - offset, "bx": bx, "by": by, "sx": sx, "sy": sy,
            "mass_bar": cfg["mass_kg"] * area_bar / (area_bar + area_stem),
            "mass_stem": cfg["mass_kg"] * area_stem / (area_bar + area_stem)}


def t_mask(points_xy: np.ndarray, center_xy, yaw: float, layout: dict) -> np.ndarray:
    """Boolean mask: which world xy points lie inside the T placed at (center, yaw)."""
    local = (points_xy - np.asarray(center_xy)) @ rot_z(yaw)[:2, :2]      # world -> local
    lx, ly = local[:, 0], local[:, 1]
    bar = (np.abs(lx - layout["bar_x"]) <= layout["bx"]) & (np.abs(ly) <= layout["by"])
    stem = (np.abs(lx - layout["stem_x"]) <= layout["sx"]) & (np.abs(ly) <= layout["sy"])
    return bar | stem


def coverage(layout: dict, pose_a, pose_b, grid: float) -> tuple[float, float]:
    """(intersection / area of b, IoU) of two T placements; pose = (x, y, yaw)."""
    reach = 0.12
    cx, cy = (pose_a[0] + pose_b[0]) / 2, (pose_a[1] + pose_b[1]) / 2
    span = reach + max(abs(pose_a[0] - pose_b[0]), abs(pose_a[1] - pose_b[1]))
    axis_x, axis_y = np.arange(cx - span, cx + span, grid), np.arange(cy - span, cy + span, grid)
    pts = np.stack(np.meshgrid(axis_x, axis_y), axis=-1).reshape(-1, 2)
    ma, mb = t_mask(pts, pose_a[:2], pose_a[2], layout), t_mask(pts, pose_b[:2], pose_b[2], layout)
    inter, union = float((ma & mb).sum()), float((ma | mb).sum())
    return inter / max(1.0, float(mb.sum())), inter / max(1.0, union)


class PushTEvaluator(BaseEvaluator):
    """success: xy_error and yaw_error under thresholds, never lifted, T at rest."""

    needs_probe = True

    def reset(self, state: SceneState) -> None:
        self.layout = self.plan["layout"]
        self.goal = (*self.plan["goal"]["center"], self.plan["goal"]["yaw"])
        body = state.bodies["push_t"]
        self.z0 = float(body.pos[2])
        self.start_xy = body.pos[:2].copy()
        self.max_delta_z = 0.0
        self.max_tilt = 0.0
        self.violation_time: float | None = None
        self.touched = False
        self.path_length = 0.0
        self.min_z = self.z0
        self._last_xy = self.start_xy.copy()

    def observe(self, state: SceneState) -> None:
        body = state.bodies["push_t"]
        self.max_delta_z = max(self.max_delta_z, abs(float(body.pos[2]) - self.z0))
        self.max_tilt = max(self.max_tilt, upright_error(body.quat))
        self.min_z = min(self.min_z, float(body.pos[2]))
        self.path_length += float(np.linalg.norm(body.pos[:2] - self._last_xy))
        self._last_xy = body.pos[:2].copy()
        if state.robot_contacts("push_t"):
            self.touched = True
        if self.violation_time is None and (self.max_delta_z > self.cfg["max_delta_z_m"]
                                            or self.max_tilt > self.cfg["max_tilt_rad"]):
            self.violation_time = state.time

    def pose(self, state: SceneState) -> tuple[float, float, float]:
        body = state.bodies["push_t"]
        return float(body.pos[0]), float(body.pos[1]), yaw_from_quat(body.quat)

    def evaluate(self, state: SceneState, probe=None, timed_out: bool = False) -> dict[str, Any]:
        cfg = self.cfg
        pose = self.pose(state)
        xy_error = math.hypot(pose[0] - self.goal[0], pose[1] - self.goal[1])
        yaw_err = yaw_error(pose[2], self.goal[2])
        cover, iou = coverage(self.layout, pose, self.goal, cfg["coverage_grid_m"])
        stats = probe_stats(probe, ["push_t"]) if probe else {"max_linear_speed": None, "max_angular_speed": None,
                                                              "max_pose_drift": None}
        at_rest = probe is not None and (stats["max_linear_speed"] <= cfg["max_linear_speed"]
                                         and stats["max_angular_speed"] <= cfg["max_angular_speed"])
        lifted = self.violation_time is not None
        fell = self.min_z < self.z0 - cfg["fell_below_table_m"]
        pose_ok = xy_error <= cfg["xy_tolerance_m"] and yaw_err <= cfg["yaw_tolerance_rad"]
        success = bool(pose_ok and not lifted and not fell and at_rest)
        metrics = {
            "xy_error": round(xy_error, 4), "yaw_error": round(yaw_err, 4),
            "yaw_error_deg": round(math.degrees(yaw_err), 2),
            "coverage": round(cover, 4), "iou": round(iou, 4),
            "max_delta_z": round(self.max_delta_z, 4), "max_tilt_rad": round(self.max_tilt, 4),
            "lifted": lifted, "fell": fell, "at_rest": bool(at_rest), **stats,
            "pushed_path_length": round(self.path_length, 4), "touched": self.touched,
            "pose": [round(v, 4) for v in pose], "goal_pose": [round(v, 4) for v in self.goal],
        }
        if success:
            score = 1.0
        elif lifted or fell:
            score = 0.0
        else:
            score = min(cfg["partial_score_cap"], cover)
        reason = None
        if not success:
            if lifted:
                reason = FailureReason.PUSH_LIFT_VIOLATION
            elif fell:
                reason = FailureReason.OBJECT_DROPPED
            elif timed_out:
                reason = FailureReason.TIMEOUT
            elif xy_error <= cfg["xy_tolerance_m"] and yaw_err > cfg["yaw_tolerance_rad"]:
                reason = FailureReason.ORIENTATION_ERROR
            else:
                reason = FailureReason.PLACEMENT_ERROR
        partial = (not success) and not lifted and not fell and cover >= cfg["coverage_partial"]
        return result(success, score, metrics, reason, partial)


class PushTEnv(TaskEnv):
    TASK = "push_t"

    def plan_scene(self, rng, overrides):
        cfg, common = self.cfg, self.common
        overrides = overrides or {}
        layout = t_layout(cfg)
        goal = {"center": list(overrides.get("goal_center", cfg["goal"]["center_xy"])),
                "yaw": float(overrides.get("goal_yaw", cfg["goal"]["yaw_rad"]))}
        init = cfg["init"]
        for _ in range(10000):
            x, y = float(rng.uniform(*init["x"])), float(rng.uniform(*init["y"]))
            if not init["radius"][0] <= math.hypot(x, y) <= init["radius"][1]:
                continue
            if not init["goal_distance_m"][0] <= math.hypot(x - goal["center"][0], y - goal["center"][1]) <= init["goal_distance_m"][1]:
                continue
            offset = float(rng.uniform(*init["yaw_offset_rad"])) * (1.0 if rng.random() < 0.5 else -1.0)
            yaw = wrap_angle(goal["yaw"] + offset)
            break
        else:
            raise RuntimeError("could not sample a T start pose")
        if "t_init" in overrides:                           # explicit scenario (tests / curricula)
            x, y, yaw = overrides["t_init"]
        t = {"bar_half": list(cfg["bar_half"]), "stem_half": list(cfg["stem_half"]), "height": cfg["height_m"],
             "mass": cfg["mass_kg"], "friction": list(cfg["friction"]), "rgba": list(T_RGBA)}
        return {"t": t, "layout": layout, "goal": goal,
                "init_pose": [x, y, float(yaw)],
                "pos": [x, y, common["table"]["top_z"] + cfg["height_m"] / 2 + 1e-4]}

    def fill_scene(self, world, plan):
        cfg, col = self.cfg, self.common["collision"]["object"]
        layout, h, top = plan["layout"], self.cfg["height_m"], self.common["table"]["top_z"]
        body = free_body(world, "push_t", plan["pos"], quat_from_yaw(plan["init_pose"][2]))
        kwargs = dict(friction=cfg["friction"], contype=col["contype"], conaffinity=col["conaffinity"],
                      condim=self.common["object_physics"]["condim"], rgba=T_RGBA)
        geom(body, "push_t_bar", "box", (layout["bx"], layout["by"], h / 2), pos=(layout["bar_x"], 0, 0),
             mass=layout["mass_bar"], **kwargs)
        geom(body, "push_t_stem", "box", (layout["sx"], layout["sy"], h / 2), pos=(layout["stem_x"], 0, 0),
             mass=layout["mass_stem"], **kwargs)
        # goal outline: thin, non-colliding, green
        gx, gy = plan["goal"]["center"]
        rot = rot_z(plan["goal"]["yaw"])[:2, :2]
        for name, lx, half in (("goal_bar", layout["bar_x"], (layout["bx"], layout["by"])),
                               ("goal_stem", layout["stem_x"], (layout["sx"], layout["sy"]))):
            wx, wy = rot @ np.array([lx, 0.0])
            geom(world, name, "box", (half[0], half[1], 0.0015), pos=(gx + wx, gy + wy, top + 0.0015),
                 quat=quat_from_yaw(plan["goal"]["yaw"]), rgba=GOAL_RGBA, contype=0, conaffinity=0)

    def tracked_bodies(self, plan):
        return ["push_t"]

    def make_evaluator(self, plan):
        return PushTEvaluator(self.cfg["evaluator"], plan, self.common)

    def instruction_for(self, plan):
        return "Push the blue T-shaped block onto the green T-shaped outline so that it matches the outline's position and orientation. Push only; do not lift the block."
