"""Generic evaluation layer.

Evaluators read simulator ground truth (``SceneState``) and nothing else.  They
decide ``success`` / ``partial`` / failure and compute metrics.  They never feed
information back to the policy.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any

import numpy as np

from .mathutil import quat_to_mat, rotation_angle, wrap_angle
from .state import SceneState


class FailureReason(str, Enum):
    WRONG_OBJECT = "WRONG_OBJECT"
    GRASP_FAILED = "GRASP_FAILED"
    OBJECT_DROPPED = "OBJECT_DROPPED"
    PLACEMENT_ERROR = "PLACEMENT_ERROR"
    ORIENTATION_ERROR = "ORIENTATION_ERROR"
    STACK_UNSTABLE = "STACK_UNSTABLE"
    NOT_RELEASED = "NOT_RELEASED"
    PUSH_LIFT_VIOLATION = "PUSH_LIFT_VIOLATION"
    INSERTION_FAILED = "INSERTION_FAILED"
    SEQUENCE_ERROR = "SEQUENCE_ERROR"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


def result(success: bool, score: float, metrics: dict[str, Any],
           failure_reason: FailureReason | None = None, partial: bool = False) -> dict[str, Any]:
    """Standard evaluator return value."""
    if success:
        failure_reason = None
    return {
        "success": bool(success),
        "partial": bool(partial and not success),
        "score": float(round(score, 4)),
        "metrics": metrics,
        "failure_reason": None if failure_reason is None else failure_reason.value,
    }


# ---------------------------------------------------------------- geometry helpers

def is_inside_region(point, center_xy, half_size_xy, yaw: float = 0.0, margin: float = 0.0,
                     z_range: tuple[float, float] | None = None) -> bool:
    """Point inside a (rotated) rectangle on the table, optionally within a z range.

    ``margin`` > 0 enlarges the rectangle.  For "mostly inside" tests with a
    footprint, call this per footprint corner and count (see ``fraction_inside``).
    """
    dx, dy = float(point[0]) - center_xy[0], float(point[1]) - center_xy[1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx, ly = c * dx - s * dy, s * dx + c * dy
    inside = abs(lx) <= half_size_xy[0] + margin and abs(ly) <= half_size_xy[1] + margin
    if inside and z_range is not None:
        inside = z_range[0] <= float(point[2]) <= z_range[1]
    return inside


def fraction_inside(points, center_xy, half_size_xy, yaw: float = 0.0) -> float:
    pts = list(points)
    return sum(is_inside_region(p, center_xy, half_size_xy, yaw) for p in pts) / max(1, len(pts))


def orientation_error(quat_a, quat_b) -> float:
    """Geodesic angle (rad) between two orientations."""
    return rotation_angle(np.asarray(quat_a, float), np.asarray(quat_b, float))


def yaw_error(yaw_a: float, yaw_b: float) -> float:
    """Absolute wrapped yaw difference (rad), in [0, pi]."""
    return abs(wrap_angle(yaw_a - yaw_b))


def upright_error(quat) -> float:
    """Angle between body z and world z (rad)."""
    return math.acos(max(-1.0, min(1.0, quat_to_mat(quat)[2, 2])))


# ---------------------------------------------------------------- probe helpers

def probe_stats(probe: list[SceneState], names: list[str]) -> dict[str, float]:
    """Worst-case speed and pose drift of ``names`` over a probe rollout."""
    max_lin = max_ang = max_drift = 0.0
    first = probe[0]
    for state in probe:
        for name in names:
            body, ref = state.bodies[name], first.bodies[name]
            max_lin = max(max_lin, float(np.linalg.norm(body.linvel)))
            max_ang = max(max_ang, float(np.linalg.norm(body.angvel)))
            max_drift = max(max_drift, float(np.linalg.norm(body.pos - ref.pos)))
    return {"max_linear_speed": max_lin, "max_angular_speed": max_ang, "max_pose_drift": max_drift}


class BaseEvaluator:
    """Subclass per task.  ``cfg`` is the task's ``evaluator`` config section."""

    needs_probe = False

    def __init__(self, cfg: dict, plan: dict, common: dict):
        self.cfg, self.plan, self.common = cfg, plan, common

    def reset(self, state: SceneState) -> None:  # called once after settling
        raise NotImplementedError

    def observe(self, state: SceneState) -> None:  # called every control tick
        raise NotImplementedError

    def evaluate(self, state: SceneState, probe: list[SceneState] | None = None,
                 timed_out: bool = False) -> dict[str, Any]:
        raise NotImplementedError
