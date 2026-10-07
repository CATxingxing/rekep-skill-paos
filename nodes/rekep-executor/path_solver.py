from __future__ import annotations

import math
from typing import Any

from constraint_evaluator import EvaluationContext, residual
from rekep_core.contracts import ContractError


def _subdivide(waypoints: list[list[float]], maximum_step_m: float) -> list[list[float]]:
    if not math.isfinite(maximum_step_m) or maximum_step_m <= 0:
        raise ContractError("maximum Cartesian step must be finite and positive")
    result = [list(waypoints[0])]
    for left, right in zip(waypoints, waypoints[1:], strict=False):
        steps = max(1, math.ceil(math.dist(left, right) / maximum_step_m))
        for index in range(1, steps + 1):
            alpha = index / steps
            result.append([(1.0 - alpha) * left[axis] + alpha * right[axis] for axis in range(3)])
    return result


def _over(start: list[float], end: list[float], height: float) -> list[list[float]]:
    """Rise vertically to ``height``, travel level, descend vertically.

    From a start above ``height`` the travel descends directly to the point
    ``height`` above the endpoint: the arm reaches less far at the height of
    a raised home pose, and everything to avoid lies below ``height``.
    """
    points = [list(start)]
    if height > start[2] + 1e-9:
        points.append([start[0], start[1], height])
    points.append([end[0], end[1], max(height, end[2])])
    if height > end[2] + 1e-9:
        points.append(list(end))
    return points


def _evidence(waypoints: list[list[float]], constraints: list[dict[str, Any]], snapshot: dict[str, Any], quaternion: list[float], held_object: str | None, held_anchor: tuple[list[float], list[float]] | None, samples_per_segment: int) -> list[dict[str, Any]]:
    evidence = []
    for constraint in constraints:
        samples = []
        for left, right in zip(waypoints, waypoints[1:], strict=False):
            for index in range(samples_per_segment):
                alpha = index / max(1, samples_per_segment - 1)
                point = [(1 - alpha) * left[axis] + alpha * right[axis] for axis in range(3)]
                samples.append(residual(constraint, EvaluationContext(snapshot, point, quaternion, held_object, held_anchor)))
        worst = max(samples, key=lambda item: item["violation"])
        worst["dense_samples"] = len(samples)
        evidence.append(worst)
    return evidence


def solve_path(start: list[float], end: list[float], constraints: list[dict[str, Any]], snapshot: dict[str, Any], quaternion: list[float], held_object: str | None, held_anchor: tuple[list[float], list[float]] | None, *, clearance_m: float, samples_per_segment: int, max_cartesian_step_m: float, approach_height_m: float | None = None, held_descent_m: float = 0.0, free_approach_height_m: float | None = None, motion: str = "auto") -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Cartesian waypoints from ``start`` to ``end`` that satisfy the path constraints.

    Candidate shapes, first feasible wins:
    * grasp approach: travel ``approach_height_m`` above the grasp pose, then descend onto it;
    * holding an object: travel at a transit height (at least ``clearance_m``)
      and finish with a vertical ``held_descent_m`` onto the placement pose;
    * free motion: rise, travel and descend ``free_approach_height_m`` so the
      open gripper does not sweep through objects;
    * a straight line (always tried last; the only shape for ``motion: straight``,
      e.g. a push whose path constraints pin the height).
    """
    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise ContractError("transport clearance height must be finite and non-negative")
    lateral = math.hypot(end[0] - start[0], end[1] - start[1]) > 1e-6
    candidates: list[list[list[float]]] = []
    if motion != "straight":
        if approach_height_m is not None:
            if not math.isfinite(approach_height_m) or approach_height_m <= 0:
                raise ContractError("pre-grasp approach height must be finite and positive")
            candidates.append(_over(start, end, end[2] + approach_height_m))
        elif held_object is not None and lateral:
            # A minimum transit height, not an increment on every stage: the
            # program specifies its own lifting constraints.
            candidates.append(_over(start, end, max(end[2] + held_descent_m, clearance_m)))
        elif free_approach_height_m is not None and lateral:
            candidates.append(_over(start, end, max(end[2] + free_approach_height_m, clearance_m)))
    candidates.append([list(start), list(end)])
    failure = None
    for shape in candidates:
        waypoints = _subdivide(shape, max_cartesian_step_m)
        evidence = _evidence(waypoints, constraints, snapshot, quaternion, held_object, held_anchor, samples_per_segment)
        worst = max(evidence, key=lambda item: item["violation"], default=None)
        if worst is None or worst["violation"] <= 1e-6:
            return waypoints, evidence
        failure = failure or worst
    assert failure is not None
    raise ContractError(f"path constraints remain unsatisfied: {failure['violation']:.6g} ({failure['constraint_id']})")
