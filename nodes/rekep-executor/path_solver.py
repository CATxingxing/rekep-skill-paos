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


def solve_path(start: list[float], end: list[float], constraints: list[dict[str, Any]], snapshot: dict[str, Any], quaternion: list[float], held_object: str | None, held_anchor: tuple[list[float], list[float]] | None, *, clearance_m: float, samples_per_segment: int, max_cartesian_step_m: float) -> tuple[list[list[float]], list[dict[str, Any]]]:
    if not math.isfinite(clearance_m) or clearance_m < 0:
        raise ContractError("transport clearance height must be finite and non-negative")
    if held_object is not None and start[:2] != end[:2]:
        # A minimum transit height, not an increment on every stage. The
        # program already specifies lifting/clearance constraints; adding a
        # second lift here can leave the reachable workspace. Pure vertical
        # moves need no lateral transit and must not overshoot their endpoint.
        clearance = max(start[2], end[2], clearance_m)
        waypoints = [list(start), [start[0], start[1], clearance], [end[0], end[1], clearance], list(end)]
    else:
        waypoints = [list(start), list(end)]
    waypoints = _subdivide(waypoints, max_cartesian_step_m)
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
    if any(item["violation"] > 1e-6 for item in evidence):
        raise ContractError(f"path constraints remain unsatisfied: {max(item['violation'] for item in evidence):.6g}")
    return waypoints, evidence
