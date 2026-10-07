from __future__ import annotations

import math
from typing import Any

from constraint_evaluator import EvaluationContext, evaluate, residual
from rekep_core.contracts import ContractError

# The solver aims inside each tolerance band and keeps the remaining half as
# margin for dense path sampling, the next stage's start and control tracking.
# Acceptance still uses the full program tolerance.
PLANNING_TOLERANCE_FRACTION = 0.5


def _candidate_points(expression: Any, context: EvaluationContext) -> list[list[float]]:
    result: list[list[float]] = []
    if not isinstance(expression, dict):
        return result
    if expression.get("op") in {"point", "region_center", "add", "sub", "scale"}:
        try:
            value = evaluate(expression, context)
            if isinstance(value, list) and len(value) == 3:
                result.append([float(item) for item in value])
        except (ContractError, KeyError, ValueError):
            pass
    for key in ("arg", "left", "right", "point", "normal"):
        result.extend(_candidate_points(expression.get(key), context))
    for child in expression.get("args", []) if isinstance(expression.get("args"), list) else []:
        result.extend(_candidate_points(child, context))
    return result


def solve_subgoal(constraints: list[dict[str, Any]], snapshot: dict[str, Any], seed: list[float], quaternion: list[float], held_object: str | None, held_anchor: tuple[list[float], list[float]] | None, bounds: dict[str, list[float]]) -> tuple[list[float], list[dict[str, Any]]]:
    context = EvaluationContext(snapshot, seed, quaternion, held_object, held_anchor)
    candidates = [list(seed)]
    for constraint in constraints:
        candidates.extend(_candidate_points(constraint["expression"], context))
    if held_object is not None and held_anchor is not None:
        # Target positions are written for held keypoints, not for the pinch
        # frame. Also seed with the end-effector pose that would put the
        # anchor keypoint there, so the grasp offset is not lost.
        anchor_ee, anchor_point = held_anchor
        offset = [anchor_ee[index] - anchor_point[index] for index in range(3)]
        candidates.extend([[point[index] + offset[index] for index in range(3)] for point in candidates[1:]])
    limits = [bounds[axis] for axis in ("x", "y", "z")]
    candidates = [[max(float(limits[index][0]), min(float(limits[index][1]), point[index])) for index in range(3)] for point in candidates]
    planning_constraints = [{**item, "tolerance": float(item["tolerance"]) * PLANNING_TOLERANCE_FRACTION} for item in constraints]

    def violations(point: list[float], items: list[dict[str, Any]]) -> float:
        current = EvaluationContext(snapshot, point, quaternion, held_object, held_anchor)
        return sum(residual(item, current)["violation"] ** 2 for item in items)

    def planning_objective(point: list[float]) -> float:
        return violations(point, planning_constraints)

    def acceptance_objective(point: list[float]) -> float:
        return violations(point, constraints)

    def refine(initial: list[float]) -> list[float]:
        best = list(initial)
        step = 0.10
        while step >= 0.001:
            improved = False
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = list(best)
                    candidate[axis] = max(float(limits[axis][0]), min(float(limits[axis][1]), candidate[axis] + sign * step))
                    if planning_objective(candidate) + 1e-12 < planning_objective(best):
                        best, improved = candidate, True
            if not improved:
                step *= 0.5
        return best

    # Refine all geometrically useful seeds, then break feasible ties by the
    # shortest motion. This prevents height-only constraints from changing x/y.
    refined = [refine(candidate) for candidate in candidates]
    objective = planning_objective
    feasible = [point for point in refined if objective(point) <= 1e-12]
    if not feasible:
        # The margin is a preference; a point inside the program tolerance
        # remains acceptable when the margin cannot be reached.
        objective = acceptance_objective
        feasible = [point for point in refined if objective(point) <= 1e-12]
    # The violation-only objective becomes flat as soon as a constraint is
    # satisfied. Contract each feasible candidate toward the original seed to
    # remove coarse coordinate-search overshoot (e.g. an unnecessary 10 cm
    # lift). Keep a verified feasible endpoint at every iteration.
    if objective(seed) > 1e-12:
        contracted = []
        for point in feasible:
            lower, upper = 0.0, 1.0
            best_feasible = point
            for _ in range(32):
                fraction = (lower + upper) / 2
                candidate = [seed[i] + fraction * (point[i] - seed[i]) for i in range(3)]
                if objective(candidate) <= 1e-12:
                    upper, best_feasible = fraction, candidate
                else:
                    lower = fraction
            contracted.append(best_feasible)
        feasible = contracted
    best = min(feasible, key=lambda point: math.dist(point, seed)) if feasible else min(refined, key=acceptance_objective)
    evidence = [residual(item, EvaluationContext(snapshot, best, quaternion, held_object, held_anchor)) for item in constraints]
    worst = max(evidence, key=lambda item: item["violation"], default=None)
    if worst is not None and worst["violation"] > 1e-6:
        raise ContractError(f"subgoal constraints remain unsatisfied: {worst['violation']:.6g} ({worst['constraint_id']})")
    return best, evidence
