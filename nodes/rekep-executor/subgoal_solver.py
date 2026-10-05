from __future__ import annotations

import math
from typing import Any

from constraint_evaluator import EvaluationContext, evaluate, residual
from rekep_core.contracts import ContractError


def _candidate_points(expression: Any, context: EvaluationContext) -> list[list[float]]:
    result: list[list[float]] = []
    if not isinstance(expression, dict):
        return result
    if expression.get("op") in {"point", "region_center", "add", "sub"}:
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
    limits = [bounds[axis] for axis in ("x", "y", "z")]
    candidates = [[max(float(limits[index][0]), min(float(limits[index][1]), point[index])) for index in range(3)] for point in candidates]

    def objective(point: list[float]) -> float:
        current = EvaluationContext(snapshot, point, quaternion, held_object, held_anchor)
        return sum(residual(item, current)["violation"] ** 2 for item in constraints)

    def refine(initial: list[float]) -> list[float]:
        best = list(initial)
        step = 0.10
        while step >= 0.001:
            improved = False
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = list(best)
                    candidate[axis] = max(float(limits[axis][0]), min(float(limits[axis][1]), candidate[axis] + sign * step))
                    if objective(candidate) + 1e-12 < objective(best):
                        best, improved = candidate, True
            if not improved:
                step *= 0.5
        return best

    # Refine all geometrically useful seeds, then break feasible ties by the
    # shortest motion. This prevents height-only constraints from changing x/y.
    refined = [refine(candidate) for candidate in candidates]
    feasible = [point for point in refined if objective(point) <= 1e-12]
    best = min(feasible, key=lambda point: math.dist(point, seed)) if feasible else min(refined, key=objective)
    evidence = [residual(item, EvaluationContext(snapshot, best, quaternion, held_object, held_anchor)) for item in constraints]
    if any(item["violation"] > 1e-6 for item in evidence):
        raise ContractError(f"subgoal constraints remain unsatisfied: {max(item['violation'] for item in evidence):.6g}")
    return best, evidence
