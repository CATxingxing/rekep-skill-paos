from __future__ import annotations

import math
from typing import Any

from rekep_core.contracts import ContractError, require_number
from rekep_core.geometry import finite_vector


Vector = list[float]


def _add(left: Vector, right: Vector) -> Vector:
    return [left[index] + right[index] for index in range(3)]


def _sub(left: Vector, right: Vector) -> Vector:
    return [left[index] - right[index] for index in range(3)]


def _dot(left: Vector, right: Vector) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _norm(value: Vector) -> float:
    return math.sqrt(_dot(value, value))


class EvaluationContext:
    def __init__(self, snapshot: dict[str, Any], ee_position: Vector, ee_quaternion_xyzw: list[float], held_object: str | None = None, held_anchor: tuple[Vector, Vector] | None = None):
        self.ee_position = finite_vector(ee_position, 3, "ee_position")
        self.ee_quaternion_xyzw = finite_vector(ee_quaternion_xyzw, 4, "ee_quaternion_xyzw")
        self.points = {item["keypoint_id"]: list(item["position_m"]) for item in snapshot["keypoints"]}
        self.regions = {item["region_id"]: item for item in snapshot.get("regions", [])}
        if held_object is not None and held_anchor is not None:
            anchor_ee, anchor_point = held_anchor
            delta = _sub(self.ee_position, anchor_ee)
            for item in snapshot["keypoints"]:
                if item["object_id"] == held_object:
                    self.points[item["keypoint_id"]] = _add(list(item["position_m"]), delta)


def evaluate(expression: dict[str, Any], context: EvaluationContext) -> float | Vector:
    op = expression["op"]
    if op == "constant":
        return require_number(expression["value"], "constant.value")
    if op == "vector":
        return finite_vector(expression["value"], 3, "vector.value")
    if op == "point":
        identifier = expression["keypoint_id"]
        return list(context.ee_position if identifier == "$ee" else context.points[identifier])
    if op == "region_center":
        region = context.regions[expression["region_id"]]
        lower, upper = region["bounds_min_m"], region["bounds_max_m"]
        return [(float(lower[index]) + float(upper[index])) / 2 for index in range(3)]
    if op in {"add", "sub"}:
        left = finite_vector(evaluate(expression["left"], context), 3, f"{op}.left")
        right = finite_vector(evaluate(expression["right"], context), 3, f"{op}.right")
        return _add(left, right) if op == "add" else _sub(left, right)
    if op == "norm":
        return _norm(finite_vector(evaluate(expression["arg"], context), 3, "norm.arg"))
    if op == "distance":
        left = finite_vector(evaluate(expression["left"], context), 3, "distance.left")
        right = finite_vector(evaluate(expression["right"], context), 3, "distance.right")
        return _norm(_sub(left, right))
    if op == "dot":
        return _dot(finite_vector(evaluate(expression["left"], context), 3, "dot.left"), finite_vector(evaluate(expression["right"], context), 3, "dot.right"))
    if op == "angle":
        left = finite_vector(evaluate(expression["left"], context), 3, "angle.left")
        right = finite_vector(evaluate(expression["right"], context), 3, "angle.right")
        denominator = _norm(left) * _norm(right)
        if denominator <= 1e-12:
            raise ContractError("angle operands must be non-zero")
        return math.acos(max(-1.0, min(1.0, _dot(left, right) / denominator)))
    if op in {"abs", "neg"}:
        value = require_number(evaluate(expression["arg"], context), f"{op}.arg")
        return abs(value) if op == "abs" else -value
    if op in {"min", "max"}:
        values = [require_number(evaluate(item, context), f"{op}.args") for item in expression["args"]]
        return min(values) if op == "min" else max(values)
    if op == "point_plane_signed_distance":
        point = finite_vector(evaluate(expression["point"], context), 3, "plane.point")
        normal = finite_vector(evaluate(expression["normal"], context), 3, "plane.normal")
        origin = finite_vector(expression.get("origin_m", [0, 0, 0]), 3, "plane.origin")
        length = _norm(normal)
        if length <= 1e-12:
            raise ContractError("plane normal must be non-zero")
        return _dot(_sub(point, origin), [item / length for item in normal])
    if op == "inside_region":
        point_expression = expression.get("point", expression.get("arg"))
        point = finite_vector(evaluate(point_expression, context), 3, "inside_region.point")
        region = context.regions[expression["region_id"]]
        lower, upper = region["bounds_min_m"], region["bounds_max_m"]
        return max(max(float(lower[index]) - point[index], point[index] - float(upper[index]), 0.0) for index in range(3))
    if op == "orientation_error":
        target = finite_vector(expression["target_quaternion_xyzw"], 4, "target quaternion")
        dot = abs(sum(target[index] * context.ee_quaternion_xyzw[index] for index in range(4)))
        return 2.0 * math.acos(max(-1.0, min(1.0, dot)))
    raise ContractError(f"unsupported constraint operator: {op}")


def residual(constraint: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:
    raw = require_number(evaluate(constraint["expression"], context), "constraint expression")
    target = float(constraint["target"])
    tolerance = float(constraint["tolerance"])
    relation = constraint["relation"]
    if relation == "le":
        violation = max(0.0, raw - target - tolerance)
    elif relation == "ge":
        violation = max(0.0, target - raw - tolerance)
    else:
        violation = max(0.0, abs(raw - target) - tolerance)
    return {"constraint_id": constraint["constraint_id"], "raw_value": raw, "relation": relation, "target": target, "tolerance": tolerance, "violation": violation}
