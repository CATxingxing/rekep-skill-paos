"""Non-prehensile pushing toward keypoint constraints on the pushed object.

A stage with a push event constrains the pushed object's keypoints (e.g.
onto target keypoints of an outline). The target planar pose (x, y, yaw
change) of that object is solved from those constraints exactly like an
end-effector subgoal, then reached in closed loop: observe, choose one
quasi-static stroke with the closed gripper as pusher, execute, re-observe.
Nothing here depends on a particular object or task; contact points come
from the object's observed outline (footprint_xy).
"""
from __future__ import annotations

import math
from typing import Any

from constraint_evaluator import EvaluationContext, residual
from rekep_core.contracts import ContractError


def _moved(snapshot: dict[str, Any], object_id: str, center: list[float], delta: tuple[float, float, float]) -> dict[str, Any]:
    dx, dy, dtheta = delta
    c, s = math.cos(dtheta), math.sin(dtheta)
    keypoints = []
    for item in snapshot["keypoints"]:
        if item.get("object_id") == object_id:
            px, py = item["position_m"][0] - center[0], item["position_m"][1] - center[1]
            item = {**item, "position_m": [center[0] + dx + c * px - s * py, center[1] + dy + s * px + c * py, item["position_m"][2]]}
        keypoints.append(item)
    return {**snapshot, "keypoints": keypoints}


def object_center(snapshot: dict[str, Any], object_id: str) -> list[float]:
    points = [item for item in snapshot["keypoints"] if item.get("object_id") == object_id]
    if not points:
        raise ContractError(f"no keypoints observed for pushed object {object_id}")
    center = next((item for item in points if item.get("kind") == "object_center_estimate"), points[0])
    return list(center["position_m"])


def solve_target(constraints: list[dict[str, Any]], snapshot: dict[str, Any], object_id: str, ee: list[float], quaternion: list[float]) -> dict[str, Any]:
    """Smallest planar motion of the object that satisfies the constraints."""
    center = object_center(snapshot, object_id)

    def violation(delta: tuple[float, float, float], fraction: float) -> float:
        context = EvaluationContext(_moved(snapshot, object_id, center, delta), ee, quaternion)
        return sum(residual({**item, "tolerance": float(item["tolerance"]) * fraction}, context)["violation"] ** 2 for item in constraints)

    def descend(start: tuple[float, float, float], fraction: float) -> tuple[float, float, float]:
        best, steps = list(start), [0.05, 0.05, 0.4]
        while steps[0] >= 0.0005:
            improved = False
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = list(best)
                    candidate[axis] += sign * steps[axis]
                    if violation(tuple(candidate), fraction) + 1e-14 < violation(tuple(best), fraction):
                        best, improved = candidate, True
            if not improved:
                steps = [value * 0.5 for value in steps]
        return tuple(best)

    candidates = [descend((0.0, 0.0, start), 0.5) for start in (0.0, math.pi / 2, -math.pi / 2, math.pi)]
    best = min(candidates, key=lambda delta: (round(violation(delta, 1.0), 12), abs(math.remainder(delta[2], math.tau)) + math.hypot(delta[0], delta[1])))
    best = (best[0], best[1], math.remainder(best[2], math.tau))
    context = EvaluationContext(snapshot, ee, quaternion)
    current = [residual(item, context) for item in constraints]
    return {"delta": best, "center": center, "feasible": violation(best, 1.0) <= 1e-12,
            # The same micrometre slack as "feasible" (sum of squares <= 1e-12),
            # so a target that needs no motion is never reported as unsatisfied.
            "satisfied": all(item["violation"] <= 1e-6 for item in current), "current_evidence": current}


def choose_stroke(snapshot: dict[str, Any], object_id: str, target: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """One straight push: start, end (pusher center, base frame) and its purpose."""
    dx, dy, dtheta = target["delta"]
    center = target["center"]
    entry = next((item for item in snapshot["objects"] if item["object_id"] == object_id), {})
    footprint = [list(point) for point in entry.get("footprint_xy", [])]
    radius = float(config.get("pusher_radius_m", 0.015))
    standoff = float(config.get("push_standoff_m", 0.02))
    longest = float(config.get("push_max_stroke_m", 0.05))
    extremes = [item["position_m"] for item in snapshot["keypoints"] if item.get("object_id") == object_id and item.get("kind") == "top_extreme"]
    translation = math.hypot(dx, dy)
    levers = [(math.hypot(p[0] - center[0], p[1] - center[1]), p) for p in extremes]
    longest_lever = max((lever for lever, _ in levers), default=0.0)
    # Rotate when the turn dominates, or when only a (small) turn is left.
    turn_only = translation < float(config.get("push_min_translation_m", 0.002)) and abs(dtheta) > 1e-3
    rotate = longest_lever > 0.01 and (turn_only or (abs(dtheta) > float(config.get("push_rotation_threshold_rad", 0.12)) and abs(dtheta) * longest_lever >= 0.3 * translation))
    if rotate:
        # A push perpendicular to the lever arm at an outer point turns the
        # object and also drags it along the push; among the long levers,
        # take the one whose push direction also helps the translation.
        sign = 1.0 if dtheta > 0 else -1.0
        options = []
        for lever, point in levers:
            if lever < 0.6 * longest_lever:
                continue
            rx, ry = point[0] - center[0], point[1] - center[1]
            direction = (-sign * ry / lever, sign * rx / lever)
            helps = (direction[0] * dx + direction[1] * dy) / translation if translation > 1e-6 else 0.0
            options.append((helps + 0.5 * lever / longest_lever, lever, point, direction))
        _, lever, lever_point, direction = max(options, key=lambda item: item[0])
        near = [p for p in footprint if math.hypot(p[0] - lever_point[0], p[1] - lever_point[1]) <= 0.02] or [lever_point[:2]]
        # The gain is the push length per radian per metre of lever arm,
        # learned from this object's responses to earlier strokes.
        gain = float(config.get("push_gain_rotate", 1.0))
        length = min(max(0.7 * abs(dtheta) * lever * gain, 0.004), longest)
        predicted = {"dtheta": sign * length / (lever * gain), "advance": 0.0}
        purpose = "rotate"
    else:
        if translation < 1e-6:
            raise ContractError("push target requires neither translation nor rotation")
        direction = (dx / translation, dy / translation)
        near = [p for p in footprint if abs(direction[0] * (p[1] - center[1]) - direction[1] * (p[0] - center[0])) <= 0.012] or [center[:2]]
        gain = float(config.get("push_gain_translate", 1.0))
        length = min(max(translation * gain, 0.004), longest)
        predicted = {"dtheta": 0.0, "advance": length / gain}
        purpose = "translate"
    length = min(length + float(config.get("push_extra_depth_m", 0.0)), longest)
    contact = min(near, key=lambda p: p[0] * direction[0] + p[1] * direction[1])
    # Push low: a contact high on the object tips it over its far bottom edge.
    bottom = next((item["position_m"][2] for item in snapshot["keypoints"] if item.get("object_id") == object_id and item.get("kind") == "bottom_center"), center[2])
    # The closed gripper reaches below the tip frame (pusher_bottom_below_tip_m);
    # a stroke lower than that would press it into the support and the
    # descent could never reach its goal.
    floor = float(bottom) + float(config.get("pusher_bottom_below_tip_m", 0.0)) + float(config.get("push_support_clearance_m", 0.002))
    z = max(min(float(center[2]), float(bottom) + float(config.get("push_height_above_bottom_m", 0.008))), floor,
            float(config["workspace_m"]["z"][0]) + 0.002)
    start = [contact[0] - direction[0] * (radius + standoff), contact[1] - direction[1] * (radius + standoff), z]
    # The pusher descends vertically onto the stroke start; it must not land
    # on the object (a concave outline can reach behind the contact point).
    # The closed fingers' outer links reach beyond the pads, so the clearance
    # is larger than the pusher radius.
    clear = radius + float(config.get("push_start_clearance_m", 0.015))
    for _ in range(int(config.get("push_start_max_shift_steps", 8))):
        if all(math.hypot(p[0] - start[0], p[1] - start[1]) >= clear for p in footprint):
            break
        start = [start[0] - direction[0] * 0.005, start[1] - direction[1] * 0.005, z]
    end = [contact[0] + direction[0] * (length - radius), contact[1] + direction[1] * (length - radius), z]
    # Release before lifting: only ease the pads off the face, then rise.
    # The closed Robotiq fingers' outer links slope outward above the pads
    # and can rest on the object's top edge; any horizontal retreat while
    # they touch it drags that edge and tips the object, rising lifts them
    # off it (diagnosed from contacts in the push_t harness, see rekep-bug.md).
    back = float(config.get("push_release_m", 0.003))
    backoff = [end[0] - direction[0] * back, end[1] - direction[1] * back, z]
    return {"backoff": backoff, "purpose": purpose, "direction": list(direction), "contact_xy": list(contact[:2]), "length_m": length, "start": start, "end": end, "target_delta": list(target["delta"]), "predicted": predicted, "gain": gain}


def update_gain(gain: float, stroke: dict[str, Any], before: tuple[float, float, float], after: tuple[float, float, float]) -> float:
    """Rescale the purpose's gain by how far the object actually responded.

    A stroke that moved the object more than predicted shortens later ones
    (square-root step, bounded) so small corrections near the goal do not
    overshoot; one that moved it less lengthens them.
    """
    predicted = stroke["predicted"]
    if stroke["purpose"] == "rotate":
        expected = abs(predicted["dtheta"])
        actual = math.copysign(1.0, predicted["dtheta"]) * math.remainder(after[2] - before[2], math.tau)
    else:
        expected = predicted["advance"]
        actual = (after[0] - before[0]) * stroke["direction"][0] + (after[1] - before[1]) * stroke["direction"][1]
    if expected <= 1e-6:
        return gain
    ratio = min(max(actual / expected, 0.25), 4.0)
    return min(max(gain / math.sqrt(ratio), 0.3), 3.0)
