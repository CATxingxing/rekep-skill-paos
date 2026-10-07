from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .geometry import finite_vector
from .ids import canonical_json, digest

SNAPSHOT_SCHEMA = "rekep.perception.v5"
PROGRAM_SCHEMA = "rekep.constraint_program.v4"
RESULT_SCHEMA = "rekep.execution_result.v3"
DSL_OPERATORS = frozenset({
    "constant", "point", "region_center", "vector", "add", "sub", "norm",
    "distance", "dot", "angle", "abs", "neg", "min", "max",
    "point_plane_signed_distance", "inside_region", "orientation_error",
    "scale", "normalize", "component", "horizontal_distance",
})


class ContractError(ValueError):
    """An input is malformed, stale, unbound, or outside the safe contract."""


def runtime_root() -> Path:
    override = os.environ.get("REKEP_RUNTIME_ROOT")
    return Path(override).expanduser() if override else Path.home() / ".paos-rekep"


def atomic_json(path: Path, value: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def _unique(items: Iterable[dict[str, Any]], field: str, label: str) -> set[str]:
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ContractError(f"{label}[{index}] must be an object")
        identifier = require_string(item.get(field), f"{label}[{index}].{field}")
        if identifier in seen:
            raise ContractError(f"duplicate {field}: {identifier}")
        seen.add(identifier)
    return seen


def validate_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ContractError(f"schema_version must be {SNAPSHOT_SCHEMA}")
    allowed = {
        "schema_version", "session_id", "scene_revision", "observation_id",
        "timestamp_ns", "frame_id", "rgb_digest", "depth_digest",
        "calibration_digest", "objects", "keypoints", "regions", "overlay_ref",
        "robot_state", "model",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContractError("unknown snapshot field(s): " + ", ".join(unknown))
    for field in ("session_id", "observation_id", "frame_id", "rgb_digest", "depth_digest", "calibration_digest"):
        require_string(value.get(field), field)
    revision = value.get("scene_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ContractError("scene_revision must be a non-negative integer")
    timestamp = value.get("timestamp_ns")
    if not isinstance(timestamp, str) or not timestamp.isascii() or not timestamp.isdecimal() or int(timestamp) <= 0:
        raise ContractError("timestamp_ns must be a positive decimal string")
    objects, keypoints, regions = value.get("objects"), value.get("keypoints"), value.get("regions", [])
    if not isinstance(objects, list) or not objects:
        raise ContractError("objects must be a non-empty array")
    if not isinstance(keypoints, list) or not keypoints:
        raise ContractError("keypoints must be a non-empty array")
    if not isinstance(regions, list):
        raise ContractError("regions must be an array")
    object_ids = _unique(objects, "object_id", "objects")
    _unique(keypoints, "keypoint_id", "keypoints")
    region_ids = _unique(regions, "region_id", "regions")
    for index, keypoint in enumerate(keypoints):
        # A keypoint lies on exactly one observed object or one static region.
        if "region_id" in keypoint:
            if "object_id" in keypoint or keypoint.get("region_id") not in region_ids:
                raise ContractError(f"keypoints[{index}] references an unknown region")
        elif keypoint.get("object_id") not in object_ids:
            raise ContractError(f"keypoints[{index}] references an unknown object")
        try:
            finite_vector(keypoint.get("position_m"), 3, f"keypoints[{index}].position_m")
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
    unsigned = {key: item for key, item in value.items() if key not in {"observation_id", "overlay_ref"}}
    if value["observation_id"] != digest(unsigned):
        raise ContractError("observation_id does not match snapshot content")
    return value


def _validate_expression(expression: Any, *, keypoints: set[str], regions: set[str], depth: int, count: list[int]) -> None:
    if depth > 12:
        raise ContractError("constraint expression exceeds maximum depth")
    count[0] += 1
    if count[0] > 512:
        raise ContractError("constraint program exceeds maximum AST node count")
    if not isinstance(expression, dict):
        raise ContractError("constraint expression must be an object")
    op = expression.get("op")
    if op not in DSL_OPERATORS:
        raise ContractError(f"unsupported constraint operator: {op!r}")
    if op == "constant":
        require_number(expression.get("value"), "constant.value")
    elif op == "point":
        reference = expression.get("keypoint_id")
        if reference != "$ee" and reference not in keypoints:
            raise ContractError(f"unknown keypoint reference {reference!r}")
    elif op == "region_center":
        if expression.get("region_id") not in regions:
            raise ContractError(f"unknown region reference {expression.get('region_id')!r}")
    elif op == "vector":
        try:
            finite_vector(expression.get("value"), 3, "vector.value")
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
    elif op == "inside_region" and expression.get("region_id") not in regions:
        raise ContractError(f"unknown region reference {expression.get('region_id')!r}")
    elif op == "scale":
        require_number(expression.get("factor"), "scale.factor")
    elif op == "component":
        if expression.get("axis") not in {"x", "y", "z"}:
            raise ContractError("component.axis must be x, y, or z")
    elif op == "orientation_error":
        try:
            finite_vector(expression.get("target_quaternion_xyzw"), 4, "target quaternion")
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
    for field in ("arg", "left", "right", "point", "normal"):
        if field in expression:
            _validate_expression(expression[field], keypoints=keypoints, regions=regions, depth=depth + 1, count=count)
    if "args" in expression:
        children = expression["args"]
        if not isinstance(children, list) or not children:
            raise ContractError("expression.args must be a non-empty array")
        for child in children:
            _validate_expression(child, keypoints=keypoints, regions=regions, depth=depth + 1, count=count)


def validate_program(value: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    validate_snapshot(snapshot)
    if not isinstance(value, dict) or value.get("schema_version") != PROGRAM_SCHEMA:
        raise ContractError(f"schema_version must be {PROGRAM_SCHEMA}")
    allowed = {
        "schema_version", "plan_id", "plan_digest", "session_id", "scene_revision",
        "observation_id", "instruction", "instruction_digest", "frame_id", "units",
        "generator", "stages",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContractError("unknown program field(s): " + ", ".join(unknown))
    for field in ("plan_id", "instruction", "instruction_digest", "frame_id"):
        require_string(value.get(field), field)
    for field in ("session_id", "scene_revision", "observation_id", "frame_id"):
        if value.get(field) != snapshot.get(field):
            raise ContractError(f"{field} does not match the bound snapshot")
    if value.get("instruction_digest") != digest(value.get("instruction")):
        raise ContractError("instruction_digest does not match instruction")
    if value.get("units") != {"length": "m", "angle": "rad", "quaternion": "xyzw"}:
        raise ContractError("units must be SI and quaternion xyzw")
    generator = value.get("generator")
    if not isinstance(generator, dict) or generator.get("mode") != "vlm":
        raise ContractError("generator.mode must be vlm")
    stages = value.get("stages")
    if not isinstance(stages, list) or not stages or len(stages) > 64:
        raise ContractError("stages must contain between 1 and 64 items")
    keypoints = {item["keypoint_id"] for item in snapshot["keypoints"]}
    objects = {item["object_id"] for item in snapshot["objects"]}
    regions = {item["region_id"] for item in snapshot.get("regions", [])}
    seen_stages: set[str] = set()
    seen_constraints: set[str] = set()
    held: str | None = None
    count = [0]
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ContractError(f"stages[{stage_index}] must be an object")
        stage_id = require_string(stage.get("stage_id"), f"stages[{stage_index}].stage_id")
        if stage_id in seen_stages:
            raise ContractError(f"duplicate stage_id: {stage_id}")
        dependencies = stage.get("depends_on", [])
        if not isinstance(dependencies, list) or any(item not in seen_stages for item in dependencies):
            raise ContractError(f"stage {stage_id} depends_on must reference prior stages")
        seen_stages.add(stage_id)
        for group in ("subgoal_constraints", "path_constraints"):
            constraints = stage.get(group)
            if not isinstance(constraints, list):
                raise ContractError(f"stage {stage_id}.{group} must be an array")
            for constraint in constraints:
                if not isinstance(constraint, dict):
                    raise ContractError(f"stage {stage_id} has a non-object constraint")
                constraint_id = require_string(constraint.get("constraint_id"), "constraint_id")
                if constraint_id in seen_constraints:
                    raise ContractError(f"duplicate constraint_id: {constraint_id}")
                seen_constraints.add(constraint_id)
                if constraint.get("relation") not in {"le", "ge", "eq"}:
                    raise ContractError(f"constraint {constraint_id} has an invalid relation")
                require_number(constraint.get("target"), f"constraint {constraint_id}.target")
                if require_number(constraint.get("tolerance"), f"constraint {constraint_id}.tolerance") < 0:
                    raise ContractError(f"constraint {constraint_id}.tolerance must be non-negative")
                _validate_expression(constraint.get("expression"), keypoints=keypoints, regions=regions, depth=0, count=count)
        if stage.get("motion", "auto") not in {"auto", "straight"}:
            raise ContractError(f"stage {stage_id}.motion must be auto or straight")
        events = stage.get("events", [])
        if not isinstance(events, list):
            raise ContractError(f"stage {stage_id}.events must be an array")
        for event in events:
            if not isinstance(event, dict) or event.get("type") not in {"grasp", "release", "push"}:
                raise ContractError(f"stage {stage_id} has an unsupported event")
            object_id = event.get("object_id")
            if object_id not in objects:
                raise ContractError(f"unknown object reference {object_id!r}")
            if event["type"] == "push":
                if held is not None:
                    raise ContractError("cannot push while an object is held")
                if len(events) != 1:
                    raise ContractError(f"stage {stage_id}: a push event must be the stage's only event")
            elif event["type"] == "grasp":
                if held is not None:
                    raise ContractError("cannot grasp while another object is held")
                held = object_id
            elif held != object_id:
                raise ContractError("release must match the held object")
            else:
                held = None
    unsigned = dict(value)
    supplied = unsigned.pop("plan_digest", None)
    if supplied != digest(unsigned):
        raise ContractError("plan_digest does not match program content")
    return value


def verify_binding(program: dict[str, Any], *, session_id: str, scene_revision: int, observation_id: str, plan_id: str, plan_digest: str) -> None:
    expected = {"session_id": session_id, "scene_revision": scene_revision, "observation_id": observation_id, "plan_id": plan_id, "plan_digest": plan_digest}
    for field, value in expected.items():
        if program.get(field) != value:
            raise ContractError(f"{field} does not match")
    unsigned = dict(program)
    supplied = unsigned.pop("plan_digest", None)
    if supplied != digest(unsigned):
        raise ContractError("plan digest does not match content")
