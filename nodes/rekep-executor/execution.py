from __future__ import annotations

import json
import math
import threading
import time
import uuid
from typing import Any, Callable

from constraint_evaluator import EvaluationContext, residual
from kinematics import KinematicGuard
from path_solver import solve_path
from push import choose_stroke, solve_target, update_gain
from rekep_core.contracts import RESULT_SCHEMA, ContractError, atomic_json, runtime_root, validate_snapshot
from subgoal_solver import solve_subgoal


def solve_stage(stages: list[dict[str, Any]], stage_index: int, snapshot: dict[str, Any], config: dict[str, Any], ee: list[float], quaternion: list[float], held_object: str | None, held_anchor: tuple[list[float], list[float]] | None) -> dict[str, Any]:
    """Endpoint, Cartesian path and Action segments of one stage from one belief state."""
    guard = KinematicGuard(config)
    stage = stages[stage_index]
    pushed = next((event["object_id"] for event in stage.get("events", []) if event["type"] == "push"), None)
    if pushed is not None:
        # Strokes depend on how the object actually moves; they are chosen
        # in closed loop at execution. Here only the target must exist.
        target = solve_target(stage["subgoal_constraints"], snapshot, pushed, ee, quaternion)
        if not target["feasible"]:
            raise ContractError(f"push target of stage {stage['stage_id']} is infeasible")
        evidence = [{**item, "role": "push_target_now"} for item in target["current_evidence"]]
        return {"stage_id": stage["stage_id"], "segments": [], "constraints": stage, "solver_evidence": evidence,
                "endpoint": list(ee), "held_object": held_object, "held_anchor": held_anchor, "push_object": pushed,
                "push_target_delta": list(target["delta"])}
    # A path constraint holds on the whole stage path, so the endpoint must
    # satisfy it as well. The endpoint is also the next stage's start; a
    # grasp anchors at this endpoint, so the held state there is unchanged.
    # After a release the evaluator has no model of the dropped object,
    # so the next stage's path start is only checked by solve_path.
    endpoint_constraints = [("subgoal", item) for item in stage["subgoal_constraints"]]
    endpoint_constraints += [("path_endpoint", item) for item in stage["path_constraints"]]
    releases = any(event["type"] != "grasp" for event in stage.get("events", []))
    if stage_index + 1 < len(stages) and not releases:
        endpoint_constraints += [("next_path_start", item) for item in stages[stage_index + 1]["path_constraints"]]
    endpoint, subgoal_evidence = solve_subgoal([item for _, item in endpoint_constraints], snapshot, ee, quaternion, held_object, held_anchor, config["workspace_m"])
    subgoal_evidence = [{**item, "role": role} for (role, _), item in zip(endpoint_constraints, subgoal_evidence, strict=True)]
    grasping = any(event["type"] == "grasp" for event in stage.get("events", []))
    free_approach = config.get("free_approach_height_m")
    waypoints, path_evidence = solve_path(
        ee, endpoint, stage["path_constraints"], snapshot, quaternion, held_object, held_anchor,
        clearance_m=float(config.get("transport_clearance_m", 0.12)),
        samples_per_segment=int(config.get("path_samples_per_segment", 9)),
        max_cartesian_step_m=float(config.get("max_cartesian_step_m", 0.05)),
        approach_height_m=float(config.get("pregrasp_approach_height_m", 0.10)) if grasping else None,
        held_descent_m=float(config.get("held_descent_m", 0.0)),
        free_approach_height_m=None if free_approach is None else float(free_approach),
        motion=str(stage.get("motion", "auto")),
    )
    segments = []
    # Moves while holding may use gentler trajectory limits than free motion.
    holding = held_object is not None
    velocity_scale = float(config.get("held_velocity_scale" if holding else "velocity_scale", config.get("velocity_scale", 0.4)))
    acceleration_scale = float(config.get("held_acceleration_scale" if holding else "acceleration_scale", config.get("acceleration_scale", 0.4)))
    for index, waypoint in enumerate(waypoints[1:]):
        guard.validate_pose({"position_m": waypoint, "quaternion_xyzw": quaternion})
        guard.validate_segment(waypoints[index], waypoint)
    if config.get("continuous_paths", False):
        runs = _straight_runs(waypoints)
    else:
        runs = [[list(point)] for before, point in zip(waypoints, waypoints[1:]) if point != before]
    for index, run in enumerate(runs):
        segment = {
            "segment_id": f"{stage['stage_id']}.move.{index}",
            "type": "move_pose",
            "target_pose": {"position_m": run[-1], "quaternion_xyzw": quaternion},
            # One move deadline per Cartesian step in the run.
            "deadline_ms": int(config.get("move_deadline_ms", 15000)) * len(run),
            "settle_ms": int(config.get("inter_segment_settle_ms", 100)),
            "velocity_scale": velocity_scale,
            "acceleration_scale": acceleration_scale,
        }
        if len(run) > 1:
            segment["via_positions"] = [list(point) for point in run[:-1]]
        segments.append(segment)
    for index, event in enumerate(stage.get("events", [])):
        segments.append({"segment_id": f"{stage['stage_id']}.event.{index}", "type": "gripper", "event": event, "command_position": 255.0 if event["type"] == "grasp" else 0.0, "deadline_ms": int(config.get("gripper_deadline_ms", 8000))})
        if event["type"] == "grasp":
            if "grasp_contact_position_range" in config:
                segments[-1]["contact_position_range"] = config["grasp_contact_position_range"]
            held_object = event["object_id"]
            anchor = next((item["position_m"] for item in snapshot["keypoints"] if item.get("object_id") == held_object), endpoint)
            held_anchor = (list(endpoint), list(anchor))
        else:
            held_object = None
            held_anchor = None
    evidence = subgoal_evidence + [{**item, "role": "path"} for item in path_evidence]
    return {"stage_id": stage["stage_id"], "segments": segments, "constraints": stage, "solver_evidence": evidence,
            "endpoint": endpoint, "held_object": held_object, "held_anchor": held_anchor}


def _predict_release(snapshot: dict[str, Any], held_object: str, held_anchor: tuple[list[float], list[float]], ee: list[float]) -> dict[str, Any]:
    """Belief after a release: the object's keypoints stay where the gripper left them."""
    delta = [ee[index] - held_anchor[0][index] for index in range(3)]
    keypoints = []
    for item in snapshot["keypoints"]:
        if item.get("object_id") == held_object:
            item = {**item, "position_m": [item["position_m"][index] + delta[index] for index in range(3)]}
        keypoints.append(item)
    return {**snapshot, "keypoints": keypoints}


def merge_observation(belief: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Fresh observation, keeping the believed keypoints of objects it did not see.

    An object can be fully hidden, e.g. by the arm in front of a tower; it
    has not moved unless it was held, and a held object is moved with the
    end effector by the evaluation context.
    """
    observed = {item["object_id"] for item in observation["objects"]}
    seen = {item["keypoint_id"] for item in observation["keypoints"]}
    keypoints = list(observation["keypoints"])
    keypoints += [item for item in belief["keypoints"] if item["keypoint_id"] not in seen and "object_id" in item and item["object_id"] not in observed]
    objects = list(observation["objects"]) + [item for item in belief["objects"] if item["object_id"] not in observed]
    regions = {item["region_id"]: item for item in belief.get("regions", [])}
    regions.update({item["region_id"]: item for item in observation.get("regions", [])})
    keypoints += [item for item in belief["keypoints"] if item["keypoint_id"] not in seen and "region_id" in item]
    return {**observation, "objects": objects, "keypoints": keypoints, "regions": list(regions.values())}


def solve_program(program: dict[str, Any], snapshot: dict[str, Any], config: dict[str, Any], progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    state = snapshot.get("robot_state", {})
    pose = state.get("end_effector_pose", {}) if isinstance(state, dict) else {}
    ee = list(pose.get("position_m", config["default_end_effector_pose"]["position_m"]))
    quaternion = list(pose.get("quaternion_xyzw", config["default_end_effector_pose"]["quaternion_xyzw"]))
    held_object: str | None = None
    held_anchor: tuple[list[float], list[float]] | None = None
    belief = snapshot
    solved_stages, all_evidence = [], []
    stages = program["stages"]
    for stage_index, stage in enumerate(stages):
        solved = solve_stage(stages, stage_index, belief, config, ee, quaternion, held_object, held_anchor)
        if held_object is not None and solved["held_object"] is None:
            belief = _predict_release(belief, held_object, held_anchor, solved["endpoint"])
        held_object, held_anchor = solved["held_object"], solved["held_anchor"]
        all_evidence.extend(solved["solver_evidence"])
        solved_stages.append({key: solved[key] for key in ("stage_id", "segments", "constraints", "solver_evidence")})
        ee = solved["endpoint"]
        progress({"phase": "solve_stage", "stage_id": stage["stage_id"], "percent": round((stage_index + 1) / len(stages) * 30)})
    return {"stages": solved_stages, "evidence": all_evidence,
            "checks": {"finite": True, "workspace": True, "segment_length": True, "ik_and_collision": "enforced_by_motion_server_terminal_result"}}


def _straight_runs(waypoints: list[list[float]]) -> list[list[list[float]]]:
    """Group consecutive collinear Cartesian steps; corners start a new run.

    A run becomes one MovePose blended through its intermediate waypoints, so
    the arm stops only at real direction changes (e.g. lift -> transit), never
    in the middle of a straight line.
    """
    runs: list[list[list[float]]] = []
    direction: list[float] | None = None
    previous = waypoints[0]
    for point in waypoints[1:]:
        delta = [point[axis] - previous[axis] for axis in range(3)]
        length = math.sqrt(sum(value * value for value in delta))
        if length <= 1e-12:
            continue
        unit = [value / length for value in delta]
        if direction is None or sum(a * b for a, b in zip(unit, direction)) < 1.0 - 1e-6:
            runs.append([])
        runs[-1].append(list(point))
        direction, previous = unit, point
    return runs


def _metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    parameters = value.get("parameters")
    return parameters if isinstance(parameters, dict) else value


class ChildActionFailure(RuntimeError):
    def __init__(self, status: str | None, result: Any):
        self.terminal_result = result.model_dump(mode="json")
        self.goal_status = status
        self.error_code = result.error_code
        super().__init__(f"child action failed: status={status}, code={result.error_code}, message={result.message}")


class GraspVerificationFailure(RuntimeError):
    error_code = "REKEP_GRASP_CONTACT_INVALID"


def verify_grasp_contact(segment: dict[str, Any], result: dict[str, Any]) -> None:
    bounds = segment.get("contact_position_range")
    if bounds is None:
        return
    position = result.get("position")
    if not result.get("stalled") or not isinstance(position, (float, int)) or not float(bounds[0]) <= position <= float(bounds[1]):
        raise GraspVerificationFailure(f"grasp contact outside calibrated range {bounds}: position={position}, stalled={result.get('stalled')}, reached_goal={result.get('reached_goal')}")


class ChildActionBridge:
    """Only one child action can be active; each result must be terminal-success."""

    def __init__(self, send_output: Callable[..., None]):
        self.send_output = send_output
        self.condition = threading.Condition()
        self.active_goal_id: str | None = None
        self.result: Any = None
        self.result_status: str | None = None
        self.latest_snapshot: dict[str, Any] | None = None
        self.snapshot_sequence = 0

    def handle_input(self, input_id: str, value: Any, metadata: dict[str, Any]) -> None:
        from forge_msgs import GripperCommandResult, MovePoseResult

        with self.condition:
            if input_id in {"move_pose_result", "gripper_result"}:
                parameters = _metadata(metadata)
                if parameters.get("goal_id") != self.active_goal_id:
                    return
                self.result = MovePoseResult.from_arrow(value) if input_id == "move_pose_result" else GripperCommandResult.from_arrow(value)
                self.result_status = str(parameters.get("goal_status", ""))
                self.condition.notify_all()
            elif input_id == "snapshot_in":
                try:
                    raw = value[0].as_py() if hasattr(value, "__getitem__") else bytes(value).decode("utf-8")
                    self.latest_snapshot = validate_snapshot(json.loads(raw))
                    self.snapshot_sequence += 1
                    self.condition.notify_all()
                except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, ContractError):
                    return

    def _run_child(self, *, output_id: str, cancel_output_id: str, goal: Any, deadline_s: float, cancel: threading.Event, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        import pyarrow as pa

        goal_id = f"rekep-{uuid.uuid4().hex}"
        with self.condition:
            if self.active_goal_id is not None:
                raise RuntimeError("a child action is already active")
            self.active_goal_id, self.result, self.result_status = goal_id, None, None
        self.send_output(output_id, goal.to_arrow(), metadata={**(metadata or {}), "goal_id": goal_id})
        deadline = time.monotonic() + deadline_s
        cancel_sent = False
        with self.condition:
            while self.result is None:
                if cancel.is_set() and not cancel_sent:
                    self.send_output(cancel_output_id, pa.array([True], type=pa.bool_()), metadata={"goal_id": goal_id})
                    cancel_sent = True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not cancel_sent:
                        self.send_output(cancel_output_id, pa.array([True], type=pa.bool_()), metadata={"goal_id": goal_id})
                    self.active_goal_id = None
                    raise TimeoutError(f"child action timed out: {goal_id}")
                self.condition.wait(timeout=min(0.1, remaining))
            result, status = self.result, self.result_status
            self.active_goal_id = None
        if status != "succeeded" or result.error_code != "SUCCESS":
            raise ChildActionFailure(status, result)
        return result.model_dump(mode="json")

    def move_pose(self, segment: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        from forge_msgs import MovePoseGoal, Pose

        position = segment["target_pose"]["position_m"]
        quaternion = segment["target_pose"]["quaternion_xyzw"]
        goal = MovePoseGoal(
            group_name="nova2_arm", reference_frame="base_link", target_frame="pinch",
            target_pose=Pose(x=position[0], y=position[1], z=position[2], qx=quaternion[0], qy=quaternion[1], qz=quaternion[2], qw=quaternion[3]),
            velocity_scale=float(segment.get("velocity_scale", 0.4)), acceleration_scale=float(segment.get("acceleration_scale", 0.4)),
            position_tolerance_m=0.01, orientation_tolerance_rad=0.08,
        )
        metadata = None
        if segment.get("via_positions"):
            # Patched motion_server: blend through these poses without stopping.
            metadata = {"via_poses": json.dumps([[*point, *quaternion] for point in segment["via_positions"]])}
        return self._run_child(output_id="move_pose_goal", cancel_output_id="move_pose_cancel", goal=goal, deadline_s=segment["deadline_ms"] / 1000, cancel=cancel, metadata=metadata)

    def gripper(self, segment: dict[str, Any], cancel: threading.Event) -> dict[str, Any]:
        from forge_msgs import GripperCommandGoal

        goal = GripperCommandGoal(position=float(segment["command_position"]))
        return self._run_child(output_id="gripper_goal", cancel_output_id="gripper_cancel", goal=goal, deadline_s=segment["deadline_ms"] / 1000, cancel=cancel)

    def observe(self, timeout_s: float = 15) -> dict[str, Any]:
        import pyarrow as pa

        with self.condition:
            sequence = self.snapshot_sequence
        self.send_output("observe_request", pa.array([json.dumps({"refresh": True})], type=pa.string()))
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.snapshot_sequence == sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a refreshed observation")
                self.condition.wait(timeout=min(0.1, remaining))
            assert self.latest_snapshot is not None
            return self.latest_snapshot


class StageVerificationFailure(RuntimeError):
    error_code = "REKEP_POST_STAGE_CONSTRAINT_VIOLATED"


def _observe_bound(bridge: ChildActionBridge, program: dict[str, Any]) -> dict[str, Any]:
    observation = bridge.observe()
    if observation.get("session_id") != program["session_id"] or observation.get("scene_revision") != program["scene_revision"]:
        raise RuntimeError("session or scene revision changed during execution")
    return observation


def _verify_subgoal(stage: dict[str, Any], observation: dict[str, Any], record: dict[str, Any]) -> None:
    target = next((item["target_pose"] for item in reversed(stage["segments"]) if item["type"] == "move_pose"), None)
    if target is None:
        return
    context = EvaluationContext(observation, target["position_m"], target["quaternion_xyzw"])
    evidence = [residual(item, context) for item in stage["constraints"]["subgoal_constraints"]]
    record["constraint_evidence"] = evidence
    if any(item["violation"] > 0.01 for item in evidence):
        raise StageVerificationFailure(f"post-stage constraint verification failed in {stage['stage_id']}")


class StageReplanFailure(RuntimeError):
    error_code = "REKEP_REPLAN_INFEASIBLE"


def _reached(result: dict[str, Any], fallback: list[float]) -> list[float]:
    pose = result.get("final_pose") if isinstance(result, dict) else None
    try:
        return [float(pose["x"]), float(pose["y"]), float(pose["z"])]
    except (TypeError, KeyError, ValueError):
        return list(fallback)


def _anchor(belief: dict[str, Any], held_object: str, ee: list[float]) -> tuple[list[float], list[float]]:
    point = next((item["position_m"] for item in belief["keypoints"] if item.get("object_id") == held_object), ee)
    return (list(ee), list(point))


def _carried(belief: dict[str, Any], held_object: str | None, held_anchor: tuple[list[float], list[float]] | None, ee: list[float]) -> dict[str, Any]:
    """Belief with the held object's keypoints moved along with the end effector."""
    if held_object is None or held_anchor is None:
        return belief
    return _predict_release(belief, held_object, held_anchor, ee)


def _with_believed(belief: dict[str, Any], observation: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Verification view: observed keypoints plus the last observed ones of hidden objects.

    An object the arm hides was not held, so it stays where it was last
    seen; the record lists every keypoint verified from memory.
    """
    seen = {item["keypoint_id"] for item in observation["keypoints"]}
    remembered = sorted(item["keypoint_id"] for item in belief["keypoints"] if item["keypoint_id"] not in seen)
    if remembered:
        record.setdefault("verified_with_last_seen_keypoints", sorted(set(record.get("verified_with_last_seen_keypoints", [])) | set(remembered)))
    return belief


class PushNotConverged(RuntimeError):
    error_code = "REKEP_PUSH_NOT_CONVERGED"


def _segments(stage_id: str, label: str, waypoints: list[list[float]], quaternion: list[float], config: dict[str, Any], *, free: bool = False) -> list[dict[str, Any]]:
    guard = KinematicGuard(config)
    for index, waypoint in enumerate(waypoints[1:]):
        guard.validate_pose({"position_m": waypoint, "quaternion_xyzw": quaternion})
        guard.validate_segment(waypoints[index], waypoint)
    segments = []
    for index, run in enumerate(_straight_runs(waypoints)):
        segment = {"segment_id": f"{stage_id}.{label}.{index}", "type": "move_pose",
                   "target_pose": {"position_m": run[-1], "quaternion_xyzw": quaternion},
                   "deadline_ms": int(config.get("move_deadline_ms", 15000)) * len(run),
                   "settle_ms": int(config.get("inter_segment_settle_ms", 100)),
                   "velocity_scale": float(config.get("velocity_scale", 0.4)) if free else float(config.get("push_velocity_scale", config.get("held_velocity_scale", 0.2))),
                   "acceleration_scale": float(config.get("acceleration_scale", 0.4)) if free else float(config.get("push_acceleration_scale", config.get("held_acceleration_scale", 0.2)))}
        if len(run) > 1:
            segment["via_positions"] = [list(point) for point in run[:-1]]
        segments.append(segment)
    return segments


def _run_push(stage: dict[str, Any], object_id: str, program: dict[str, Any], belief: dict[str, Any], ee: list[float], quaternion: list[float], config: dict[str, Any], bridge: ChildActionBridge, cancel: threading.Event, record: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    """Closed-loop strokes until the pushed object's constraints hold.

    The gripper is closed into a compact pusher for the strokes and opened
    again afterwards, so a later grasp starts from an open gripper.
    """
    belief, ee = _push_strokes(stage, object_id, program, belief, ee, quaternion, config, bridge, cancel, record)
    opening = {"segment_id": f"{stage['stage_id']}.open", "type": "gripper", "command_position": 0.0, "deadline_ms": int(config.get("gripper_deadline_ms", 8000))}
    record["segments"].append({"segment_id": opening["segment_id"], "status": "succeeded", "terminal_result": bridge.gripper(opening, cancel)})
    return belief, ee


def _planar_pose(snapshot: dict[str, Any], object_id: str) -> tuple[float, float, float] | None:
    entry = next((item for item in snapshot.get("objects", []) if item.get("object_id") == object_id), None)
    if entry is None or "centroid_m" not in entry:
        return None
    return (float(entry["centroid_m"][0]), float(entry["centroid_m"][1]), float(entry.get("yaw_since_first_seen_rad", 0.0)))


def _push_strokes(stage: dict[str, Any], object_id: str, program: dict[str, Any], belief: dict[str, Any], ee: list[float], quaternion: list[float], config: dict[str, Any], bridge: ChildActionBridge, cancel: threading.Event, record: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    constraints = stage["constraints"]["subgoal_constraints"]
    close = {"segment_id": f"{stage['stage_id']}.close", "type": "gripper", "command_position": 255.0, "deadline_ms": int(config.get("gripper_deadline_ms", 8000))}
    record["segments"].append({"segment_id": close["segment_id"], "status": "succeeded", "terminal_result": bridge.gripper(close, cancel)})
    strokes = record.setdefault("push_strokes", [])
    step = float(config.get("max_cartesian_step_m", 0.025))
    # A stroke the object did not respond to (pusher fell short of the face)
    # is followed by a deeper one in that direction. The extra depth is kept
    # once the object responds: it estimates the pusher's shortfall, which
    # depends on the push direction relative to the closed fingers.
    sectors = 8
    extra_depth = [0.0] * sectors
    gains = {"rotate": float(config.get("push_gain_rotate", 1.0)), "translate": float(config.get("push_gain_translate", 1.0))}
    for _ in range(int(config.get("push_max_strokes", 16))):
        if cancel.is_set():
            return belief, ee
        target = solve_target(constraints, belief, object_id, ee, quaternion)
        if target["satisfied"]:
            record["push_result"] = {"strokes": len(strokes), "evidence": target["current_evidence"]}
            return belief, ee
        if not target["feasible"]:
            raise PushNotConverged(f"push target became infeasible after {len(strokes)} strokes")
        before = _planar_pose(belief, object_id)
        tuned = {**config, "push_gain_rotate": gains["rotate"], "push_gain_translate": gains["translate"]}
        stroke = choose_stroke(belief, object_id, target, tuned)
        sector = int(round(math.atan2(stroke["direction"][1], stroke["direction"][0]) / (2 * math.pi / sectors))) % sectors
        if extra_depth[sector] > 0.0:
            stroke = choose_stroke(belief, object_id, target, {**tuned, "push_extra_depth_m": extra_depth[sector]})
        # Travel just above the pushed object: far strokes are near the
        # edge of the reachable workspace, which shrinks with height.
        approach, _ = solve_path(ee, stroke["start"], [], belief, quaternion, None, None,
                                 clearance_m=0.0, samples_per_segment=2,
                                 max_cartesian_step_m=step, free_approach_height_m=float(config.get("push_approach_height_m", 0.05)))
        push, _ = solve_path(stroke["start"], stroke["end"], [], belief, quaternion, None, None, clearance_m=0.0,
                             samples_per_segment=2, max_cartesian_step_m=step, motion="straight")
        lift = [list(stroke["end"]), list(stroke["backoff"]), [stroke["backoff"][0], stroke["backoff"][1], stroke["backoff"][2] + float(config.get("push_retreat_m", 0.06))]]
        index = len(strokes)
        segments = (_segments(stage["stage_id"], f"push{index}.approach", approach, quaternion, config)
                    + _segments(stage["stage_id"], f"push{index}.stroke", push, quaternion, config)
                    + _segments(stage["stage_id"], f"push{index}.retreat", lift, quaternion, config))
        if config.get("push_observe_from_default_pose", True):
            # Observe from the pose of the first observation: an arm left
            # just above the object hides it from the camera, and tracking
            # a mostly hidden object after a push can lock onto a wrong pose.
            home = list(config["default_end_effector_pose"]["position_m"])
            clear, _ = solve_path(lift[-1], home, [], belief, quaternion, None, None, clearance_m=0.0,
                                  samples_per_segment=2, max_cartesian_step_m=step)
            segments += _segments(stage["stage_id"], f"push{index}.observe", clear, quaternion, config, free=True)
        for segment in segments:
            result = bridge.move_pose(segment, cancel)
            record["segments"].append({"segment_id": segment["segment_id"], "status": "succeeded", "terminal_result": result})
            ee = _reached(result, segment["target_pose"]["position_m"])
            if cancel.wait(max(0, int(segment.get("settle_ms", 0))) / 1000):
                return belief, ee
        observation = _observe_bound(bridge, program)
        belief = merge_observation(belief, observation)
        after = _planar_pose(belief, object_id)
        moved = before is None or after is None or math.dist(before[:2], after[:2]) > float(config.get("push_min_response_m", 0.002)) or abs(after[2] - before[2]) > float(config.get("push_min_response_rad", 0.02))
        if not moved:
            extra_depth[sector] += float(config.get("push_no_response_depth_step_m", 0.01))
        elif before is not None and after is not None and extra_depth[sector] == 0.0:
            gains[stroke["purpose"]] = update_gain(gains[stroke["purpose"]], stroke, before, after)
        strokes.append({**stroke, "observation_id": observation["observation_id"], "object_moved": moved})
    target = solve_target(constraints, belief, object_id, ee, quaternion)
    if target["satisfied"]:
        record["push_result"] = {"strokes": len(strokes), "evidence": target["current_evidence"]}
        return belief, ee
    raise PushNotConverged(f"push constraints still violated after {len(strokes)} strokes")


def execute(program: dict[str, Any], solved: dict[str, Any], bridge: ChildActionBridge, cancel: threading.Event, progress: Callable[[dict[str, Any]], None], *, deadline_ms: int, config: dict[str, Any] | None = None, snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the stages serially.

    With ``replan_each_stage`` every stage after the first is re-solved from
    the latest observation (ReKep's closed loop: constraints are fixed,
    keypoint positions are re-observed), the reached end-effector pose and
    the object actually held. Without it the pre-solved segments are used.
    """
    started = time.monotonic()
    absolute_deadline = started + deadline_ms / 1000
    run_id = f"execution_{uuid.uuid4().hex}"
    records = []
    status, failure, failure_code = "succeeded", None, None
    replan = bool(config and snapshot is not None and config.get("replan_each_stage", False))
    belief = snapshot
    quaternion: list[float] = []
    ee: list[float] = []
    if replan:
        quaternion = list(config["default_end_effector_pose"]["quaternion_xyzw"])
        ee = list(config["default_end_effector_pose"]["position_m"])
    held_object: str | None = None
    held_anchor: tuple[list[float], list[float]] | None = None
    try:
        for stage_index, planned in enumerate(solved["stages"]):
            if cancel.is_set():
                status, failure = "cancelled", "parent action cancelled"
                break
            if time.monotonic() >= absolute_deadline:
                raise TimeoutError("execution deadline expired")
            record = {"stage_id": planned["stage_id"], "status": "executing", "segments": []}
            records.append(record)
            stage = planned
            if replan and stage_index > 0:
                try:
                    fresh = solve_stage(program["stages"], stage_index, belief, config, ee, quaternion, held_object, held_anchor)
                except ContractError as exc:
                    raise StageReplanFailure(f"stage {planned['stage_id']} is infeasible from the latest observation: {exc}") from exc
                stage = {key: fresh[key] for key in ("stage_id", "segments", "constraints", "solver_evidence")}
                record["replanned_from_observation_id"] = belief["observation_id"]
                record["replanned_start_m"] = list(ee)
                record["replanned_segments"] = stage["segments"]
                record["replan_evidence"] = stage["solver_evidence"]
            last_target: list[float] | None = None
            for segment in stage["segments"]:
                if cancel.is_set():
                    break
                is_grasp = segment["type"] == "gripper" and segment.get("event", {}).get("type") == "grasp"
                is_release = segment["type"] == "gripper" and segment.get("event", {}).get("type") == "release"
                if is_grasp:
                    observation = bridge.observe()
                    record["pre_grasp_observation_id"] = observation["observation_id"]
                    if replan:
                        belief = merge_observation(belief, observation)
                if is_release:
                    # The stage subgoal describes the held object at the end
                    # of the motion. Verify it before opening: afterwards the
                    # object drops and settles, which the subgoal never
                    # constrained. Fail closed without releasing.
                    observation = _observe_bound(bridge, program)
                    record["pre_release_observation_id"] = observation["observation_id"]
                    if replan:
                        belief = merge_observation(_carried(belief, held_object, held_anchor, ee), observation)
                        if held_object is not None:
                            held_anchor = _anchor(belief, held_object, ee)
                        observation = _with_believed(belief, observation, record)
                    _verify_subgoal(stage, observation, record)
                    record["constraint_verification"] = "verified_before_release"
                try:
                    result = bridge.move_pose(segment, cancel) if segment["type"] == "move_pose" else bridge.gripper(segment, cancel)
                except ChildActionFailure as exc:
                    record["segments"].append({"segment_id": segment["segment_id"], "status": "failed", "goal_status": exc.goal_status, "terminal_result": exc.terminal_result})
                    raise
                segment_record = {"segment_id": segment["segment_id"], "status": "succeeded", "terminal_result": result}
                record["segments"].append(segment_record)
                if segment["type"] == "move_pose":
                    last_target = segment["target_pose"]["position_m"]
                    if replan:
                        ee = _reached(result, last_target)
                if is_grasp:
                    segment_record["command_position"] = segment["command_position"]
                    segment_record["contact_position_range"] = segment.get("contact_position_range")
                    try:
                        verify_grasp_contact(segment, result)
                    except GraspVerificationFailure as exc:
                        segment_record["status"] = "failed"
                        segment_record["failure_code"] = exc.error_code
                        # Diagnostic observation is best-effort: its failure
                        # must not replace the primary empty-grasp evidence.
                        try:
                            record["observation_id"] = bridge.observe()["observation_id"]
                        except Exception as observation_error:
                            record["observation_error"] = str(observation_error)
                        raise
                    if replan:
                        held_object = segment["event"]["object_id"]
                        held_anchor = _anchor(belief, held_object, ee)
                if is_release and replan:
                    held_object, held_anchor = None, None
                if segment["type"] == "move_pose" and cancel.wait(max(0, int(segment.get("settle_ms", 0))) / 1000):
                    break
            pushed = next((event["object_id"] for event in stage["constraints"].get("events", []) if event["type"] == "push"), None)
            if pushed is not None and not cancel.is_set():
                if not replan:
                    raise StageReplanFailure("push stages require replan_each_stage")
                belief, ee = _run_push(stage, pushed, program, belief, ee, quaternion, config, bridge, cancel, record)
            if cancel.is_set():
                record["status"] = "cancelled"
                status, failure = "cancelled", "parent action cancelled"
                break
            observation = _observe_bound(bridge, program)
            record["observation_id"] = observation["observation_id"]
            if replan:
                belief = merge_observation(_carried(belief, held_object, held_anchor, ee), observation)
                if held_object is not None:
                    held_anchor = _anchor(belief, held_object, ee)
                observation = _with_believed(belief, observation, record)
            target = next((item["target_pose"] for item in reversed(stage["segments"]) if item["type"] == "move_pose"), None)
            stage_events = stage["constraints"].get("events", [])
            grasp_deferred = any(item.get("type") == "grasp" for item in stage_events)
            released = any(item.get("type") == "release" for item in stage_events)
            if released:
                pass  # verified on the pre-release observation above
            elif target and not grasp_deferred:
                _verify_subgoal(stage, observation, record)
            elif target:
                # A grasp occludes the object. The contact-aware gripper
                # terminal result is the grasp evidence; geometric validation
                # resumes after the first post-grasp motion.
                record["constraint_verification"] = "deferred_until_post_grasp_motion"
            record["observation_id"] = observation["observation_id"]
            record["status"] = "succeeded"
            progress({"phase": "execute_stage", "stage_id": stage["stage_id"], "percent": 30 + round((stage_index + 1) / len(solved["stages"]) * 70)})
    except Exception as exc:
        status, failure = "failed", str(exc)
        failure_code = getattr(exc, "error_code", None) or failure_code or ("REKEP_TIMEOUT" if isinstance(exc, TimeoutError) else "REKEP_EXECUTION_FAILED")
        if records and records[-1]["status"] == "executing":
            records[-1]["status"] = "failed"
    result = {
        "schema_version": RESULT_SCHEMA,
        "execution_id": run_id,
        "status": status,
        "session_id": program["session_id"],
        "scene_revision": program["scene_revision"],
        "observation_id": program["observation_id"],
        "plan_id": program["plan_id"],
        "plan_digest": program["plan_digest"],
        "stages": records,
        "failure_reason": failure,
        "failure_code": failure_code,
        "solver_evidence": solved["evidence"],
        "checks": solved["checks"],
    }
    atomic_json(runtime_root() / "run" / "rekep" / "executions" / f"{run_id}.json", result)
    return result
