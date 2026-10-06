from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Callable

from constraint_evaluator import EvaluationContext, residual
from kinematics import KinematicGuard
from path_solver import solve_path
from rekep_core.contracts import RESULT_SCHEMA, ContractError, atomic_json, runtime_root, validate_snapshot
from subgoal_solver import solve_subgoal


def solve_program(program: dict[str, Any], snapshot: dict[str, Any], config: dict[str, Any], progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    guard = KinematicGuard(config)
    state = snapshot.get("robot_state", {})
    pose = state.get("end_effector_pose", {}) if isinstance(state, dict) else {}
    ee = list(pose.get("position_m", config["default_end_effector_pose"]["position_m"]))
    quaternion = list(pose.get("quaternion_xyzw", config["default_end_effector_pose"]["quaternion_xyzw"]))
    held_object: str | None = None
    held_anchor: tuple[list[float], list[float]] | None = None
    solved_stages, all_evidence = [], []
    for stage_index, stage in enumerate(program["stages"]):
        endpoint, subgoal_evidence = solve_subgoal(stage["subgoal_constraints"], snapshot, ee, quaternion, held_object, held_anchor, config["workspace_m"])
        waypoints, path_evidence = solve_path(
            ee, endpoint, stage["path_constraints"], snapshot, quaternion, held_object, held_anchor,
            clearance_m=float(config.get("transport_clearance_m", 0.12)),
            samples_per_segment=int(config.get("path_samples_per_segment", 9)),
            max_cartesian_step_m=float(config.get("max_cartesian_step_m", 0.05)),
        )
        segments = []
        for index, waypoint in enumerate(waypoints[1:]):
            pose_value = {"position_m": waypoint, "quaternion_xyzw": quaternion}
            guard.validate_pose(pose_value)
            guard.validate_segment(waypoints[index], waypoint)
            if waypoints[index] != waypoint:
                segments.append({
                    "segment_id": f"{stage['stage_id']}.move.{index}",
                    "type": "move_pose",
                    "target_pose": pose_value,
                    "deadline_ms": int(config.get("move_deadline_ms", 15000)),
                    "settle_ms": int(config.get("inter_segment_settle_ms", 100)),
                })
        for index, event in enumerate(stage.get("events", [])):
            segments.append({"segment_id": f"{stage['stage_id']}.event.{index}", "type": "gripper", "event": event, "command_position": 255.0 if event["type"] == "grasp" else 0.0, "deadline_ms": int(config.get("gripper_deadline_ms", 8000))})
            if event["type"] == "grasp":
                held_object = event["object_id"]
                anchor = next((item["position_m"] for item in snapshot["keypoints"] if item["object_id"] == held_object), endpoint)
                held_anchor = (list(endpoint), list(anchor))
            else:
                held_object = None
                held_anchor = None
        evidence = subgoal_evidence + path_evidence
        all_evidence.extend(evidence)
        solved_stages.append({"stage_id": stage["stage_id"], "segments": segments, "constraints": stage, "solver_evidence": evidence})
        ee = endpoint
        progress({"phase": "solve_stage", "stage_id": stage["stage_id"], "percent": round((stage_index + 1) / len(program["stages"]) * 30)})
    return {"stages": solved_stages, "evidence": all_evidence, "checks": {"finite": True, "workspace": True, "segment_length": True, "ik_and_collision": "enforced_by_motion_server_terminal_result"}}


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

    def _run_child(self, *, output_id: str, cancel_output_id: str, goal: Any, deadline_s: float, cancel: threading.Event) -> dict[str, Any]:
        import pyarrow as pa

        goal_id = f"rekep-{uuid.uuid4().hex}"
        with self.condition:
            if self.active_goal_id is not None:
                raise RuntimeError("a child action is already active")
            self.active_goal_id, self.result, self.result_status = goal_id, None, None
        self.send_output(output_id, goal.to_arrow(), metadata={"goal_id": goal_id})
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
            velocity_scale=0.4, acceleration_scale=0.4,
            position_tolerance_m=0.01, orientation_tolerance_rad=0.08,
        )
        return self._run_child(output_id="move_pose_goal", cancel_output_id="move_pose_cancel", goal=goal, deadline_s=segment["deadline_ms"] / 1000, cancel=cancel)

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


def execute(program: dict[str, Any], solved: dict[str, Any], bridge: ChildActionBridge, cancel: threading.Event, progress: Callable[[dict[str, Any]], None], *, deadline_ms: int) -> dict[str, Any]:
    started = time.monotonic()
    absolute_deadline = started + deadline_ms / 1000
    run_id = f"execution_{uuid.uuid4().hex}"
    records = []
    status, failure, failure_code = "succeeded", None, None
    try:
        for stage_index, stage in enumerate(solved["stages"]):
            if cancel.is_set():
                status, failure = "cancelled", "parent action cancelled"
                break
            if time.monotonic() >= absolute_deadline:
                raise TimeoutError("execution deadline expired")
            record = {"stage_id": stage["stage_id"], "status": "executing", "segments": []}
            records.append(record)
            for segment in stage["segments"]:
                if cancel.is_set():
                    break
                try:
                    result = bridge.move_pose(segment, cancel) if segment["type"] == "move_pose" else bridge.gripper(segment, cancel)
                except ChildActionFailure as exc:
                    record["segments"].append({"segment_id": segment["segment_id"], "status": "failed", "goal_status": exc.goal_status, "terminal_result": exc.terminal_result})
                    raise
                record["segments"].append({"segment_id": segment["segment_id"], "status": "succeeded", "terminal_result": result})
                if segment["type"] == "move_pose" and cancel.wait(max(0, int(segment.get("settle_ms", 0))) / 1000):
                    break
            if cancel.is_set():
                record["status"] = "cancelled"
                status, failure = "cancelled", "parent action cancelled"
                break
            observation = bridge.observe()
            record["observation_id"] = observation["observation_id"]
            if observation.get("session_id") != program["session_id"] or observation.get("scene_revision") != program["scene_revision"]:
                raise RuntimeError("session or scene revision changed during execution")
            target = next((item["target_pose"] for item in reversed(stage["segments"]) if item["type"] == "move_pose"), None)
            stage_events = stage["constraints"].get("events", [])
            grasp_deferred = any(item.get("type") == "grasp" for item in stage_events)
            if target and not grasp_deferred:
                context = EvaluationContext(observation, target["position_m"], target["quaternion_xyzw"])
                evidence = [residual(item, context) for item in stage["constraints"]["subgoal_constraints"]]
                record["constraint_evidence"] = evidence
                if any(item["violation"] > 0.01 for item in evidence):
                    failure_code = "REKEP_POST_STAGE_CONSTRAINT_VIOLATED"
                    raise RuntimeError(f"post-stage constraint verification failed in {stage['stage_id']}")
            elif target:
                # A grasp occludes the object and DINO re-proposes keypoints on
                # the remaining visible surface.  The contact-aware gripper
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
