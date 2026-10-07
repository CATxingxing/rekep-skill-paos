from __future__ import annotations

import threading
import pytest

from execution import ChildActionFailure, execute
from helpers import program, snapshot


class Bridge:
    def __init__(self):
        self.calls = []

    def move_pose(self, segment, _cancel):
        self.calls.append(segment["segment_id"])
        return {"error_code": "SUCCESS"}

    def gripper(self, segment, _cancel):
        self.calls.append(segment["segment_id"])
        return {"error_code": "SUCCESS"}

    def observe(self):
        return snapshot()


class ReproposedKeypointBridge(Bridge):
    def observe(self):
        value = snapshot()
        value["keypoints"][0]["position_m"] = [0.2, -0.4, 0.0]
        return value


def test_executor_waits_and_preserves_segment_order(tmp_path, monkeypatch):
    monkeypatch.setenv("REKEP_RUNTIME_ROOT", str(tmp_path))
    value = program()
    solved = {
        "stages": [{
            "stage_id": "one",
            "segments": [
                {"segment_id": "one.move.0", "type": "move_pose", "target_pose": {"position_m": [0.2, -0.4, 0.05], "quaternion_xyzw": [1, 0, 0, 0]}, "deadline_ms": 1000},
                {"segment_id": "one.event.0", "type": "gripper", "command_position": 255.0, "deadline_ms": 1000},
            ],
            "constraints": {"subgoal_constraints": [], "path_constraints": [], "events": []},
            "solver_evidence": [],
        }],
        "evidence": [],
        "checks": {"workspace": True},
    }
    bridge = Bridge()
    result = execute(value, solved, bridge, threading.Event(), lambda _: None, deadline_ms=5000)
    assert result["status"] == "succeeded"
    assert bridge.calls == ["one.move.0", "one.event.0"]


def test_grasp_geometry_is_deferred_when_keypoints_are_reproposed(tmp_path, monkeypatch):
    monkeypatch.setenv("REKEP_RUNTIME_ROOT", str(tmp_path))
    value = program()
    solved = {
        "stages": [{
            "stage_id": "grasp",
            "segments": [
                {"segment_id": "grasp.move.0", "type": "move_pose", "target_pose": {"position_m": [0.2, -0.4, 0.05], "quaternion_xyzw": [1, 0, 0, 0]}, "deadline_ms": 1000},
                {"segment_id": "grasp.event.0", "type": "gripper", "command_position": 255.0, "deadline_ms": 1000},
            ],
            "constraints": value["stages"][0],
            "solver_evidence": [],
        }],
        "evidence": [],
        "checks": {"workspace": True},
    }

    result = execute(value, solved, ReproposedKeypointBridge(), threading.Event(), lambda _: None, deadline_ms=5000)

    assert result["status"] == "succeeded"
    assert result["stages"][0]["constraint_verification"] == "deferred_until_post_grasp_motion"


def test_failed_child_preserves_diagnostics_and_stops_serial_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("REKEP_RUNTIME_ROOT", str(tmp_path))
    class Result:
        error_code = "FINAL_POSE_TOLERANCE_VIOLATED"
        message = "final Cartesian pose tolerance was violated"
        def model_dump(self, **_):
            return {"error_code": self.error_code, "final_position_error_m": .01019}
    class FailedBridge(Bridge):
        def move_pose(self, segment, cancel):
            self.calls.append(segment["segment_id"])
            raise ChildActionFailure("aborted", Result())
    bridge = FailedBridge()
    solved = {"stages": [{"stage_id": "transfer", "segments": [
        {"segment_id": "failed", "type": "move_pose"},
        {"segment_id": "must_not_run", "type": "move_pose"},
    ]}], "evidence": [], "checks": {}}
    result = execute(program(), solved, bridge, threading.Event(), lambda _: None, deadline_ms=5000)
    assert bridge.calls == ["failed"]
    assert result["failure_code"] == "FINAL_POSE_TOLERANCE_VIOLATED"
    assert result["stages"][0]["status"] == "failed"
    assert result["stages"][0]["segments"][0]["terminal_result"]["final_position_error_m"] == .01019


@pytest.mark.parametrize("position,stalled", [(251.58, True), (255., False), (50., True), (125., False), (None, True)])
@pytest.mark.parametrize("observation_fails", [False, True])
def test_invalid_grasp_stops_before_lift_and_preserves_evidence(tmp_path, monkeypatch, position, stalled, observation_fails):
    monkeypatch.setenv("REKEP_RUNTIME_ROOT", str(tmp_path))
    class GraspBridge(Bridge):
        observations = 0
        def gripper(self, segment, cancel):
            super().gripper(segment, cancel)
            return {"error_code": "SUCCESS", "position": position, "stalled": stalled, "reached_goal": False}
        def observe(self):
            self.observations += 1
            if observation_fails and self.observations > 1:
                raise TimeoutError("diagnostic camera timed out")
            return snapshot()
    bridge = GraspBridge()
    solved = {"stages": [{"stage_id": "grasp", "segments": [
        {"segment_id": "close", "type": "gripper", "event": {"type": "grasp"},
         "command_position": 255., "contact_position_range": [80., 180.]},
        {"segment_id": "must_not_lift", "type": "move_pose"},
    ]}], "evidence": [], "checks": {}}
    result = execute(program(), solved, bridge, threading.Event(), lambda _: None, deadline_ms=5000)
    assert bridge.calls == ["close"]
    assert result["failure_code"] == "REKEP_GRASP_CONTACT_INVALID"
    record = result["stages"][0]
    assert record["status"] == "failed"
    assert record["pre_grasp_observation_id"] == snapshot()["observation_id"]
    assert record["segments"][0]["command_position"] == 255.
    assert record["segments"][0]["terminal_result"]["error_code"] == "SUCCESS"
    assert record["segments"][0]["status"] == "failed"
    if observation_fails:
        assert record["observation_error"] == "diagnostic camera timed out"


def test_calibrated_contact_is_only_a_pre_lift_gate():
    from execution import verify_grasp_contact
    verify_grasp_contact({"contact_position_range": [80., 180.]},
        {"position": 125., "stalled": True, "reached_goal": False})


class SequencedBridge(Bridge):
    """Records observations among motion calls; returns queued keypoints."""

    def __init__(self, positions):
        super().__init__()
        self.positions = list(positions)

    def observe(self):
        self.calls.append("observe")
        value = snapshot()
        value["keypoints"][0]["position_m"] = self.positions.pop(0)
        return value


def _release_stage(value):
    return {
        "stages": [{
            "stage_id": "place",
            "segments": [
                {"segment_id": "place.move.0", "type": "move_pose", "target_pose": {"position_m": [0.0, -0.41, 0.07], "quaternion_xyzw": [1, 0, 0, 0]}, "deadline_ms": 1000},
                {"segment_id": "place.event.0", "type": "gripper", "event": {"type": "release", "object_id": "object_000"}, "command_position": 0.0, "deadline_ms": 1000},
            ],
            "constraints": value["stages"][1],
            "solver_evidence": [],
        }],
        "evidence": [],
        "checks": {"workspace": True},
    }


def test_release_stage_is_verified_on_the_held_object_before_opening(tmp_path, monkeypatch):
    monkeypatch.setenv("REKEP_RUNTIME_ROOT", str(tmp_path))
    value = program()
    # Inside the region while held; after release it settles outside the
    # subgoal tolerance, which the subgoal never constrained.
    bridge = SequencedBridge([[0.0, -0.41, 0.07], [0.04, -0.41, 0.03]])
    result = execute(value, _release_stage(value), bridge, threading.Event(), lambda _: None, deadline_ms=5000)
    assert result["status"] == "succeeded"
    assert bridge.calls == ["place.move.0", "observe", "place.event.0", "observe"]
    record = result["stages"][0]
    assert record["constraint_verification"] == "verified_before_release"
    assert record["pre_release_observation_id"] and record["observation_id"]
    assert all(item["violation"] == 0.0 for item in record["constraint_evidence"])


def test_failed_pre_release_verification_keeps_the_object_held(tmp_path, monkeypatch):
    monkeypatch.setenv("REKEP_RUNTIME_ROOT", str(tmp_path))
    value = program()
    bridge = SequencedBridge([[0.2, -0.41, 0.07]])
    result = execute(value, _release_stage(value), bridge, threading.Event(), lambda _: None, deadline_ms=5000)
    assert result["status"] == "failed"
    assert result["failure_code"] == "REKEP_POST_STAGE_CONSTRAINT_VIOLATED"
    assert bridge.calls == ["place.move.0", "observe"]  # no release event sent
