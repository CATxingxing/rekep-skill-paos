from __future__ import annotations

import threading

from execution import execute
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
