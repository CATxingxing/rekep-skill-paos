from __future__ import annotations

from rekep_core.contracts import PROGRAM_SCHEMA, SNAPSHOT_SCHEMA
from rekep_core.ids import digest


def snapshot() -> dict:
    value = {
        "schema_version": SNAPSHOT_SCHEMA,
        "session_id": "session_test",
        "scene_revision": 2,
        "timestamp_ns": "123456789",
        "frame_id": "nova2_base",
        "rgb_digest": "sha256:" + "1" * 64,
        "depth_digest": "sha256:" + "2" * 64,
        "calibration_digest": "sha256:" + "3" * 64,
        "objects": [{"object_id": "object_000", "manipulable": True}],
        "keypoints": [{"keypoint_id": "object_000.kp_00", "object_id": "object_000", "position_m": [0.2, -0.4, 0.05]}],
        "regions": [{"region_id": "region_000", "bounds_min_m": [-0.05, -0.46, 0.02], "bounds_max_m": [0.05, -0.36, 0.12]}],
        "overlay_ref": "/tmp/observation.jpg",
        "robot_state": {},
        "model": {"name": "dinov2_vits14", "weights_digest": "sha256:" + "4" * 64},
    }
    value["observation_id"] = digest({key: item for key, item in value.items() if key != "overlay_ref"})
    return value


def program(bound_snapshot: dict | None = None) -> dict:
    snap = bound_snapshot or snapshot()
    point = {"op": "point", "keypoint_id": "object_000.kp_00"}
    ee = {"op": "point", "keypoint_id": "$ee"}
    instruction = "place the observed object in the observed region"
    value = {
        "schema_version": PROGRAM_SCHEMA,
        "plan_id": "plan_test",
        "session_id": snap["session_id"],
        "scene_revision": snap["scene_revision"],
        "observation_id": snap["observation_id"],
        "instruction": instruction,
        "instruction_digest": digest(instruction),
        "frame_id": snap["frame_id"],
        "units": {"length": "m", "angle": "rad", "quaternion": "xyzw"},
        "generator": {"mode": "vlm", "provider": "test", "model": "vision-test"},
        "stages": [
            {
                "stage_id": "grasp",
                "depends_on": [],
                "subgoal_constraints": [{"constraint_id": "at-object", "expression": {"op": "distance", "left": ee, "right": point}, "relation": "eq", "target": 0.0, "tolerance": 0.015}],
                "path_constraints": [],
                "events": [{"type": "grasp", "object_id": "object_000"}],
            },
            {
                "stage_id": "place",
                "depends_on": ["grasp"],
                "subgoal_constraints": [{"constraint_id": "inside", "expression": {"op": "inside_region", "point": point, "region_id": "region_000"}, "relation": "le", "target": 0.0, "tolerance": 0.005}],
                "path_constraints": [],
                "events": [{"type": "release", "object_id": "object_000"}],
            },
        ],
    }
    value["plan_digest"] = digest(value)
    return value
