from __future__ import annotations

import math

import pytest

from constraint_evaluator import EvaluationContext, evaluate
from execution import solve_program
from helpers import program, snapshot
from rekep_core.contracts import ContractError
from subgoal_solver import solve_subgoal
from path_solver import solve_path


CONFIG = {
    "workspace_m": {"x": [-0.60, 0.60], "y": [-0.80, 0.20], "z": [0.02, 0.75]},
    "default_end_effector_pose": {"position_m": [0.35, -0.20, 0.35], "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0]},
    "max_segment_length_m": 0.45,
    "max_cartesian_step_m": 0.05,
    "transport_clearance_m": 0.12,
    "path_samples_per_segment": 9,
    "move_deadline_ms": 15000,
    "inter_segment_settle_ms": 100,
    "gripper_deadline_ms": 8000,
}


def test_solver_emits_checked_serial_segments():
    solved = solve_program(program(), snapshot(), CONFIG, lambda _: None)
    segments = [segment for stage in solved["stages"] for segment in stage["segments"]]
    assert segments
    assert all(segment["type"] in {"move_pose", "gripper"} for segment in segments)
    assert solved["checks"]["workspace"] is True
    assert all(item["violation"] <= 1e-6 for item in solved["evidence"])
    assert all(segment.get("settle_ms") == 100 for segment in segments if segment["type"] == "move_pose")


def test_solver_subdivides_cartesian_moves_for_incremental_ik():
    solved = solve_program(program(), snapshot(), CONFIG, lambda _: None)
    previous = CONFIG["default_end_effector_pose"]["position_m"]
    for stage in solved["stages"]:
        for segment in stage["segments"]:
            if segment["type"] != "move_pose":
                continue
            target = segment["target_pose"]["position_m"]
            assert __import__("math").dist(previous, target) <= CONFIG["max_cartesian_step_m"] + 1e-9
            previous = target


def test_clearance_constraint_preserves_unconstrained_xy():
    seed = [0.22, -0.36, 0.05]
    constraint = {
        "constraint_id": "vertical_clearance",
        "expression": {
            "op": "dot",
            "left": {
                "op": "sub",
                "left": {"op": "point", "keypoint_id": "$ee"},
                "right": {"op": "region_center", "region_id": "region_000"},
            },
            "right": {"op": "vector", "value": [0.0, 0.0, 1.0]},
        },
        "relation": "ge",
        "target": 0.10,
        "tolerance": 0.002,
    }

    endpoint, _ = solve_subgoal(
        [constraint], snapshot(), seed, [1.0, 0.0, 0.0, 0.0], None, None, CONFIG["workspace_m"]
    )

    assert endpoint[:2] == seed[:2]
    assert endpoint[2] > seed[2]
    # Region center z=.07, target clearance .10 with .002 tolerance. The
    # solver plans to the middle of the tolerance band (.169), keeping .001
    # margin, without a coarse 10 cm overshoot.
    assert endpoint[2] == pytest.approx(0.169, abs=2e-6)


def test_post_stage_missing_keypoint_is_a_structured_contract_error():
    context = EvaluationContext(snapshot(), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ContractError, match="missing keypoint 'object_000.kp_02'"):
        evaluate({"op": "point", "keypoint_id": "object_000.kp_02"}, context)


@pytest.mark.parametrize("start,end,expected_max", [
    ([0.2, -0.34, 0.04], [0.2, -0.34, 0.24], 0.24),
    ([0.2, -0.34, 0.24], [-0.1, -0.48, 0.24], 0.24),
    ([-0.1, -0.48, 0.24], [-0.1, -0.48, 0.05], 0.24),
    ([0.2, -0.34, 0.04], [-0.1, -0.48, 0.05], 0.12),
])
def test_held_paths_do_not_repeat_lift_above_required_height(start, end, expected_max):
    points, _ = solve_path(
        start, end, [], snapshot(), [1., 0., 0., 0.], "object_000",
        (start, start), clearance_m=0.12, samples_per_segment=9,
        max_cartesian_step_m=0.025,
    )
    assert points[0] == start
    assert points[-1] == end
    assert max(p[2] for p in points) == pytest.approx(expected_max)
    if start[:2] == end[:2]:
        assert all(min(start[2], end[2]) <= p[2] <= max(start[2], end[2]) for p in points)


def _relative(axis: list[float]) -> dict:
    return {
        "op": "dot",
        "left": {"op": "sub", "left": {"op": "point", "keypoint_id": "object_000.kp_00"}, "right": {"op": "region_center", "region_id": "region_000"}},
        "right": {"op": "vector", "value": axis},
    }


def _transport_case(lift_target: float) -> tuple[dict, dict]:
    # Geometry of Run 0318-run1: a centre keypoint at the cube centre, a
    # region volume whose centre is above the pick height, and a grasp
    # tolerance that leaves the pinch frame offset from the held keypoint.
    value = snapshot()
    value["keypoints"] = [
        {"keypoint_id": "object_000.kp_00", "object_id": "object_000", "position_m": [0.2282797831, -0.3397294044, 0.0206237110]},
        {"keypoint_id": "object_000.kp_01", "object_id": "object_000", "position_m": [0.2282797831, -0.3627294044, 0.0456237110]},
    ]
    value["regions"] = [{"region_id": "region_000", "bounds_min_m": [-0.1528618452, -0.5247473403, -0.0040509823], "bounds_max_m": [-0.0573232247, -0.4372458500, 0.1019488326]}]
    clearance = {"constraint_id": "maintain_transport_clearance", "expression": _relative([0, 0, 1]), "relation": "ge", "target": 0.12, "tolerance": 0.002}
    reach = {"op": "distance", "left": {"op": "point", "keypoint_id": "$ee"}, "right": {"op": "point", "keypoint_id": "object_000.kp_00"}}
    stages = [
        {"stage_id": "grasp", "subgoal_constraints": [{"constraint_id": "reach", "expression": reach, "relation": "le", "target": 0, "tolerance": 0.005}], "path_constraints": [], "events": [{"type": "grasp", "object_id": "object_000"}]},
        {"stage_id": "clearance", "subgoal_constraints": [{"constraint_id": "lift", "expression": _relative([0, 0, 1]), "relation": "ge", "target": lift_target, "tolerance": 0.002}], "path_constraints": [], "events": []},
        {"stage_id": "transfer", "subgoal_constraints": [
            {"constraint_id": "x", "expression": _relative([1, 0, 0]), "relation": "eq", "target": 0, "tolerance": 0.003},
            {"constraint_id": "y", "expression": _relative([0, 1, 0]), "relation": "eq", "target": 0, "tolerance": 0.003},
        ], "path_constraints": [clearance], "events": []},
    ]
    return {"stages": stages}, value


def _replay(value: dict, program_value: dict, solved: dict) -> dict:
    # Rebuild each stage's Cartesian path and held state from the emitted
    # segments, independently of the solver's own evidence.
    ee, held, anchor, result = list(CONFIG["default_end_effector_pose"]["position_m"]), None, None, {}
    for stage, out in zip(program_value["stages"], solved["stages"], strict=True):
        points = [ee] + [item["target_pose"]["position_m"] for item in out["segments"] if item["type"] == "move_pose"]
        result[stage["stage_id"]] = (points, held, anchor)
        for event in stage["events"]:
            if event["type"] == "grasp":
                held, anchor = event["object_id"], (list(points[-1]), list(value["keypoints"][0]["position_m"]))
            else:
                held, anchor = None, None
        ee = points[-1]
    return result


def _clearance(value: dict, point: list[float], held: str | None, anchor: tuple | None) -> float:
    return evaluate(_relative([0, 0, 1]), EvaluationContext(value, point, [1.0, 0.0, 0.0, 0.0], held, anchor))


def test_transfer_keeps_region_relative_clearance_with_offset_grasp_anchor():
    program_value, value = _transport_case(lift_target=0.12)
    solved = solve_program(program_value, value, {**CONFIG, "max_cartesian_step_m": 0.025}, lambda _: None)
    stages = _replay(value, program_value, solved)
    grasp_end = stages["grasp"][0][-1]
    offset = [value["keypoints"][0]["position_m"][axis] - grasp_end[axis] for axis in range(3)]
    # The grasp is accepted inside its tolerance, so the anchor is not zero.
    assert math.dist(offset, [0, 0, 0]) > 1e-3
    points, held, anchor = stages["transfer"]
    assert held == "object_000"
    assert points[0][:2] != points[-1][:2]
    for left, right in zip(points, points[1:]):
        for index in range(9):
            alpha = index / 8
            sample = [(1 - alpha) * left[axis] + alpha * right[axis] for axis in range(3)]
            assert _clearance(value, sample, held, anchor) >= 0.118
    # The endpoint is part of the path; it must not trade clearance for the
    # pinch-frame offset of the held keypoint.
    assert _clearance(value, points[-1], held, anchor) >= 0.119 - 1e-6
    assert all(item["violation"] <= 1e-6 for item in solved["evidence"])


def test_contracted_subgoal_satisfies_next_stage_path_start():
    # The lift subgoal alone allows .098 m; the next stage requires .118 m
    # along its whole path, including its start, which is this endpoint.
    program_value, value = _transport_case(lift_target=0.10)
    solved = solve_program(program_value, value, {**CONFIG, "max_cartesian_step_m": 0.025}, lambda _: None)
    points, held, anchor = _replay(value, program_value, solved)["clearance"]
    assert _clearance(value, points[-1], held, anchor) >= 0.119 - 1e-6
    lookahead = [item for item in solved["stages"][1]["solver_evidence"] if item["role"] == "next_path_start"]
    assert [item["constraint_id"] for item in lookahead] == ["maintain_transport_clearance"]
    transfer_path = [item for item in solved["stages"][2]["solver_evidence"] if item["role"] == "path"]
    assert transfer_path and all(item["violation"] == 0.0 for item in transfer_path)


def test_held_subgoal_seeds_with_the_grasp_anchor_offset():
    _, value = _transport_case(lift_target=0.12)
    kp = value["keypoints"][0]["position_m"]
    anchor_ee = [kp[0] + 0.0016, kp[1] + 0.0018, kp[2] + 0.0044]
    seed = [anchor_ee[0], anchor_ee[1], anchor_ee[2] + 0.15]
    x = {"constraint_id": "x", "expression": _relative([1, 0, 0]), "relation": "eq", "target": 0, "tolerance": 0.003}
    y = {"constraint_id": "y", "expression": _relative([0, 1, 0]), "relation": "eq", "target": 0, "tolerance": 0.003}
    endpoint, _ = solve_subgoal([x, y], value, seed, [1.0, 0.0, 0.0, 0.0], "object_000", (anchor_ee, kp), CONFIG["workspace_m"])
    # Only x/y are constrained; the transfer must not drop by the held offset.
    assert endpoint[2] == pytest.approx(seed[2], abs=1e-9)


def test_grasp_aligns_above_object_before_vertical_descent():
    value, scene = _transport_case(lift_target=0.12)
    solved = solve_program(value, scene, {**CONFIG, "pregrasp_approach_height_m": .10,
        "grasp_contact_position_range": [80., 180.]}, lambda _: None)
    grasp = solved["stages"][0]["segments"]
    points = [CONFIG["default_end_effector_pose"]["position_m"]] + [
        s["target_pose"]["position_m"] for s in grasp if s["type"] == "move_pose"]
    end = points[-1]
    for left, right in zip(points, points[1:]):
        if min(left[2], right[2]) < end[2] + .10 - 1e-9:
            assert left[:2] == pytest.approx(end[:2])
            assert right[:2] == pytest.approx(end[:2])
    assert grasp[-1]["contact_position_range"] == [80., 180.]


def test_moves_while_holding_use_the_held_trajectory_scales():
    config = {**CONFIG, "velocity_scale": 0.4, "acceleration_scale": 0.4, "held_velocity_scale": 0.1, "held_acceleration_scale": 0.05}
    solved = solve_program(program(), snapshot(), config, lambda _: None)
    by_stage = {stage["stage_id"]: [s for s in stage["segments"] if s["type"] == "move_pose"] for stage in solved["stages"]}
    assert by_stage["grasp"] and all((s["velocity_scale"], s["acceleration_scale"]) == (0.4, 0.4) for s in by_stage["grasp"])
    assert by_stage["place"] and all((s["velocity_scale"], s["acceleration_scale"]) == (0.1, 0.05) for s in by_stage["place"])


def test_move_pose_goal_carries_segment_trajectory_scales():
    from execution import ChildActionBridge

    bridge = ChildActionBridge(lambda *_args, **_kwargs: None)
    captured = {}
    bridge._run_child = lambda **kwargs: captured.update(kwargs) or {}
    segment = {"target_pose": {"position_m": [0.2, -0.4, 0.2], "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0]}, "deadline_ms": 1000, "velocity_scale": 0.1, "acceleration_scale": 0.05}
    bridge.move_pose(segment, __import__("threading").Event())
    assert (captured["goal"].velocity_scale, captured["goal"].acceleration_scale) == (0.1, 0.05)
    del segment["velocity_scale"], segment["acceleration_scale"]
    bridge.move_pose(segment, __import__("threading").Event())
    assert (captured["goal"].velocity_scale, captured["goal"].acceleration_scale) == (0.4, 0.4)


def test_continuous_paths_blend_straight_runs_and_stop_only_at_corners():
    from execution import _straight_runs

    waypoints = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.025], [0.0, 0.0, 0.05], [0.025, 0.0, 0.05], [0.05, 0.0, 0.05], [0.05, 0.0, 0.05], [0.05, 0.0, 0.025]]
    assert _straight_runs(waypoints) == [
        [[0.0, 0.0, 0.025], [0.0, 0.0, 0.05]],
        [[0.025, 0.0, 0.05], [0.05, 0.0, 0.05]],
        [[0.05, 0.0, 0.025]],
    ]
    config = {**CONFIG, "continuous_paths": True, "max_cartesian_step_m": 0.025}
    solved = solve_program(program(), snapshot(), config, lambda _: None)
    stepwise = solve_program(program(), snapshot(), {**config, "continuous_paths": False}, lambda _: None)
    for blended, single in zip(solved["stages"], stepwise["stages"], strict=True):
        moves = [s for s in blended["segments"] if s["type"] == "move_pose"]
        steps = [s for s in single["segments"] if s["type"] == "move_pose"]
        # Same Cartesian waypoints, far fewer stops.
        flattened = [p for s in moves for p in (*s.get("via_positions", []), s["target_pose"]["position_m"])]
        assert flattened == [s["target_pose"]["position_m"] for s in steps]
        assert len(moves) < len(steps)
        assert all(s["deadline_ms"] == CONFIG["move_deadline_ms"] * (1 + len(s.get("via_positions", []))) for s in moves)


def test_blended_move_sends_via_poses_as_goal_metadata():
    import json
    from execution import ChildActionBridge

    bridge = ChildActionBridge(lambda *_args, **_kwargs: None)
    captured = {}
    bridge._run_child = lambda **kwargs: captured.update(kwargs) or {}
    segment = {"target_pose": {"position_m": [0.2, -0.4, 0.2], "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0]}, "deadline_ms": 2000,
               "via_positions": [[0.2, -0.4, 0.15], [0.2, -0.4, 0.175]]}
    bridge.move_pose(segment, __import__("threading").Event())
    assert json.loads(captured["metadata"]["via_poses"]) == [[0.2, -0.4, 0.15, 1.0, 0.0, 0.0, 0.0], [0.2, -0.4, 0.175, 1.0, 0.0, 0.0, 0.0]]
    del segment["via_positions"]
    bridge.move_pose(segment, __import__("threading").Event())
    assert captured["metadata"] is None
