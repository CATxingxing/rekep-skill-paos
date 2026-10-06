from __future__ import annotations

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
    # Region center z=.07, target clearance .10 with .002 tolerance:
    # the nearest feasible height is .168, without a coarse 10 cm overshoot.
    assert endpoint[2] == pytest.approx(0.168, abs=2e-6)


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
