"""R2: reference constraint parsing from a frozen VLM output (no network)."""

from __future__ import annotations

from rekep_sim import refimpl

refimpl.ensure_reference_on_path()

import constraint_generation  # noqa: E402
from rekep_sim.sandbox import load_functions  # noqa: E402

# Frozen output in the exact shape the reference prompt requests.
FROZEN_VLM_OUTPUT = """
num_stages = 2

### stage 1 sub-goal constraints
def stage1_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(end_effector - keypoints[1])

### stage 1 path constraints
def stage1_path_constraint1(end_effector, keypoints):
    return get_grasping_cost_by_keypoint_idx(0)

### stage 2 sub-goal constraints
def stage2_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(keypoints[1] - keypoints[2])

grasp_keypoints = [1, -1]
release_keypoints = [-1, 1]
"""


def _generator() -> "constraint_generation.ConstraintGenerator":
    return constraint_generation.ConstraintGenerator(
        {"model": "gpt-5.4", "temperature": 0.0, "max_tokens": 2048}
    )


def test_parse_metadata():
    meta = _generator()._parse_other_metadata(FROZEN_VLM_OUTPUT)
    assert meta["num_stages"] == 2
    assert meta["grasp_keypoints"] == [1, -1]
    assert meta["release_keypoints"] == [-1, 1]


def test_parse_and_save_grouping(tmp_path):
    _generator()._parse_and_save_constraints(FROZEN_VLM_OUTPUT, str(tmp_path))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "stage1_path_constraints.txt",
        "stage1_subgoal_constraints.txt",
        "stage2_subgoal_constraints.txt",
    ]


def test_saved_constraints_run_in_sandbox(tmp_path):
    _generator()._parse_and_save_constraints(FROZEN_VLM_OUTPUT, str(tmp_path))
    src = (tmp_path / "stage1_path_constraints.txt").read_text(encoding="utf-8")
    fns = load_functions(src, grasped_keypoints=[0])
    assert len(fns) == 1
    import numpy as np

    costs, _ = __import__("rekep_sim.sandbox", fromlist=["evaluate"]).evaluate(
        fns, np.zeros(3), np.zeros((3, 3))
    )
    assert costs[0] == 0.0
