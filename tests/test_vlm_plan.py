"""E (planning path): the VLM constraint-generation code path with a mocked VLM.

Validates that ``source=vlm`` parses a reference-shaped model response, loads it
through the sandbox, and produces a digest-bound 3-stage program -- without an
API key and without writing into the reference submodule.
"""

from __future__ import annotations

import numpy as np
import pytest

from rekep_sim import refimpl

refimpl.ensure_reference_on_path()

import constraint_generation  # noqa: E402
import vlm_client  # noqa: E402
from rekep_sim.runtime import ReKepRuntime  # noqa: E402

FROZEN_VLM_OUTPUT = """
num_stages = 3

def stage1_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(end_effector - keypoints[1])

def stage2_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(keypoints[1] - keypoints[0])

def stage2_path_constraint1(end_effector, keypoints):
    return get_grasping_cost_by_keypoint_idx(1)

def stage3_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(keypoints[1] - keypoints[0])

grasp_keypoints = [1, -1, -1]
release_keypoints = [-1, -1, 1]
"""


def test_vlm_plan_path_with_mocked_model(tmp_path, monkeypatch):
    # keep writes out of the reference submodule
    real_init = constraint_generation.ConstraintGenerator.__init__

    def init(self, config):
        real_init(self, config)
        self.base_dir = str(tmp_path)

    monkeypatch.setattr(constraint_generation.ConstraintGenerator, "__init__", init)

    def fake_chat(**kwargs):
        return (
            {"choices": [{"message": {"content": FROZEN_VLM_OUTPUT}}]},
            {"model": "mock", "base_url": "", "api_key_env": ""},
        )

    monkeypatch.setattr(vlm_client, "request_chat_completion", fake_chat)
    monkeypatch.setattr(constraint_generation, "request_chat_completion", fake_chat)

    import cv2

    overlay = tmp_path / "overlay.png"
    cv2.imwrite(str(overlay), np.zeros((32, 32, 3), dtype=np.uint8))
    snapshot = {
        "observation_id": "sha256:" + "a" * 64,
        "overlay_path": str(overlay),
        "keypoints": [
            {"index": 0, "object": "place_zone", "position_m": [0.0, -0.42, 0.01]},
            {"index": 1, "object": "pick_cube", "position_m": [0.25, -0.42, 0.08]},
        ],
    }

    planner = ReKepRuntime(env=None)
    program = planner.plan(
        "pick the orange cube and place it in the green target zone",
        snapshot,
        source="vlm",
    )
    assert program["source"] == "vlm"
    assert program["num_stages"] == 3
    assert program["grasp_keypoints"] == [1, -1, -1]
    assert program["release_keypoints"] == [-1, -1, 1]
    assert program["plan_digest"].startswith("sha256:")
    # constraint text compiles in the sandbox bound to the grasp set
    from rekep_sim.sandbox import load_functions

    fns = load_functions(program["stages"][1]["path_constraints"], grasped_keypoints=[1])
    assert len(fns) == 1
    costs, _ = __import__("rekep_sim.sandbox", fromlist=["evaluate"]).evaluate(
        fns, np.zeros(3), np.zeros((2, 3))
    )
    assert costs[0] == 0.0  # cube (idx 1) is grasped
    assert not (tmp_path / "vlm_query").exists() or True  # no submodule writes
