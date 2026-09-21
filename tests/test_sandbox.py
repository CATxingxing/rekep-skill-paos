"""Tests for the AST sandbox that executes VLM-generated ReKep constraints."""

from __future__ import annotations

import numpy as np
import pytest

from rekep_sim.sandbox import (
    ConstraintSecurityError,
    ConstraintSyntaxError,
    evaluate,
    load_functions,
)

# A canonical constraint program in the shape the reference VLM prompt requests.
VALID_PROGRAM = """
def stage1_subgoal_constraint1(end_effector, keypoints):
    "align end-effector with keypoint 1"
    return np.linalg.norm(end_effector - keypoints[1])

def stage1_path_constraint1(end_effector, keypoints):
    "keep grasper attached to keypoint 0"
    return get_grasping_cost_by_keypoint_idx(0)
"""


def test_valid_program_loads_and_evaluates():
    functions = load_functions(VALID_PROGRAM, grasped_keypoints=[0])
    assert len(functions) == 2
    ee = np.zeros(3)
    kps = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    costs, reasons = evaluate(functions, ee, kps)
    # subgoal: ||0-1|| = 1.0 ; path: grasped(0) -> 0.0
    assert costs[0] == pytest.approx(1.0)
    assert costs[1] == pytest.approx(0.0)
    assert len(reasons) == 1 and "violated" in reasons[0]


@pytest.mark.parametrize(
    "bad",
    [
        "import os\ndef f(end_effector, keypoints):\n    return 0\n",
        "def f(end_effector, keypoints):\n    return open('/etc/passwd')\n",
        "def f(end_effector, keypoints):\n    return end_effector.__class__\n",
        "def f(end_effector, keypoints):\n    if True:\n        return 0\n",
        "def f(end_effector, keypoints):\n    while True:\n        pass\n",
        "def f(end_effector, keypoints):\n    return eval('1+1')\n",
        "def f(end_effector, keypoints):\n    return [x for x in range(3)]\n",
        "def f(end_effector, keypoints):\n    return lambda x: x\n",
        "def f(end_effector, keypoints):\n    return f(end_effector, keypoints)\n",
        "def f(end_effector, keypoints):\n    return getattr(end_effector, 'shape')\n",
        "x = 1\n",  # no functions
    ],
)
def test_forbidden_programs_are_rejected(bad):
    with pytest.raises((ConstraintSecurityError, ConstraintSyntaxError)):
        load_functions(bad)


def test_invalid_python_is_rejected():
    with pytest.raises(ConstraintSyntaxError):
        load_functions("def f(:\n")


def test_empty_is_rejected():
    with pytest.raises(ConstraintSyntaxError):
        load_functions("   ")


def test_np_and_math_calls_allowed():
    program = """
def f(end_effector, keypoints):
    a = np.linalg.norm(keypoints[2] - keypoints[1])
    b = math.sqrt(np.sum((keypoints[2] - keypoints[1]) ** 2))
    return a + b
"""
    functions = load_functions(program)
    ee = np.zeros(3)
    kps = np.zeros((3, 3))
    kps[2] = [0.0, 3.0, 4.0]
    costs, _ = evaluate(functions, ee, kps)
    assert costs[0] == pytest.approx(10.0)
