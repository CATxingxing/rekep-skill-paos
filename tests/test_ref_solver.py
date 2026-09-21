"""R3: reference sub-goal / path solvers run and are reproducible under a fixed RNG.

The reference IK solver depends on OmniGibson/Lula, which is not installed here,
so we inject a deterministic IK stub. The solver *algorithms* are the reference
ones (imported verbatim); only the IK backend is swapped, mirroring how the
MuJoCo env will supply its own IK.
"""

from __future__ import annotations

import numpy as np
import pytest

from rekep_sim import refimpl

refimpl.ensure_reference_on_path()

import path_solver as path_solver_mod  # noqa: E402
import subgoal_solver as subgoal_solver_mod  # noqa: E402

BOUNDS_MIN = [-0.45, -0.75, 0.698]
BOUNDS_MAX = [0.10, 0.60, 1.2]


class IKStub:
    """Minimal stand-in for the reference Lula IK result."""

    def __init__(self, dof: int = 8):
        self.dof = dof

    def solve(
        self,
        target_pose_homo,
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        position_weight=1.0,
        orientation_weight=0.05,
        max_iterations=150,
        initial_joint_pos=None,
    ):
        class _Result:
            success = True
            num_descents = 1
            position_error = 0.0
            orientation_error = 0.0

        result = _Result()
        result.cspace_position = np.zeros(self.dof)
        return result


def _subgoal_config():
    return {
        "bounds_min": BOUNDS_MIN,
        "bounds_max": BOUNDS_MAX,
        "sampling_maxfun": 200,
        "max_collision_points": 32,
        "constraint_tolerance": 1e-4,
        "minimizer_options": {"maxiter": 50},
        "grasp_axis_local": [1.0, 0.0, 0.0],
        "grasp_preferred_world_dir": [0.0, 0.0, -1.0],
    }


def _path_config():
    cfg = _subgoal_config()
    cfg.update(
        {
            "opt_pos_step_size": 0.20,
            "opt_rot_step_size": 0.78,
            "opt_interpolate_pos_step_size": 0.05,
            "opt_interpolate_rot_step_size": 0.20,
        }
    )
    return cfg


def _inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    ee_pose = np.array([0.0, -0.1, 0.95, 0.0, 0.0, 0.0, 1.0])
    keypoints = np.array(
        [[0.0, -0.1, 0.95], [-0.2, -0.3, 0.85], [0.0, 0.2, 0.85]], dtype=float
    )
    mask = np.array([True, False, False])
    sdf = np.ones((8, 8, 8), dtype=float) * 0.5
    collision = rng.random((20, 3)) * 0.1 + np.array([0.0, -0.1, 0.9])
    reset_joint_pos = np.zeros(8)
    return ee_pose, keypoints, mask, sdf, collision, reset_joint_pos


def _reseed(seed: int = 0) -> None:
    np.random.seed(seed)


@pytest.mark.slow
def test_subgoal_solver_runs_and_is_deterministic():
    ee_pose, keypoints, mask, sdf, collision, reset = _inputs()
    solver = subgoal_solver_mod.SubgoalSolver(_subgoal_config(), IKStub(), reset)

    def goal(ee, kps):
        return np.linalg.norm(ee - kps[1])

    _reseed(0)
    sol1, dbg1 = solver.solve(
        ee_pose, keypoints, mask, [goal], [], sdf, collision, False, reset, from_scratch=True
    )
    _reseed(0)
    sol2, dbg2 = solver.solve(
        ee_pose, keypoints, mask, [goal], [], sdf, collision, False, reset, from_scratch=True
    )
    assert sol1.shape == (7,)
    assert dbg1["type"] == "subgoal_solver"
    np.testing.assert_allclose(sol1, sol2, atol=1e-9)


@pytest.mark.slow
def test_path_solver_runs_and_is_deterministic():
    _, keypoints, mask, sdf, collision, reset = _inputs()
    solver = path_solver_mod.PathSolver(_path_config(), IKStub(), reset)
    start = np.array([0.0, -0.1, 0.95, 0.0, 0.0, 0.0, 1.0])
    end = np.array([0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 1.0])

    _reseed(0)
    path1, dbg1 = solver.solve(start, end, keypoints, mask, [], sdf, collision, None, from_scratch=True)
    _reseed(0)
    path2, dbg2 = solver.solve(start, end, keypoints, mask, [], sdf, collision, None, from_scratch=True)
    assert path1.ndim == 2 and path1.shape[1] == 7
    assert dbg1["type"] == "path_solver"
    np.testing.assert_allclose(path1, path2, atol=1e-9)
