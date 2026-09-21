"""R5: reference geometry/utils helpers behave as expected (port sanity)."""

from __future__ import annotations

import numpy as np
import pytest

from rekep_sim import refimpl

refimpl.ensure_reference_on_path()

import transform_utils as T  # noqa: E402
import utils  # noqa: E402


def test_filter_points_by_bounds():
    pts = np.array([[0.0, 0.0, 0.9], [1.0, 1.0, 1.0], [-0.2, -0.3, 0.8]])
    bmin = np.array([-0.45, -0.75, 0.698])
    bmax = np.array([0.10, 0.60, 1.2])
    mask = utils.filter_points_by_bounds(pts, bmin, bmax, strict=True)
    assert mask.tolist() == [True, False, True]


def test_transform_keypoints_moves_only_movable():
    # transform is identity rotation + translation [0,0,1]
    transform = np.eye(4)
    transform[2, 3] = 1.0
    keypoints = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    movable = np.array([False, True])
    out = utils.transform_keypoints(transform, keypoints, movable)
    # only movable (index 1) is transformed; index 0 is left untouched
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [0.0, 0.0, 1.0])


def test_pose_quat_mat_roundtrip():
    quat = T.euler2quat(np.array([0.1, -0.2, 0.3]))
    mat = T.quat2mat(quat)
    quat2 = T.mat2quat(mat)
    # quaternions are equal up to sign
    assert np.allclose(quat, quat2) or np.allclose(quat, -quat2)


def test_linear_interpolation_steps_positive():
    a = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1.0])
    b = np.array([0.0, 0.0, 0.2, 0, 0, 0, 1.0])
    n = utils.get_linear_interpolation_steps(a, b, 0.05, 0.34)
    assert n >= 2
