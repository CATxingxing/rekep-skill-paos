from __future__ import annotations

import pytest

from helpers import snapshot
from rekep_core.contracts import ContractError, validate_snapshot
from perception import Perception


def test_snapshot_content_digest_is_required():
    value = snapshot()
    assert validate_snapshot(value) is value
    value["keypoints"][0]["position_m"][0] += 0.01
    with pytest.raises(ContractError, match="observation_id"):
        validate_snapshot(value)


def test_snapshot_rejects_unknown_object_reference():
    value = snapshot()
    value["keypoints"][0]["object_id"] = "invented"
    with pytest.raises(ContractError, match="unknown object"):
        validate_snapshot(value)


def test_snapshot_timestamp_is_json_interoperable_decimal_string():
    value = snapshot()
    value["timestamp_ns"] = 1_700_000_000_000_000_000
    with pytest.raises(ContractError, match="decimal string"):
        validate_snapshot(value)


def test_mask_point_falls_back_to_nearest_valid_depth():
    import numpy as np

    perception = object.__new__(Perception)
    perception.config = {
        "camera": {
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.0,
            "cy": 0.0,
            "depth_min_m": 0.02,
            "depth_max_m": 2.0,
            "depth_unit_scale": 1.0,
            "base_from_camera": np.eye(4).tolist(),
        }
    }
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 2] = 0.5
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1:3] = True
    u, v, point = perception._mask_point(1, 1, depth, mask)
    assert (u, v) == (2, 1)
    assert point == [1.0, 0.5, 0.5]


def test_single_object_track_survives_a_manipulation_sized_jump():
    perception = object.__new__(Perception)
    perception.config = {"tracker": {"max_reassociation_distance_m": 0.08}}
    perception._tracks = {"object_000": [0.2, -0.36, 0.04]}
    perception._next_track = 1

    assert perception._track_positions([[0.2, -0.36, 0.22]]) == ["object_000"]
    assert perception._tracks["object_000"] == [0.2, -0.36, 0.22]


def test_multi_object_track_assignment_is_one_to_one():
    perception = object.__new__(Perception)
    perception.config = {"tracker": {"max_reassociation_distance_m": 0.08}}
    perception._tracks = {
        "object_000": [0.0, 0.0, 0.0],
        "object_001": [0.1, 0.0, 0.0],
    }
    perception._next_track = 2

    identifiers = perception._track_positions(
        [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]
    )

    assert identifiers == ["object_000", "object_001"]
    assert len(set(identifiers)) == 2


def test_object_center_estimate_applies_profile_calibration():
    perception = object.__new__(Perception)
    perception.config = {
        "keypoints": {"object_center_offset_m": [0.0, 0.023, 0.0]}
    }

    assert perception._object_center_estimate([0.228, -0.363, 0.046]) == pytest.approx(
        [0.228, -0.340, 0.046]
    )


def test_object_center_estimate_rejects_non_finite_calibration():
    perception = object.__new__(Perception)
    perception.config = {
        "keypoints": {"object_center_offset_m": [0.0, float("nan"), 0.0]}
    }

    with pytest.raises(RuntimeError, match="finite 3-vector"):
        perception._object_center_estimate([0.0, 0.0, 0.0])
