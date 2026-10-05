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
