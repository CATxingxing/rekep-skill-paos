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


def _tracker(tracks):
    perception = object.__new__(Perception)
    perception.config = {"tracker": {"max_reassociation_distance_m": 0.08}}
    perception._tracks = dict(tracks)
    perception._next_track = len(tracks)
    return perception


def test_single_object_track_survives_a_manipulation_sized_jump():
    perception = _tracker({"object_000": ("red", [0.2, -0.36, 0.04])})
    assert perception._track_positions([("red", [0.2, -0.36, 0.22])]) == ["object_000"]
    assert perception._tracks["object_000"] == ("red", [0.2, -0.36, 0.22])


def test_unique_color_keeps_identity_across_a_transport():
    perception = _tracker({"object_000": ("red", [0.2, -0.36, 0.04]), "object_001": ("blue", [0.0, -0.4, 0.04])})
    # The red object was carried 30 cm, past the blue one's old position.
    assert perception._track_positions([("blue", [0.0, -0.4, 0.04]), ("red", [-0.05, -0.45, 0.09])]) == ["object_001", "object_000"]


def test_multi_object_track_assignment_is_one_to_one():
    perception = _tracker({"object_000": ("red", [0.0, 0.0, 0.0]), "object_001": ("red", [0.1, 0.0, 0.0])})
    identifiers = perception._track_positions([("red", [0.01, 0.0, 0.0]), ("red", [0.02, 0.0, 0.0])])
    assert identifiers == ["object_000", "object_001"]
    # A different color never inherits a track.
    assert perception._track_positions([("blue", [0.0, 0.0, 0.0])]) == ["object_002"]


def _flat_camera(perception, size):
    import numpy as np

    perception.config.setdefault("camera", {}).update({
        "fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0, "depth_min_m": 0.02, "depth_max_m": 2.0,
        "base_from_camera": np.eye(4).tolist()})
    depth = np.full(size, 0.5)
    return perception._base_cloud(depth)


PALETTE = {
    "backend": "sim_color_palette", "minimum_component_pixels": 4, "min_saturation": 45,
    "min_object_value": 60, "max_hue_distance_deg": 18,
    "object_colors": {"red": [0.85, 0.12, 0.12], "orange": [0.95, 0.5, 0.08], "yellow": [0.95, 0.85, 0.12]},
    "region_colors": {"green": [0.10, 0.85, 0.20]},
}


def test_region_mask_keeps_the_shadowed_target_zone():
    import numpy as np
    perception = object.__new__(Perception)
    perception.config = {"segmentation": dict(PALETTE)}
    rgb = np.full((6, 12, 3), 128, dtype=np.uint8)
    rgb[1:5, 1:5] = (62, 255, 124)
    rgb[1:5, 5:7] = (9, 73, 17)  # the same zone under the arm's shadow
    rgb[1:5, 9:11] = (230, 89, 25)
    cloud, valid = _flat_camera(perception, rgb.shape[:2])
    objects, regions = perception._segment(rgb, cloud, valid)
    assert len(regions) == 1 and regions[0][0].sum() == 24 and regions[0][1] == "green"
    assert regions[0][0][1:5, 1:7].all()
    assert len(objects) == 1 and objects[0][0][1:5, 9:11].all() and objects[0][0].sum() == 8 and objects[0][1] == "orange"


def test_clipped_top_face_takes_the_color_of_its_object():
    """A lit orange top renders with a yellow hue; it joins the orange object."""
    import numpy as np
    perception = object.__new__(Perception)
    perception.config = {"segmentation": dict(PALETTE)}
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    rgb[1:4, 1:7] = (255, 255, 40)   # clipped top face, yellow-looking
    rgb[4:7, 1:7] = (200, 100, 20)   # orange side
    cloud, valid = _flat_camera(perception, rgb.shape[:2])
    objects, _ = perception._segment(rgb, cloud, valid)
    assert [(mask.sum(), label) for mask, label in objects] == [(36, "orange")]


def test_touching_objects_of_two_colors_stay_separate():
    import numpy as np
    perception = object.__new__(Perception)
    perception.config = {"segmentation": dict(PALETTE)}
    rgb = np.full((10, 6, 3), 128, dtype=np.uint8)
    rgb[1:5, 1:5] = (220, 190, 30)   # yellow block stacked on ...
    rgb[5:9, 1:5] = (200, 30, 30)    # ... a red block
    cloud, valid = _flat_camera(perception, rgb.shape[:2])
    objects, _ = perception._segment(rgb, cloud, valid)
    assert sorted((label, int(mask.sum())) for mask, label in objects) == [("red", 16), ("yellow", 16)]


def test_region_cells_accumulate_and_ignore_off_surface_edge_pixels():
    import numpy as np
    perception = object.__new__(Perception)
    perception.config = {"targets": {"volume_height_m": .1, "surface_height_tolerance_m": .002}, "keypoints": {"grid_cell_m": 0.002}}
    perception._regions = []
    cloud, valid = _flat_camera(perception, (10, 10))
    depth = np.full((10, 10), .5)
    depth[:, 8] = .506  # green-tinted edge pixels whose depth hits the support behind
    cloud, valid = perception._base_cloud(depth)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:9] = True
    perception._update_regions([(mask, "green")], cloud, valid)
    regions, keypoints = perception._region_entries()
    full = (regions[0]["bounds_min_m"], regions[0]["bounds_max_m"])
    assert full[0][:2] == pytest.approx([.01, .01], abs=.002) and full[1][:2] == pytest.approx([.035, .035], abs=.002)
    assert full[1][2] - regions[0]["surface_z_m"] == pytest.approx(.1)
    hidden = mask.copy()
    hidden[3:7, 3:7] = False  # arm or held object hides interior pixels
    perception._update_regions([(hidden, "green")], cloud, valid)
    regions, _ = perception._region_entries()
    assert (regions[0]["bounds_min_m"], regions[0]["bounds_max_m"]) == (full[0], full[1])
    assert len(regions) == 1 and {item["region_id"] for item in keypoints} == {"region_000"}
    assert keypoints[0]["kind"] == "region_surface_center" and "object_id" not in keypoints[0]


def _box_cloud(center, half, yaw=0.0, faces=("top", "+x", "-y"), step=0.002):
    import math
    import numpy as np
    points = []
    hx, hy, hz = half
    xs, ys, zs = (np.arange(-h, h + 1e-9, step) for h in half)
    if "top" in faces:
        points += [(x, y, hz) for x in xs for y in ys]
    if "+x" in faces:
        points += [(hx, y, z) for y in ys for z in zs]
    if "-y" in faces:
        points += [(x, -hy, z) for x in xs for z in zs]
    local = np.asarray(points)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rotation.T + np.asarray(center)


def test_rigid_model_keeps_keypoints_on_a_moved_rotated_partially_hidden_object():
    import numpy as np
    from object_model import RigidModel

    cloud = _box_cloud([0.1, -0.4, 0.02], [0.03, 0.015, 0.025])
    keypoint = {"kind": "top_extreme", "offset": [0.03, 0.0, 0.025], "confidence": 1.0, "description": "end"}
    model = RigidModel(cloud, np.array([0.1, -0.4, 0.02]), [keypoint], {})
    moved = _box_cloud([0.25, -0.30, 0.10], [0.03, 0.015, 0.025], yaw=0.5)
    visible = moved[moved[:, 1] < -0.29]  # a finger hides part of it
    model.track(visible)
    (_, position), = model.keypoint_positions()
    expected = np.array([0.25, -0.30, 0.10]) + np.array([0.03 * np.cos(0.5), 0.03 * np.sin(0.5), 0.025])
    assert np.linalg.norm(np.asarray(position) - expected) < 0.003
    assert abs(model.yaw - 0.5) < 0.03 or abs(abs(model.yaw - 0.5) - np.pi) < 0.03


def test_outline_fit_treats_cells_hidden_under_an_object_as_unknown():
    import math
    import numpy as np
    from object_model import _rot, fit_shape

    cell = 0.003
    grid = np.array([(x, y) for x in np.arange(-0.06, 0.06, cell) + cell / 2 for y in np.arange(-0.06, 0.06, cell) + cell / 2])
    bar = (grid[:, 1] >= 0.03)
    stem = (np.abs(grid[:, 0]) <= 0.015) & (grid[:, 1] < 0.03)
    full = grid[bar | stem]                      # T outline in its own frame
    source = full - full.mean(axis=0)
    yaw, offset = 0.8, np.array([0.12, -0.42])
    rotation = _rot(yaw)[:2, :2]
    covered_by_object = full[:, 1] < 0.0         # an object stands on most of the stem
    target = full[~covered_by_object] @ rotation.T + offset

    def hidden(points):
        local = (np.asarray(points) - offset) @ rotation
        return (local[:, 1] < 0.0) & (np.abs(local[:, 0]) < 0.05)

    tx, ty, fitted_yaw, score = fit_shape(source, target, hidden=hidden)
    assert score >= 0.85
    assert np.linalg.norm(np.array([tx, ty]) - (offset + rotation @ full.mean(axis=0))) < 0.005
    assert abs(math.remainder(fitted_yaw - yaw, math.tau)) < 0.08


def test_registration_estimates_center_size_and_named_keypoints():
    import numpy as np
    perception = object.__new__(Perception)
    perception.config = {"keypoints": {"dinov2_per_object": 0}}
    points = _box_cloud([0.1, -0.4, 0.02], [0.03, 0.015, 0.025])
    shape = (40, 40)
    cloud, valid = _flat_camera(perception, shape)
    mask = np.zeros(shape, dtype=bool)  # no depth patches: falls back to the top quantile
    model = perception._register("object_000", mask, points, cloud, valid, None, None, "red")
    assert model.position == pytest.approx([0.1, -0.4, 0.02], abs=0.003)
    assert model.attributes["size_m"] == pytest.approx([0.06, 0.03, 0.05], abs=0.004)
    assert model.attributes["shape_hint"] == "box"
    kinds = [item["kind"] for item in model.keypoints]
    assert kinds[:3] == ["object_center_estimate", "top_center", "bottom_center"] and kinds.count("top_extreme") == 4
    assert model.keypoints[1]["offset"][2] == pytest.approx(0.025, abs=0.003)
