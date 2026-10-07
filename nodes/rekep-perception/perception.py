from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from dinov2_runtime import DinoV2
from object_model import RigidModel, extreme_points, fill_ratio, filled_cells, fit_shape, principal_axes, shape_hint
from recorder import VideoRecorder
from rekep_core.contracts import SNAPSHOT_SCHEMA, atomic_json, runtime_root, validate_snapshot
from rekep_core.geometry import transform_point
from rekep_core.ids import digest, timestamp_ns

EXTREME_DESCRIPTIONS = {
    "+major": "outermost top-face point along +major axis",
    "-major": "outermost top-face point along -major axis",
    "+minor": "outermost top-face point along +minor axis",
    "-minor": "outermost top-face point along -minor axis",
}


def decode_text(value: Any) -> str:
    """Text payload from Dora: a one-element string array or raw UTF-8 bytes.

    mujoco_sim publishes sim_status as bytes, which Dora delivers as a UInt8
    array; reading its first element would yield a single byte value.
    """
    import pyarrow as pa

    if isinstance(value, pa.Array):
        if pa.types.is_string(value.type) or pa.types.is_large_string(value.type):
            return str(value[0].as_py())
        return value.to_numpy(zero_copy_only=False).astype("uint8").tobytes().decode("utf-8")
    return bytes(value).decode("utf-8")


def _hue_degrees(rgb: Any) -> Any:
    import numpy as np

    values = np.asarray(rgb, dtype=float)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    high, low = values.max(axis=-1), values.min(axis=-1)
    chroma = np.maximum(high - low, 1e-9)
    hue = np.where(
        high == red, ((green - blue) / chroma) % 6.0,
        np.where(high == green, (blue - red) / chroma + 2.0, (red - green) / chroma + 4.0),
    )
    return hue * 60.0


def palette_labels(rgb: Any, palette: dict[str, list[float]], *, min_saturation: float, min_value: float, max_hue_distance: float) -> Any:
    """Per-pixel palette index (or -1) by nearest hue among saturated pixels."""
    import numpy as np

    names = list(palette)
    image = np.asarray(rgb, dtype=float)
    saturation = image.max(axis=-1) - image.min(axis=-1)
    value = image.max(axis=-1)
    hue = _hue_degrees(image)
    references = _hue_degrees(np.asarray([palette[name] for name in names], dtype=float)[None, :, :])[0]
    distance = np.abs(((hue[..., None] - references[None, None, :]) + 180.0) % 360.0 - 180.0)
    nearest = distance.argmin(axis=-1)
    good = (saturation > min_saturation) & (value > min_value) & (distance.min(axis=-1) <= max_hue_distance)
    return np.where(good, nearest, -1), names


class Perception:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.model = DinoV2(Path(self.config["model_config"]).expanduser())
        self.session_id = f"session_{uuid.uuid4().hex}"
        self.scene_revision = 0
        self._episode: int | None = None
        self._rgb: Any = None
        self._depth: Any = None
        self._robot_state: dict[str, Any] = {}
        self._rgb_at = self._depth_at = 0.0
        self._tracks: dict[str, tuple[str | None, list[float]]] = {}
        self._next_track = 0
        self._models: dict[str, RigidModel] = {}
        self._regions: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._sim_time_s: float | None = None
        recording = self.config.get("recording", {})
        self.recorder = (
            VideoRecorder(
                runtime_root() / "run" / "rekep" / "recordings",
                fps=float(recording.get("fps", 10)),
                retention_s=float(recording.get("retention_s", 3600)),
                segment_s=float(recording.get("segment_s", 600)),
                # trigger "execution": record only while the executor runs execute_task
                # (it holds active-execution.json) plus stop_delay_s afterwards.
                trigger_file=(
                    runtime_root() / "run" / "rekep" / "active-execution.json"
                    if recording.get("trigger", "always") == "execution"
                    else None
                ),
                stop_delay_s=float(recording.get("stop_delay_s", 10)),
            )
            if recording.get("enabled", False)
            else None
        )

    def reset_scene_memory(self) -> None:
        self._tracks.clear()
        self._next_track = 0
        self._models.clear()
        self._regions.clear()

    def update_rgb(self, value: Any) -> None:
        from forge_msgs import Image

        image = Image.from_arrow(value)
        if image.encoding not in {"rgb8", "bgr8"}:
            raise RuntimeError(f"RGB encoding must be rgb8 or bgr8, got {image.encoding}")
        frame = image.to_numpy()
        with self._lock:
            self._rgb = frame[..., ::-1].copy() if image.encoding == "bgr8" else frame.copy()
            self._rgb_at = time.monotonic()
            if self.recorder is not None:
                self.recorder.offer(self._rgb, self.session_id, self._sim_time_s)

    def update_depth(self, value: Any) -> None:
        from forge_msgs import Image

        image = Image.from_arrow(value)
        if image.encoding != "32FC1":
            raise RuntimeError(f"depth encoding must be 32FC1, got {image.encoding}")
        with self._lock:
            self._depth = image.to_numpy().copy()
            self._depth_at = time.monotonic()

    def update_joint_state(self, value: Any) -> None:
        from forge_msgs import JointState

        state = JointState.from_arrow(value)
        with self._lock:
            self._robot_state = {"joint_names": list(state.name), "positions": list(state.position)}

    def update_status(self, status: dict[str, Any]) -> None:
        sim_time = status.get("sim_time")
        if isinstance(sim_time, (int, float)) and not isinstance(sim_time, bool) and math.isfinite(sim_time):
            with self._lock:
                self._sim_time_s = float(sim_time)
        episode = status.get("episode_index")
        if isinstance(episode, bool) or not isinstance(episode, int):
            return
        with self._lock:
            if self._episode is not None and episode != self._episode:
                self.scene_revision += 1
                self.session_id = f"session_{uuid.uuid4().hex}"
                self.reset_scene_memory()
            self._episode = episode

    def readiness(self) -> dict[str, Any]:
        age = float(self.config.get("freshness", {}).get("max_age_ms", 1500)) / 1000.0
        now = time.monotonic()
        return {"rgb": self._rgb is not None and now - self._rgb_at <= age, "depth": self._depth is not None and now - self._depth_at <= age, "robot_state": bool(self._robot_state)}

    # ------------------------------------------------------------------ geometry
    def _point(self, u: int, v: int, depth: Any) -> list[float]:
        camera = self.config["camera"]
        z = float(depth[v, u]) * float(camera.get("depth_unit_scale", 1.0))
        if not math.isfinite(z) or not float(camera["depth_min_m"]) <= z <= float(camera["depth_max_m"]):
            raise RuntimeError(f"invalid depth at ({u}, {v})")
        point = [(u - float(camera["cx"])) * z / float(camera["fx"]), (v - float(camera["cy"])) * z / float(camera["fy"]), z]
        return transform_point(camera["base_from_camera"], point)

    def _mask_point(self, u: int, v: int, depth: Any, mask: Any) -> tuple[int, int, list[float]]:
        """Use the requested pixel, or the nearest valid depth pixel in the same mask."""
        import numpy as np

        camera = self.config["camera"]
        scale = float(camera.get("depth_unit_scale", 1.0))
        scaled = depth.astype(float) * scale
        valid = (
            np.asarray(mask, dtype=bool)
            & np.isfinite(scaled)
            & (scaled >= float(camera["depth_min_m"]))
            & (scaled <= float(camera["depth_max_m"]))
        )
        if not valid.any():
            raise RuntimeError("object mask contains no valid depth pixel")
        if not valid[v, u]:
            rows, columns = np.where(valid)
            nearest = int(np.argmin((columns - int(u)) ** 2 + (rows - int(v)) ** 2))
            u, v = int(columns[nearest]), int(rows[nearest])
        return u, v, self._point(u, v, depth)

    def _pixel(self, point: list[float], shape: tuple[int, ...]) -> list[int]:
        """Image pixel of a base-frame point, clamped to the image."""
        import numpy as np

        c = self.config["camera"]
        transform = np.asarray(c["base_from_camera"], dtype=float)
        x, y, z = transform[:3, :3].T @ (np.asarray(point, dtype=float) - transform[:3, 3])
        u = int(round(x * float(c["fx"]) / z + float(c["cx"])))
        v = int(round(y * float(c["fy"]) / z + float(c["cy"])))
        return [min(max(u, 0), shape[1] - 1), min(max(v, 0), shape[0] - 1)]

    def _hidden(self, xy: Any, height: float, depth: Any, margin: float = 0.01) -> Any:
        """Which points (x, y, height) the camera cannot see: outside the image,
        no valid depth, or something more than ``margin`` nearer along the ray."""
        import numpy as np

        c = self.config["camera"]
        transform = np.asarray(c["base_from_camera"], dtype=float)
        points = np.column_stack([np.asarray(xy, dtype=float), np.full(len(xy), float(height))])
        local = (points - transform[:3, 3]) @ transform[:3, :3]
        z = local[:, 2]
        safe = np.where(z > 1e-6, z, 1.0)
        u = np.round(local[:, 0] * float(c["fx"]) / safe + float(c["cx"])).astype(int)
        v = np.round(local[:, 1] * float(c["fy"]) / safe + float(c["cy"])).astype(int)
        inside = (z > 1e-6) & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
        hidden = np.ones(len(points), dtype=bool)
        seen = depth[v[inside], u[inside]].astype(float) * float(c.get("depth_unit_scale", 1.0))
        valid = np.isfinite(seen) & (seen >= float(c["depth_min_m"])) & (seen <= float(c["depth_max_m"]))
        hidden[inside] = ~valid | (seen < z[inside] - margin)
        return hidden

    def _base_cloud(self, depth: Any) -> tuple[Any, Any]:
        """Base-frame point of every pixel and the mask of in-range depth."""
        import numpy as np

        c = self.config["camera"]
        z = np.asarray(depth, dtype=float) * float(c.get("depth_unit_scale", 1.0))
        valid = np.isfinite(z) & (z >= c["depth_min_m"]) & (z <= c["depth_max_m"])
        v, u = np.indices(z.shape)
        camera_points = np.stack(((u-c["cx"])*z/c["fx"], (v-c["cy"])*z/c["fy"], z), axis=-1)
        transform = np.asarray(c["base_from_camera"])
        return camera_points @ transform[:3, :3].T + transform[:3, 3], valid

    @staticmethod
    def _horizontal(mask: Any, cloud: Any, valid: Any) -> Any:
        """Pixels of the mask whose local depth surface faces straight up."""
        import numpy as np

        valid = np.asarray(mask, dtype=bool) & valid
        horizontal = cloud[1:-1, 2:] - cloud[1:-1, :-2]
        vertical = cloud[2:, 1:-1] - cloud[:-2, 1:-1]
        normal = np.cross(horizontal, vertical)
        norm = np.linalg.norm(normal, axis=-1)
        interior = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
        result = np.zeros_like(valid)
        result[1:-1, 1:-1] = interior & (norm > 1e-10) & (np.abs(normal[..., 2]) >= .98 * norm)
        return result

    @staticmethod
    def _grid_centroid(xy: Any, cell: float) -> Any:
        """Area-weighted centroid: perspective puts more pixels on near surfaces."""
        import numpy as np

        cells = np.unique(np.floor(xy / cell).astype(int), axis=0)
        return (cells.mean(axis=0) + 0.5) * cell

    def _register(self, object_id: str, mask: Any, points: Any, cloud: Any, valid: Any, depth: Any, features: Any, label: str | None) -> RigidModel:
        """First observation of an object: center, size and keypoints in its frame.

        Center: area centroid of the visible top face in x/y, and the middle
        between the top face and the lowest visible point (the support
        contact) in z. Keypoints: center, top and bottom centers, the four
        outermost top-face points along its principal axes, and DINOv2
        feature proposals.
        """
        import numpy as np

        settings = self.config.get("keypoints", {})
        cell = float(settings.get("grid_cell_m", 0.002))
        horizontal = self._horizontal(mask, cloud, valid)
        heights = cloud[..., 2][horizontal]
        if len(heights) >= 10:
            top_z = float(np.quantile(heights, 0.9))
        else:
            top_z = float(np.quantile(points[:, 2], 0.98))
        top = points[np.abs(points[:, 2] - top_z) <= float(settings.get("top_face_tolerance_m", 0.004))]
        if len(top) < 10:
            top = points[points[:, 2] >= top_z - 0.01]
        bottom_z = float(np.quantile(points[:, 2], float(settings.get("bottom_quantile", 0.02))))
        center_xy = self._grid_centroid(top[:, :2], cell)
        center = np.array([center_xy[0], center_xy[1], (top_z + bottom_z) / 2.0])
        axes = principal_axes(top[:, :2])
        ratio, major, minor = fill_ratio(top[:, :2], axes, cell)
        keypoints: list[dict[str, Any]] = [
            {"kind": "object_center_estimate", "offset": [0.0, 0.0, 0.0], "confidence": 1.0, "description": "estimated center of the object volume (grasp point)"},
            {"kind": "top_center", "offset": [0.0, 0.0, top_z - center[2]], "confidence": 1.0, "description": "center of the object's top surface"},
            {"kind": "bottom_center", "offset": [0.0, 0.0, bottom_z - center[2]], "confidence": 1.0, "description": "center of the object's bottom (support contact)"},
        ]
        for name, point in extreme_points(top, center_xy, axes):
            keypoints.append({"kind": "top_extreme", "axis": name, "offset": (point - center).tolist(), "confidence": 1.0, "description": EXTREME_DESCRIPTIONS[name]})
        count = int(settings.get("dinov2_per_object", 0))
        if count > 0 and features is not None:
            rows, columns = np.where(mask)
            fallback = (int(np.median(columns)), int(np.median(rows)))
            for u, v, confidence in self.model.propose(features, mask, count):
                if not mask[v, u]:
                    u, v = fallback
                position = np.asarray(self._mask_point(u, v, depth, mask)[2])
                keypoints.append({"kind": "dinov2_feature", "offset": (position - center).tolist(), "confidence": confidence, "description": "DINOv2 feature-cluster keypoint on the visible surface"})
        attributes = {
            "color": label,
            "shape_hint": shape_hint(ratio, major, minor),
            "size_m": [round(major, 4), round(minor, 4), round(top_z - bottom_z, 4)],
            "major_axis_xy": axes[0].tolist(),
        }
        model = RigidModel(points, center, keypoints, attributes)
        # Top-face outline in the object frame: contact geometry for pushing
        # and the shape matched against region outlines.
        model.cells, model.boundary = filled_cells(top[:, :2] - center[:2], max(cell, 0.003))
        return model

    # ------------------------------------------------------------------ segmentation
    def _segment(self, rgb: Any, cloud: Any, valid: Any) -> tuple[list[tuple[Any, str | None]], list[tuple[Any, str | None]]]:
        """Object and region masks with their palette color names.

        The palette is the profile's list of recognizable colors. Pixels with
        a clipped channel (brightly lit top faces) shift hue -- an orange top
        renders like a yellow one -- so only unclipped saturated pixels vote
        for a color; clipped saturated pixels join the nearest labelled pixel
        of the blob they touch. Same-color pieces that touch in 3D (a finger
        splitting an object) are merged into one object.
        """
        import numpy as np
        from scipy import ndimage
        from scipy.spatial import cKDTree

        segmentation = self.config.get("segmentation", {})
        minimum = int(segmentation.get("minimum_component_pixels", 100))
        if segmentation.get("backend") != "sim_color_palette":
            raise RuntimeError("a supported instance-segmentation backend is required")
        objects_palette = dict(segmentation.get("object_colors") or {})
        regions_palette = dict(segmentation.get("region_colors") or {})
        if not objects_palette or not regions_palette:
            raise RuntimeError("segmentation requires object_colors and region_colors")
        keys = [("object", name) for name in objects_palette] + [("region", name) for name in regions_palette]
        colors = {str(i): (objects_palette if kind == "object" else regions_palette)[name] for i, (kind, name) in enumerate(keys)}
        image = np.asarray(rgb, dtype=float)
        high = image.max(axis=-1)
        saturation = high - image.min(axis=-1)
        clip = float(segmentation.get("clipped_value", 250))
        index, _ = palette_labels(
            rgb, colors,
            min_saturation=float(segmentation.get("min_saturation", 45)),
            min_value=0.0,
            max_hue_distance=float(segmentation.get("max_hue_distance_deg", 18)),
        )
        is_region = np.isin(index, [i for i, (kind, _) in enumerate(keys) if kind == "region"])
        # Regions are identified by hue alone (shadows darken them); objects
        # also need some brightness and an unclipped color.
        voter = (index >= 0) & ~is_region & (high > float(segmentation.get("min_object_value", 60))) & (high < clip)
        clipped = (high >= clip) & (saturation >= float(segmentation.get("min_clipped_saturation", 35))) & (index < 0) | ((high >= clip) & (index >= 0) & ~is_region)
        candidate = voter | clipped
        blobs, _ = ndimage.label(candidate)
        # Directly lit horizontal faces shift hue as well (an orange top face
        # looks yellow), so side faces vote when a blob has any.
        horizontal = ndimage.binary_dilation(self._horizontal(candidate, cloud, valid))
        side_voter = voter & ~horizontal
        with_sides = np.unique(blobs[side_voter])
        voter = side_voter | (voter & ~np.isin(blobs, with_sides))
        # Colors covering only a small share of a blob's votes are rim or
        # specular noise; a blob dominated by one color is one color.
        noise = float(segmentation.get("minority_vote_fraction", 0.15))
        for blob in np.unique(blobs[voter]):
            inside = voter & (blobs == blob)
            values, counts = np.unique(index[inside], return_counts=True)
            minor = values[counts < noise * counts.sum()]
            if len(minor) < len(values):
                voter &= ~(inside & np.isin(index, minor))
        clipped = candidate & ~voter
        labels = np.where(voter, index, -1)
        if voter.any():
            distance, (rows, columns) = ndimage.distance_transform_edt(~voter, return_indices=True)
            nearest = labels[rows, columns]
            joined = clipped & (blobs == blobs[rows, columns]) & (distance <= float(segmentation.get("max_clipped_join_px", 30)))
            labels = np.where(joined, nearest, labels)
        merge = float(segmentation.get("merge_gap_m", 0.01))
        objects, regions = [], []
        for number, (kind, name) in enumerate(keys):
            mask = (index == number) if kind == "region" else (labels == number)
            labelled, count = ndimage.label(mask)
            pieces = []
            for component in range(1, count + 1):
                piece = labelled == component
                if piece.sum() < max(3, minimum // 10):
                    continue
                points = cloud[piece & valid]
                pieces.append([piece, points[:: max(1, len(points) // 400)] if len(points) else None])
            merged: list[list[Any]] = []
            for piece, points in sorted(pieces, key=lambda item: -int(item[0].sum())):
                target = None
                if points is not None and len(points):
                    for group in merged:
                        if group[1] is not None and len(group[1]) and float(cKDTree(group[1]).query(points)[0].min()) <= merge:
                            target = group
                            break
                if target is None:
                    merged.append([piece, points])
                else:
                    target[0] = target[0] | piece
                    target[1] = np.concatenate([target[1], points])
            for piece, _ in merged:
                if piece.sum() >= minimum:
                    (objects if kind == "object" else regions).append((piece, name))
        return objects, regions

    # ------------------------------------------------------------------ tracking
    def _merge_fragments(self, detections: list[tuple[Any, str | None, Any]]) -> list[tuple[Any, str | None, Any]]:
        """Re-join pieces of one tracked object split by an occluder (e.g. the gripper).

        Only when a color has more detections than tracks: same-colored pieces
        near a track are merged while their union still fits within that
        object's registered footprint diagonal, so a separate object of the
        same color is never absorbed.
        """
        import numpy as np

        margin = float(self.config.get("tracker", {}).get("fragment_merge_margin_m", 0.02))
        result = list(detections)
        for track_label in sorted({label for label, _ in self._tracks.values() if label is not None}):
            tracks = [(identifier, old) for identifier, (label, old) in self._tracks.items() if label == track_label and identifier in self._models]
            for identifier, old in tracks:
                indices = [index for index, (_, label, _) in enumerate(result) if label == track_label]
                if len(indices) <= len(tracks):
                    break
                size = self._models[identifier].attributes.get("size_m", [0.0, 0.0])
                limit = math.hypot(float(size[0]), float(size[1])) + margin
                order = sorted(indices, key=lambda index: math.dist(result[index][2].mean(axis=0)[:2], old[:2]))
                group = [order[0]]
                for index in order[1:]:
                    xy = np.concatenate([result[member][2][:, :2] for member in group + [index]])
                    sample = xy[:: max(1, len(xy) // 300)]
                    diameter = float(np.max(np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)))
                    if diameter <= limit:
                        group.append(index)
                if len(group) > 1:
                    mask = np.logical_or.reduce([result[member][0] for member in group])
                    points = np.concatenate([result[member][2] for member in group])
                    first = min(group)
                    result = [(mask, track_label, points) if index == first else item for index, item in enumerate(result) if index == first or index not in group]
        return result

    def _track_positions(self, detections: list[tuple[str | None, list[float]]]) -> list[str]:
        """Associate one capture's detections to tracks without reusing an ID.

        A detection whose color is unique among both the detections and the
        tracks keeps that track however far it moved (a transported object).
        Otherwise a deterministic greedy assignment by distance within the
        reassociation gate is used, so two detections never claim one track.
        """
        if not detections:
            return []
        threshold = float(self.config.get("tracker", {}).get("max_reassociation_distance_m", 0.08))
        assigned: dict[int, str] = {}
        used: set[str] = set()
        labels = [label for label, _ in detections]
        track_labels = [label for label, _ in self._tracks.values()]
        for index, (label, _) in enumerate(detections):
            if label is not None and labels.count(label) == 1 and track_labels.count(label) == 1:
                identifier = next(key for key, (other, _) in self._tracks.items() if other == label)
                assigned[index] = identifier
                used.add(identifier)
        if len(detections) == 1 and len(self._tracks) == 1 and not assigned:
            only = next(iter(self._tracks))
            if self._tracks[only][0] == detections[0][0]:
                assigned[0] = only
                used.add(only)
        candidates = sorted(
            (math.dist(position, old), index, identifier)
            for index, (label, position) in enumerate(detections)
            for identifier, (old_label, old) in self._tracks.items()
            if label == old_label
        )
        for separation, index, identifier in candidates:
            if separation > threshold:
                break
            if index not in assigned and identifier not in used:
                assigned[index] = identifier
                used.add(identifier)
        identifiers = []
        for index, (label, position) in enumerate(detections):
            identifier = assigned.get(index)
            if identifier is None:
                identifier = f"object_{self._next_track:03d}"
                self._next_track += 1
            self._tracks[identifier] = (label, list(position))
            identifiers.append(identifier)
        return identifiers

    def _update_regions(self, region_masks: list[tuple[Any, str | None]], cloud: Any, valid: Any) -> None:
        """Static target regions: union of every observed surface cell.

        Regions do not move, so cells hidden in one capture (the arm, an
        object placed on the region) remain part of it. Only points at the
        median surface height count: green edge pixels can carry the depth
        of the support beneath the region.
        """
        import numpy as np

        targets = self.config.get("targets", {})
        cell = float(self.config.get("keypoints", {}).get("grid_cell_m", 0.002))
        tolerance = float(targets.get("surface_height_tolerance_m", 0.002))
        for mask, label in region_masks:
            points = cloud[np.asarray(mask, dtype=bool) & valid]
            if not len(points):
                continue
            height = float(np.median(points[:, 2]))
            surface = points[np.abs(points[:, 2] - height) <= tolerance]
            if len(surface) < 10:
                continue
            cells = {tuple(item) for item in np.floor(surface[:, :2] / cell).astype(int).tolist()}
            centroid = surface[:, :2].mean(axis=0)
            match = next((region for region in self._regions if region["label"] == label and (region["cells"] & cells or float(np.linalg.norm(region["centroid"] - centroid)) < 0.05)), None)
            if match is None:
                self._regions.append({"label": label, "cells": cells, "heights": [height], "centroid": centroid})
            else:
                match["cells"] |= cells
                match["heights"] = (match["heights"] + [height])[-50:]
                xy = (np.asarray(sorted(match["cells"]), dtype=float) + 0.5) * cell
                match["centroid"] = xy.mean(axis=0)

    @staticmethod
    def _footprint(model: RigidModel, limit: int = 48) -> Any:
        """Current top-face outline points of a registered object (base x-y)."""
        import numpy as np

        boundary = getattr(model, "boundary", None)
        if boundary is None or not len(boundary):
            return np.zeros((0, 2))
        rotation = np.array([[math.cos(model.yaw), -math.sin(model.yaw)], [math.sin(model.yaw), math.cos(model.yaw)]])
        points = boundary @ rotation.T + model.position[:2]
        order = np.argsort(np.arctan2(points[:, 1] - model.position[1], points[:, 0] - model.position[0]))
        points = points[order]
        return points[np.linspace(0, len(points) - 1, min(limit, len(points))).round().astype(int)]

    def _fit_keypoints(self, region_id: str, region_xy: Any, hidden: Any = None) -> list[dict[str, Any]]:
        """Targets for objects whose top-face outline matches this region's outline.

        If an object of the same shape and size as an outline region were
        laid into it, its center and outline extremes would be at these
        points. Area must agree within 25% and the outline fit must cover
        both shapes, so a block on a larger pad gets no target. The outline
        may be partly hidden (down to 30% of the object's area visible), e.g.
        by the object standing on it; regions accumulate, so the fit improves
        as the object moves off. ``hidden`` marks outline points the current
        depth image cannot see; the fit treats them as unknown.
        """
        import numpy as np

        targets = self.config.get("targets", {})
        minimum = float(targets.get("min_shape_fit_score", 0.85))
        cell = max(float(self.config.get("keypoints", {}).get("grid_cell_m", 0.002)), 0.003)
        region_cells, _ = filled_cells(np.asarray(region_xy), cell)
        result = []
        for object_id, model in getattr(self, "_models", {}).items():
            cells = getattr(model, "cells", None)
            ratio = len(region_cells) / max(1, len(cells)) if cells is not None else 0.0
            if cells is None or not len(cells) or not 0.3 <= ratio <= 1.33:
                continue
            tx, ty, yaw, score = fit_shape(np.asarray(cells), region_cells, hidden=hidden)
            if score < minimum:
                continue
            rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            current = dict(model.keypoint_positions_by_index())
            for index, item in enumerate(model.keypoints):
                if item["kind"] not in {"object_center_estimate", "top_center", "top_extreme"}:
                    continue
                xy = rotation @ np.asarray(item["offset"][:2]) + np.array([tx, ty])
                result.append({
                    "keypoint_id": f"{region_id}.fit_{object_id}.kp_{index:02d}", "region_id": region_id,
                    "position_m": [float(xy[0]), float(xy[1]), float(current[index][2])], "confidence": round(score, 3),
                    "kind": "region_fit_target", "for_object": object_id, "for_keypoint": f"{object_id}.kp_{index:02d}",
                    "description": f"where {object_id}.kp_{index:02d} lies when {object_id} is laid exactly into this region's outline",
                })
        return result

    def _region_entries(self, depth: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        import numpy as np

        targets = self.config.get("targets", {})
        cell = float(self.config.get("keypoints", {}).get("grid_cell_m", 0.002))
        regions, keypoints = [], []
        for index, region in enumerate(self._regions):
            region_id = f"region_{index:03d}"
            xy = (np.asarray(sorted(region["cells"]), dtype=float) + 0.5) * cell
            height = float(np.median(region["heights"]))
            lower = [float(xy[:, 0].min() - cell / 2), float(xy[:, 1].min() - cell / 2), height - float(targets.get("surface_height_tolerance_m", 0.002))]
            upper = [float(xy[:, 0].max() + cell / 2), float(xy[:, 1].max() + cell / 2), height + float(targets.get("volume_height_m", 0.10))]
            axes = principal_axes(xy)
            ratio, major, minor = fill_ratio(xy, axes, cell)
            center = xy.mean(axis=0)
            regions.append({
                "region_id": region_id,
                "label": f"{region['label']} target region" if region["label"] else "observed_target_region",
                "color": region["label"],
                "bounds_min_m": lower,
                "bounds_max_m": upper,
                "surface_z_m": height,
                "shape_hint": shape_hint(ratio, major, minor),
                "size_m": [round(major, 4), round(minor, 4)],
                "major_axis_xy": axes[0].tolist(),
            })
            surface = np.column_stack([xy, np.full(len(xy), height)])
            keypoints.append({"keypoint_id": f"{region_id}.kp_00", "region_id": region_id, "position_m": [float(center[0]), float(center[1]), height], "confidence": 1.0, "kind": "region_surface_center", "description": "area centroid of the region surface"})
            for number, (name, point) in enumerate(extreme_points(surface, center, axes), start=1):
                keypoints.append({"keypoint_id": f"{region_id}.kp_{number:02d}", "region_id": region_id, "position_m": [float(v) for v in point], "confidence": 1.0, "kind": "region_extreme", "axis": name, "description": EXTREME_DESCRIPTIONS[name].replace("top-face", "region")})
            hidden = None if depth is None else (lambda points, z=height: self._hidden(points, z, depth))
            keypoints.extend(self._fit_keypoints(region_id, xy, hidden))
        return regions, keypoints

    # ------------------------------------------------------------------ capture
    def capture(self) -> dict[str, Any]:
        with self._lock:
            ready = self.readiness()
            if not ready["rgb"] or not ready["depth"]:
                raise RuntimeError(f"fresh RGB-D is unavailable: {ready}")
            rgb, depth = self._rgb.copy(), self._depth.copy()
            session_id, revision, robot_state = self.session_id, self.scene_revision, dict(self._robot_state)
        import numpy as np
        from PIL import Image as PilImage, ImageDraw
        from scipy import ndimage

        cloud, valid = self._base_cloud(depth)
        object_masks, region_masks = self._segment(rgb, cloud, valid)
        if not object_masks:
            raise RuntimeError("segmentation produced no manipulable object")
        detections = []
        for mask, label in object_masks:
            # Edge pixels mix the object with the surface behind it.
            core = ndimage.binary_erosion(mask) & valid
            if core.sum() < 20:
                core = mask & valid
            points = cloud[core]
            if len(points) < 10:
                continue
            detections.append((mask, label, points))
        detections = self._merge_fragments(detections)
        positions = []
        for mask, label, points in detections:
            positions.append((label, points.mean(axis=0).tolist()))
        object_ids = self._track_positions(positions)
        features = None
        settings = self.config.get("keypoints", {})
        if int(settings.get("dinov2_per_object", 0)) > 0 and any(object_id not in self._models for object_id in object_ids):
            features = self.model.feature_map(rgb)
        overlay = PilImage.fromarray(rgb)
        draw = ImageDraw.Draw(overlay)
        objects, keypoints = [], []
        for (mask, label, points), object_id in zip(detections, object_ids, strict=True):
            model = self._models.get(object_id)
            if model is None:
                model = self._register(object_id, mask, points, cloud, valid, depth, features, label)
                self._models[object_id] = model
            else:
                model.track(points)
            rows, columns = np.where(mask)
            rotation = np.array([[math.cos(model.yaw), -math.sin(model.yaw)], [math.sin(model.yaw), math.cos(model.yaw)]])
            axis = rotation @ np.asarray(model.attributes["major_axis_xy"])
            objects.append({
                "object_id": object_id,
                "label": f"{label} object" if label else "observed_object",
                **{key: value for key, value in model.attributes.items() if key != "major_axis_xy"},
                "major_axis_xy": [float(axis[0]), float(axis[1])],
                "centroid_m": model.position.tolist(),
                "yaw_since_first_seen_rad": float(model.yaw),
                "track_residual_m": round(float(model.residual), 5),
                "bbox_xyxy": [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())],
                "visible_pixels": int(mask.sum()),
                "manipulable": True,
                "footprint_xy": [[round(float(v), 4) for v in point] for point in self._footprint(model)],
            })
            for index, (item, position) in enumerate(model.keypoint_positions()):
                u, v = self._pixel(position, depth.shape)
                entry = {"keypoint_id": f"{object_id}.kp_{index:02d}", "object_id": object_id, "pixel_uv": [u, v], "position_m": position, "confidence": float(item["confidence"]), "kind": item["kind"], "description": item["description"]}
                if "axis" in item:
                    entry["axis"] = item["axis"]
                keypoints.append(entry)
                color = (255, 255, 0) if index == 0 else (0, 255, 255)
                radius = 4 if index == 0 else 2
                draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=color, width=2)
            x0, y0 = int(columns.min()), int(rows.min())
            draw.text((x0, max(0, y0 - 12)), f"{object_id} {label or ''}", fill=(255, 255, 0))
        self._update_regions(region_masks, cloud, valid)
        regions, region_keypoints = self._region_entries(depth)
        if not regions:
            raise RuntimeError("segmentation produced no target region")
        for item in region_keypoints:
            u, v = self._pixel(item["position_m"], depth.shape)
            item["pixel_uv"] = [u, v]
            keypoints.append(item)
            draw.ellipse((u - 2, v - 2, u + 2, v + 2), outline=(255, 0, 255), width=2)
            if item["kind"] == "region_surface_center":
                draw.text((u + 5, v + 2), item["region_id"], fill=(255, 0, 255))
        rgb_digest = "sha256:" + hashlib.sha256(rgb.tobytes()).hexdigest()
        depth_digest = "sha256:" + hashlib.sha256(depth.tobytes()).hexdigest()
        calibration_digest = digest(self.config["camera"])
        core = {
            "schema_version": SNAPSHOT_SCHEMA,
            "session_id": session_id,
            "scene_revision": revision,
            "timestamp_ns": timestamp_ns(),
            "frame_id": self.config.get("frame_id", "nova2_base"),
            "rgb_digest": rgb_digest,
            "depth_digest": depth_digest,
            "calibration_digest": calibration_digest,
            "objects": objects,
            "keypoints": keypoints,
            "regions": regions,
            "overlay_ref": "",
            "robot_state": robot_state,
            "model": {"name": self.model.config["model_name"], "weights_digest": self.model.weights_digest},
        }
        core["observation_id"] = digest({key: value for key, value in core.items() if key != "overlay_ref"})
        directory = runtime_root() / "run" / "rekep" / "snapshots" / core["observation_id"].split(":", 1)[1]
        directory.mkdir(parents=True, exist_ok=True)
        overlay_path = directory / "keypoints.jpg"
        overlay.save(overlay_path, quality=92)
        core["overlay_ref"] = str(overlay_path)
        validate_snapshot(core)
        atomic_json(runtime_root() / "run" / "rekep" / "snapshots" / f"{core['observation_id'].split(':', 1)[1]}.json", core)
        return core
