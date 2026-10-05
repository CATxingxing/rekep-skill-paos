from __future__ import annotations

import hashlib
import importlib
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from dinov2_runtime import DinoV2
from rekep_core.contracts import SNAPSHOT_SCHEMA, atomic_json, runtime_root, validate_snapshot
from rekep_core.geometry import transform_point
from rekep_core.ids import canonical_json, digest, timestamp_ns


def _components(mask: Any, minimum: int) -> list[Any]:
    import numpy as np

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    result = []
    for start_v, start_u in np.argwhere(mask):
        if visited[start_v, start_u]:
            continue
        stack, pixels = [(int(start_v), int(start_u))], []
        visited[start_v, start_u] = True
        while stack:
            v, u = stack.pop()
            pixels.append((v, u))
            for dv, du in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nv, nu = v + dv, u + du
                if 0 <= nv < height and 0 <= nu < width and mask[nv, nu] and not visited[nv, nu]:
                    visited[nv, nu] = True
                    stack.append((nv, nu))
        if len(pixels) >= minimum:
            component = np.zeros_like(mask, dtype=bool)
            rows, columns = zip(*pixels)
            component[list(rows), list(columns)] = True
            result.append(component)
    return result


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
        self._tracks: dict[str, list[float]] = {}
        self._next_track = 0
        self._lock = threading.RLock()

    def update_rgb(self, value: Any) -> None:
        from forge_msgs import Image

        image = Image.from_arrow(value)
        if image.encoding not in {"rgb8", "bgr8"}:
            raise RuntimeError(f"RGB encoding must be rgb8 or bgr8, got {image.encoding}")
        frame = image.to_numpy()
        with self._lock:
            self._rgb = frame[..., ::-1].copy() if image.encoding == "bgr8" else frame.copy()
            self._rgb_at = time.monotonic()

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
        episode = status.get("episode_index")
        if isinstance(episode, bool) or not isinstance(episode, int):
            return
        with self._lock:
            if self._episode is not None and episode != self._episode:
                self.scene_revision += 1
                self.session_id = f"session_{uuid.uuid4().hex}"
                self._tracks.clear()
                self._next_track = 0
            self._episode = episode

    def readiness(self) -> dict[str, Any]:
        age = float(self.config.get("freshness", {}).get("max_age_ms", 1500)) / 1000.0
        now = time.monotonic()
        return {"rgb": self._rgb is not None and now - self._rgb_at <= age, "depth": self._depth is not None and now - self._depth_at <= age, "robot_state": bool(self._robot_state)}

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

    def _track(self, position: list[float]) -> str:
        threshold = float(self.config.get("tracker", {}).get("max_reassociation_distance_m", 0.08))
        if self._tracks:
            identifier, separation = min(((identifier, math.dist(position, old)) for identifier, old in self._tracks.items()), key=lambda item: item[1])
            if separation <= threshold:
                self._tracks[identifier] = position
                return identifier
        identifier = f"object_{self._next_track:03d}"
        self._next_track += 1
        self._tracks[identifier] = position
        return identifier

    def _masks(self, rgb: Any) -> tuple[list[Any], list[Any]]:
        import numpy as np

        segmentation = self.config.get("segmentation", {})
        minimum = int(segmentation.get("minimum_component_pixels", 100))
        backend = segmentation.get("backend")
        if backend == "module":
            module = importlib.import_module(segmentation["module"])
            masks, regions = module.segment(rgb, dict(segmentation))
            return [np.asarray(item, dtype=bool) for item in masks], [np.asarray(item, dtype=bool) for item in regions]
        if backend != "sim_color_components":
            raise RuntimeError("a supported instance-segmentation backend is required")
        red, green, blue = (rgb[..., index].astype(float) for index in range(3))
        saturation = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
        foreground = (saturation > 45) & (np.maximum.reduce([red, green, blue]) > 90)
        region_mask = (green > red * 1.25) & (green > blue * 1.25) & foreground
        return _components(foreground & ~region_mask, minimum), _components(region_mask, minimum)

    def capture(self) -> dict[str, Any]:
        with self._lock:
            ready = self.readiness()
            if not ready["rgb"] or not ready["depth"]:
                raise RuntimeError(f"fresh RGB-D is unavailable: {ready}")
            rgb, depth = self._rgb.copy(), self._depth.copy()
            session_id, revision, robot_state = self.session_id, self.scene_revision, dict(self._robot_state)
        import numpy as np
        from PIL import Image as PilImage, ImageDraw

        masks, region_masks = self._masks(rgb)
        if not masks:
            raise RuntimeError("segmentation produced no manipulable object")
        features = self.model.feature_map(rgb)
        overlay = PilImage.fromarray(rgb)
        draw = ImageDraw.Draw(overlay)
        objects, keypoints = [], []
        for mask in masks:
            rows, columns = np.where(mask)
            center_u, center_v = int(np.median(columns)), int(np.median(rows))
            center_u, center_v, centroid = self._mask_point(center_u, center_v, depth, mask)
            object_id = self._track(centroid)
            objects.append({"object_id": object_id, "label": "observed_object", "centroid_m": centroid, "bbox_xyxy": [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())], "manipulable": True})
            proposals = self.model.propose(features, mask, int(self.config.get("keypoints", {}).get("per_object", 3)))
            for index, (u, v, confidence) in enumerate(proposals):
                if not mask[v, u]:
                    u, v = center_u, center_v
                u, v, position = self._mask_point(u, v, depth, mask)
                keypoint_id = f"{object_id}.kp_{index:02d}"
                keypoints.append({"keypoint_id": keypoint_id, "object_id": object_id, "pixel_uv": [u, v], "position_m": position, "confidence": confidence})
                draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(0, 255, 255), width=2)
                draw.text((u + 7, v - 6), keypoint_id, fill=(0, 255, 255))
        regions = []
        for index, mask in enumerate(region_masks):
            rows, columns = np.where(mask)
            points = []
            for sample in np.linspace(0, len(rows) - 1, min(32, len(rows)), dtype=int):
                try:
                    points.append(self._point(int(columns[sample]), int(rows[sample]), depth))
                except RuntimeError:
                    pass
            if points:
                array = np.asarray(points)
                lower, upper = array.min(axis=0), array.max(axis=0)
                upper[2] += float(self.config.get("targets", {}).get("volume_height_m", 0.10))
                regions.append({"region_id": f"region_{index:03d}", "label": "observed_target_region", "bounds_min_m": lower.tolist(), "bounds_max_m": upper.tolist()})
        if not regions:
            raise RuntimeError("segmentation produced no target region")
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
