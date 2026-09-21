"""Perception: reference DINOv2 keypoint proposal on the MuJoCo env, with object labels."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from . import refimpl

DEFAULT_PROPOSER_CONFIG = {
    "num_candidates_per_mask": 5,
    "min_dist_bt_keypoints": 0.06,
    "max_mask_ratio": 0.5,
    "device": "cpu",
    "seed": 0,
}


def _nearest_object(env, point: np.ndarray) -> str:
    best_name, best_dist = None, np.inf
    for name, gid in zip(env.objects.keys(), env.objects.values()):
        # env.objects maps geom-name -> label; find the geom by name
        import mujoco

        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        d = float(np.linalg.norm(env.data.geom_xpos[geom_id] - point))
        if d < best_dist:
            best_name, best_dist = name, d
    # normalize geom names to semantic roles
    return {
        "pick_cube_geom": "pick_cube",
        "place_zone": "place_zone",
    }.get(best_name, best_name)


_PROPOSER_CACHE: dict[str, Any] = {}
_PROPOSER_LOCK = threading.Lock()


def _proposer_for(env, proposer_config: dict | None, seed: int):
    """Build (and cache) the reference keypoint proposer once per process."""
    cfg = dict(DEFAULT_PROPOSER_CONFIG)
    if proposer_config:
        cfg.update(proposer_config)
    cfg["bounds_min"] = env.bounds_min.tolist()
    cfg["bounds_max"] = env.bounds_max.tolist()
    cfg["seed"] = seed
    key = repr(sorted(cfg.items()))
    with _PROPOSER_LOCK:
        cached = _PROPOSER_CACHE.get(key)
        if cached is None:
            cached = refimpl.build_keypoint_proposer(cfg)
            _PROPOSER_CACHE[key] = cached
    return cached


def warmup(env, *, proposer_config: dict | None = None, seed: int = 0) -> None:
    """Load the DINOv2 model and run one inference on synthetic data.

    Uses no simulator rendering, so it is safe to run in a background thread
    concurrently with the provider registering (keeps node readiness fast while
    the model warms up).
    """
    proposer = _proposer_for(env, proposer_config, seed)
    h = w = 64
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[16:48, 16:48] = np.array([220, 60, 40], dtype=np.uint8)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[..., 0] = np.linspace(env.bounds_min[0], env.bounds_max[0], w)[None, :]
    points[..., 1] = np.linspace(env.bounds_min[1], env.bounds_max[1], h)[:, None]
    points[..., 2] = 0.8
    seg = np.zeros((h, w), dtype=np.int32)
    seg[16:48, 16:48] = 1
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    try:
        proposer.get_keypoints(rgb, points, seg)
    except Exception:
        # a synthetic warmup may find no candidates; the model is still loaded
        pass


def perceive(env, *, proposer_config: dict | None = None, seed: int = 0) -> tuple[dict, Any, np.ndarray]:
    cfg = dict(DEFAULT_PROPOSER_CONFIG)
    if proposer_config:
        cfg.update(proposer_config)
    cfg["bounds_min"] = env.bounds_min.tolist()
    cfg["bounds_max"] = env.bounds_max.tolist()
    cfg["seed"] = seed

    obs = env.get_cam_obs()[0]
    proposer = _proposer_for(env, proposer_config, seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    kp, projected, meta = proposer.get_keypoints(
        obs["rgb"], obs["points"], obs["seg"], return_metadata=True
    )
    keypoints = []
    pixels = meta["candidate_pixels"]
    for i, p in enumerate(kp):
        keypoints.append(
            {
                "index": i,
                "position_m": [float(x) for x in p],
                "pixel_uv": [int(pixels[i][1]), int(pixels[i][0])],
                "object": _nearest_object(env, p),
            }
        )
    rgb_sha = hashlib.sha256(obs["rgb"].tobytes()).hexdigest()
    observation_id = "sha256:" + hashlib.sha256(
        json.dumps({"rgb": rgb_sha, "kps": [k["position_m"] for k in keypoints]},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    snapshot = {
        "schema_version": "rekep.perception.v1",
        "observation_id": observation_id,
        "rgb_sha256": rgb_sha,
        "bounds_min": env.bounds_min.tolist(),
        "bounds_max": env.bounds_max.tolist(),
        "keypoints": keypoints,
        "rgb": obs["rgb"],
        "projected": projected,
    }
    return snapshot, projected, kp
