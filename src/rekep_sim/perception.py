"""Perception: reference DINOv2 keypoint proposal on the MuJoCo env, with object labels."""

from __future__ import annotations

import hashlib
import json
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


def perceive(env, *, proposer_config: dict | None = None, seed: int = 0) -> tuple[dict, Any, np.ndarray]:
    cfg = dict(DEFAULT_PROPOSER_CONFIG)
    if proposer_config:
        cfg.update(proposer_config)
    cfg["bounds_min"] = env.bounds_min.tolist()
    cfg["bounds_max"] = env.bounds_max.tolist()
    cfg["seed"] = seed

    obs = env.get_cam_obs()[0]
    proposer = refimpl.build_keypoint_proposer(cfg)
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
