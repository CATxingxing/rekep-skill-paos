"""R1: reference keypoint proposal runs locally with DINOv2 and is deterministic."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get("PAOS_REKEP_ROOT", "/data7/home/linjiongxiao/.paos-rekep"))
os.environ.setdefault("REKEP_DINOV2_REPO", str(ROOT / "models" / "dinov2"))
os.environ.setdefault("REKEP_DINOV2_WEIGHTS", str(ROOT / "models" / "weights" / "dinov2_vits14.pth"))
os.environ.setdefault("REKEP_DINOV2_MODEL", "dinov2_vits14")

from rekep_sim import refimpl  # noqa: E402

BOUNDS_MIN = np.array([-0.45, -0.75, 0.698], dtype=float)
BOUNDS_MAX = np.array([0.10, 0.60, 1.2], dtype=float)


def _synthetic_scene(seed: int = 0, h: int = 480, w: int = 640):
    rng = np.random.default_rng(seed)
    rgb = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    # give the masks distinct colors so feature clustering has signal
    rgb[100:250, 150:350] = np.array([220, 60, 40], dtype=np.uint8)
    rgb[300:400, 400:600] = np.array([40, 200, 70], dtype=np.uint8)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[..., 0] = np.linspace(BOUNDS_MIN[0], BOUNDS_MAX[0], w)[None, :]
    points[..., 1] = np.linspace(BOUNDS_MIN[1], BOUNDS_MAX[1], h)[:, None]
    # non-constant z (a real depth image varies); constant z makes the reference
    # feature-space normalization divide by zero (0/0 -> NaN -> kmeans hangs).
    zx = np.linspace(0.0, 1.0, w)[None, :]
    zy = np.linspace(0.0, 1.0, h)[:, None]
    points[..., 2] = (0.80 + 0.10 * zx + 0.05 * zy).astype(np.float32)
    masks = np.zeros((h, w), dtype=np.int32)
    masks[100:250, 150:350] = 1
    masks[300:400, 400:600] = 2
    return rgb, points, masks


def _config() -> dict:
    return {
        "num_candidates_per_mask": 5,
        "min_dist_bt_keypoints": 0.06,
        "max_mask_ratio": 0.5,
        "device": "cpu",
        "bounds_min": BOUNDS_MIN.tolist(),
        "bounds_max": BOUNDS_MAX.tolist(),
        "seed": 0,
    }


def _reseed(seed: int = 0) -> None:
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - gpu only
        torch.cuda.manual_seed(seed)


@pytest.mark.slow
def test_keypoint_proposal_runs_and_is_deterministic():
    rgb, points, masks = _synthetic_scene()
    proposer = refimpl.build_keypoint_proposer(_config())
    # The reference reseeds only in __init__; kmeans uses the global RNG, so a
    # caller must reseed before each call to get reproducible keypoints.
    _reseed(0)
    kp1, proj1, meta1 = proposer.get_keypoints(rgb, points, masks, return_metadata=True)
    _reseed(0)
    kp2, proj2, meta2 = proposer.get_keypoints(rgb, points, masks, return_metadata=True)

    assert kp1.ndim == 2 and kp1.shape[1] == 3
    assert len(kp1) > 0, "expected at least one keypoint candidate"
    assert proj1.shape == rgb.shape
    # within workspace bounds
    assert np.all(kp1 >= BOUNDS_MIN - 1e-6) and np.all(kp1 <= BOUNDS_MAX + 1e-6)
    # deterministic for identical input
    np.testing.assert_allclose(kp1, kp2)
    np.testing.assert_array_equal(meta1["candidate_pixels"], meta2["candidate_pixels"])
    np.testing.assert_array_equal(proj1, proj2)
