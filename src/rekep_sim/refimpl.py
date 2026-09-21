"""Glue to run the reference (upstream) algorithm modules with local assets.

The reference plugin's simulation line is imported **verbatim** from
``reference/rekep-real-plugin/runtime`` (a pinned submodule). This module only:

* puts that directory on ``sys.path``;
* redirects ``torch.hub.load('facebookresearch/dinov2', ...)`` to the locally
  staged repo + checkpoint (so no network and no implicit download);
* exposes small builders used by nodes and the parity harness.

No reference source file is modified.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_RUNTIME = REPO_ROOT / "reference" / "rekep-real-plugin" / "runtime"

DINOV2_HUB_NAMES = {"facebookresearch/dinov2", "dinov2"}


def ensure_reference_on_path() -> Path:
    if not REFERENCE_RUNTIME.is_dir():
        raise RuntimeError(
            f"reference runtime not found at {REFERENCE_RUNTIME}; "
            "run `git submodule update --init --recursive`"
        )
    path = str(REFERENCE_RUNTIME)
    if path not in sys.path:
        sys.path.insert(0, path)
    return REFERENCE_RUNTIME


def _dinov2_config() -> dict[str, str]:
    repo = os.environ.get("REKEP_DINOV2_REPO", "").strip()
    weights = os.environ.get("REKEP_DINOV2_WEIGHTS", "").strip()
    model = os.environ.get("REKEP_DINOV2_MODEL", "dinov2_vits14").strip()
    if not repo or not weights:
        raise RuntimeError(
            "REKEP_DINOV2_REPO and REKEP_DINOV2_WEIGHTS must point to the local "
            "DINOv2 source and checkpoint"
        )
    return {"repo_path": repo, "weights_path": weights, "model_name": model}


def load_local_dinov2(*_args: Any, **_kwargs: Any):
    """torch.hub.load replacement that loads the local DINOv2 checkpoint."""
    import torch

    cfg = _dinov2_config()
    model = torch.hub.load(
        cfg["repo_path"], cfg["model_name"], source="local", pretrained=False
    )
    state = torch.load(cfg["weights_path"], map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    if not isinstance(state, dict):
        raise RuntimeError("DINOv2 checkpoint does not contain a state dictionary")
    cleaned = {
        str(key).removeprefix("module.").removeprefix("backbone."): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(missing) > 32 or len(unexpected) > 32:
        raise RuntimeError(
            f"checkpoint incompatible: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return model


def patch_torch_hub_local() -> None:
    """Make torch.hub.load resolve DINOv2 locally (idempotent)."""
    import torch

    current = torch.hub.load
    if getattr(current, "_rekep_local_patch", False):
        return
    original = current

    def load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir) in DINOV2_HUB_NAMES:
            return load_local_dinov2(repo_or_dir, model, *args, **kwargs)
        return original(repo_or_dir, model, *args, **kwargs)

    load._rekep_local_patch = True  # type: ignore[attr-defined]
    load._rekep_original = original  # type: ignore[attr-defined]
    torch.hub.load = load  # type: ignore[assignment]


def build_keypoint_proposer(config: dict[str, Any]):
    ensure_reference_on_path()
    patch_torch_hub_local()
    import keypoint_proposal  # from the reference runtime
    _patch_kmeans(keypoint_proposal)

    return keypoint_proposal.KeypointProposer(config)


def _patch_kmeans(keypoint_proposal_module: Any) -> None:
    """Harden the reference kmeans call without editing the reference source.

    ``kmeans_pytorch.kmeans`` loops ``while True`` until ``center_shift**2 < tol``;
    with NaN input that condition is never true, so the reference hangs forever.
    We (a) silence its tqdm chatter and (b) fail fast on non-finite input. The
    reference call site (``keypoint_proposal.kmeans``) is rebound to the guarded
    function; the upstream file on disk is unchanged.
    """
    import kmeans_pytorch

    if getattr(kmeans_pytorch.kmeans, "_rekep_guarded", False):
        return

    class _SilentTqdm:
        def set_postfix(self, *a, **k):
            return None

        def update(self, *a, **k):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    kmeans_pytorch.tqdm = lambda *a, **k: _SilentTqdm()

    original = kmeans_pytorch.kmeans

    def guarded(X, *args, **kwargs):
        import torch

        if torch.is_tensor(X) and not torch.isfinite(X).all():
            raise RuntimeError(
                "kmeans input contains non-finite values; refusing to loop"
            )
        return original(X, *args, **kwargs)

    guarded._rekep_guarded = True  # type: ignore[attr-defined]
    kmeans_pytorch.kmeans = guarded
    keypoint_proposal_module.kmeans = guarded
