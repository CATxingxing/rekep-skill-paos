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

    # Cap torch threads: on a many-core host the default (all cores) causes severe
    # oversubscription and ~10x slower CPU inference.
    try:
        import torch

        torch.set_num_threads(int(os.environ.get("REKEP_TORCH_THREADS", "4")))
    except Exception:
        pass

    proposer = keypoint_proposal.KeypointProposer(config)
    # The reference constructs MeanShift(n_jobs=32); on a loaded many-core host
    # the joblib thread pool costs ~6s per call (vs 0.3s single-threaded). Force
    # single-threaded; the clustering is over a handful of candidate points.
    try:
        proposer.mean_shift.n_jobs = 1
    except Exception:
        pass
    return proposer


def _patch_kmeans(keypoint_proposal_module: Any) -> None:
    """Harden the reference kmeans call without editing the reference source.

    ``kmeans_pytorch.kmeans`` loops ``while True`` until ``center_shift**2 < tol``.
    With NaN input, or when the assignment oscillates, that condition is never
    met and the reference hangs forever (observed: a 30-minute hang in the
    perception node). We replace the call site with a bounded re-implementation
    that has identical clustering semantics (same init + pairwise distance +
    update) but a hard iteration cap and a non-finite guard.
    """
    import kmeans_pytorch

    if getattr(kmeans_pytorch.kmeans, "_rekep_bounded", False):
        return

    def bounded_kmeans(X, num_clusters, distance="euclidean", tol=1e-4,
                       device=None, max_iter=100, **_ignored):
        import numpy as np
        import torch

        device = device or torch.device("cpu")
        if distance == "euclidean":
            pairwise = kmeans_pytorch.pairwise_distance
        elif distance == "cosine":
            pairwise = kmeans_pytorch.pairwise_cosine
        else:
            raise NotImplementedError
        X = X.float().to(device)
        if not torch.isfinite(X).all():
            raise RuntimeError("kmeans input contains non-finite values")
        num_samples = int(X.shape[0])
        k = int(min(num_clusters, num_samples))
        idx = np.random.choice(num_samples, k, replace=False)
        state = X[idx].clone()
        choice = torch.zeros(num_samples, dtype=torch.long, device=device)
        for _ in range(max(1, int(max_iter))):
            dis = pairwise(X, state)
            choice = torch.argmin(dis, dim=1)
            prev = state.clone()
            for cluster in range(k):
                selected = torch.nonzero(choice == cluster).squeeze(1).to(device)
                if selected.numel() == 0:
                    continue
                state[cluster] = X.index_select(0, selected).mean(dim=0)
            center_shift = torch.sum(torch.sqrt(torch.sum((state - prev) ** 2, dim=1)))
            if bool(torch.isfinite(center_shift)) and float(center_shift) ** 2 < tol:
                break
        return choice.cpu(), state.cpu()

    bounded_kmeans._rekep_bounded = True  # type: ignore[attr-defined]
    kmeans_pytorch.kmeans = bounded_kmeans
    keypoint_proposal_module.kmeans = bounded_kmeans
