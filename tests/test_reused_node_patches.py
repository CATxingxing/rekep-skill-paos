from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_rebuild_module():
    path = ROOT / "scripts/rebuild_reused_nodes.py"
    spec = importlib.util.spec_from_file_location("rebuild_reused_nodes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gateway_patch_is_registered_and_content_addressable() -> None:
    module = _load_rebuild_module()
    patches = module.SOURCE_PATCHES["gateway"]
    assert len(patches) == 1
    patch = patches[0]
    assert patch.is_file()
    text = patch.read_text(encoding="utf-8")
    assert "if not item.accepted_established:" in text
    assert "test_action_lookup_waits_for_invoke_acceptance_barrier" in text
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert len(digest) == 64
