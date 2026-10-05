from __future__ import annotations

from pathlib import Path


def test_perception_wrapper_does_not_shadow_upstream_dinov2_package():
    root = Path(__file__).resolve().parents[1]
    perception = root / "nodes/rekep-perception"
    assert not (perception / "dinov2.py").exists()
    assert (perception / "dinov2_runtime.py").is_file()
    assert "from dinov2_runtime import DinoV2" in (perception / "perception.py").read_text(encoding="utf-8")
