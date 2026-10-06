from __future__ import annotations

from pathlib import Path


def test_perception_wrapper_does_not_shadow_upstream_dinov2_package():
    root = Path(__file__).resolve().parents[1]
    perception = root / "nodes/rekep-perception"
    assert not (perception / "dinov2.py").exists()
    assert (perception / "dinov2_runtime.py").is_file()
    assert "from dinov2_runtime import DinoV2" in (perception / "perception.py").read_text(encoding="utf-8")


def test_node_builder_smoke_tests_executable_before_archiving():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "build_nodes.py").read_text(encoding="utf-8")

    assert '[str(executable), "--help"]' in source
    assert "timeout=60" in source
