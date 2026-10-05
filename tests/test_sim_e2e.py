from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_skill_source_is_minimal_and_profile_is_connected():
    assert sorted(path.name for path in (ROOT / "skill-src/rekep").iterdir()) == ["SKILL.md", "skill.yaml"]
    flow = yaml.safe_load((ROOT / "profiles/sim-dobot-nova2-robotiq/dataflow.yaml").read_text(encoding="utf-8"))
    nodes = {item["id"]: item for item in flow["nodes"]}
    assert len(nodes) == 8
    assert sum(identifier == "mujoco_sim" for identifier in nodes) == 1
    assert nodes["planner"]["inputs"]["snapshot_in"] == "perception/snapshot_out"
    assert nodes["executor"]["inputs"]["constraint_in"] == "planner/constraint_out"
    assert nodes["executor"]["inputs"]["snapshot_in"] == "perception/snapshot_out"


def test_no_rekep_node_creates_simulator_state():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "nodes").rglob("*.py"))
    for forbidden in ("MjModel", "MjData", "data.qpos", "import mujoco"):
        assert forbidden not in source


def test_no_template_or_generated_python_path():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "nodes").rglob("*.*") if path.suffix in {".py", ".txt"})
    assert "fixture_" + "constraints" not in source
    assert "source:" + " template" not in source
    assert "exec(" not in source
