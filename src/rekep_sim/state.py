"""Shared filesystem state for the ReKep Skill nodes (perception -> plan -> execute)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def state_dir() -> Path:
    explicit = os.environ.get("REKEP_SIM_STATE_DIR")
    if explicit:
        path = Path(explicit).expanduser()
    else:
        home = os.environ.get("PAOS_HOME", str(Path.home() / ".PhyAgentOS"))
        path = Path(home) / "run" / "rekep-sim"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scene_path() -> Path:
    explicit = os.environ.get("REKEP_SIM_SCENE")
    if explicit:
        return Path(explicit).expanduser()
    root = os.environ.get("PAOS_SKILL_ROOT")
    if root:
        return (
            Path(root)
            / "assets/dobot-nova2-robotiq/mjcf/dobot_nova2_robotiq_2f85_pick_place.xml"
        )
    raise RuntimeError("REKEP_SIM_SCENE or PAOS_SKILL_ROOT must be set")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
