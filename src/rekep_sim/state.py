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


def load_service_env() -> None:
    """Load an external env file (secrets stay out of the Skill bundle).

    Dora sets each node's environment from the rendered dataflow, so values
    exported in the launching shell (e.g. REKEP_VLM_API_KEY) do not reach the
    nodes. Read them from ``$REKEP_SIM_ENV_FILE`` or
    ``$PAOS_HOME/run/rekep-sim/service.env`` at startup instead.
    """
    path = os.environ.get("REKEP_SIM_ENV_FILE")
    if not path:
        home = os.environ.get("PAOS_HOME", str(Path.home() / ".PhyAgentOS"))
        path = str(Path(home) / "run" / "rekep-sim" / "service.env")
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return
    for line in candidate.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def scene_path() -> Path:
    explicit = os.environ.get("REKEP_SIM_SCENE")
    if explicit:
        return Path(explicit).expanduser()
    root = os.environ.get("PAOS_SKILL_ROOT")
    if root:
        return (
            Path(root)
            / "assets/franka-panda/tabletop_panda.xml"
        )
    raise RuntimeError("REKEP_SIM_SCENE or PAOS_SKILL_ROOT must be set")


def env_kwargs(scene: Path) -> dict:
    """Per-scene camera/workspace defaults (Panda vs the legacy Dobot demo)."""
    name = scene.name.lower()
    if "panda" in name:
        return {
            "bounds_min": (0.10, -0.50, 0.0),
            "bounds_max": (0.85, 0.45, 1.20),
            "camera": {
                "name": "vlm",
                "eye": (1.00, -0.70, 1.00),
                "target": (0.45, -0.05, 0.10),
                "fovy": 50.0,
            },
        }
    return {"bounds_min": (-0.6, -0.9, 0.0), "bounds_max": (0.7, 0.6, 1.3), "camera": None}


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
