"""Load tasks.yaml with optional nested overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "tasks.yaml"
BASE_SCENE = REPO_ROOT / "assets" / "dobot-nova2-robotiq" / "mjcf" / "dobot_nova2_robotiq_2f85_pick_place.xml"
GENERATED_DIR = PACKAGE_DIR / "generated"


def deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    config = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if overrides:
        deep_update(config, overrides)
    return config
