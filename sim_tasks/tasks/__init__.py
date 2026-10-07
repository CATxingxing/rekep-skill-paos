"""Task registry."""
from __future__ import annotations

from ..config import load_config
from .general_pickup import GeneralPickupEnv
from .push_t import PushTEnv
from .stack_blocks import StackBlocksEnv

TASKS = {cls.TASK: cls for cls in (GeneralPickupEnv, StackBlocksEnv, PushTEnv)}


def make_env(task: str, config: dict | None = None):
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; available: {sorted(TASKS)}")
    return TASKS[task](config or load_config())
