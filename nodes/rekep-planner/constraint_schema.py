from __future__ import annotations

import uuid
from typing import Any

from rekep_core.contracts import PROGRAM_SCHEMA, validate_program
from rekep_core.ids import digest


def bind_program(stages: Any, *, instruction: str, snapshot: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
    if not isinstance(stages, list):
        raise ValueError("VLM response must contain a stages array")
    program = {
        "schema_version": PROGRAM_SCHEMA,
        "plan_id": f"plan_{uuid.uuid4().hex}",
        "session_id": snapshot["session_id"],
        "scene_revision": snapshot["scene_revision"],
        "observation_id": snapshot["observation_id"],
        "instruction": instruction,
        "instruction_digest": digest(instruction),
        "frame_id": snapshot["frame_id"],
        "units": {"length": "m", "angle": "rad", "quaternion": "xyzw"},
        "generator": {"mode": "vlm", "provider": provider, "model": model},
        "stages": stages,
    }
    program["plan_digest"] = digest(program)
    return validate_program(program, snapshot)
