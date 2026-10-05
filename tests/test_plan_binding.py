from __future__ import annotations

import pytest

from helpers import program, snapshot
from rekep_core.contracts import ContractError, verify_binding


def test_every_binding_field_is_fail_closed():
    snap = snapshot()
    value = program(snap)
    arguments = {"session_id": value["session_id"], "scene_revision": value["scene_revision"], "observation_id": value["observation_id"], "plan_id": value["plan_id"], "plan_digest": value["plan_digest"]}
    verify_binding(value, **arguments)
    replacements = {"session_id": "other", "scene_revision": 99, "observation_id": "sha256:" + "9" * 64, "plan_id": "other", "plan_digest": "sha256:" + "8" * 64}
    for field, replacement in replacements.items():
        changed = dict(arguments)
        changed[field] = replacement
        with pytest.raises(ContractError, match=field.replace("plan_digest", "plan_digest|digest")):
            verify_binding(value, **changed)
