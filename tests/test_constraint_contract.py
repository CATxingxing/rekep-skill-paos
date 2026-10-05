from __future__ import annotations

import pytest

from helpers import program, snapshot
from rekep_core.contracts import ContractError, validate_program
from rekep_core.ids import digest


def test_restricted_program_validates():
    snap = snapshot()
    assert validate_program(program(snap), snap)["generator"]["mode"] == "vlm"


def test_generated_python_operator_is_rejected():
    snap = snapshot()
    value = program(snap)
    value["stages"][0]["subgoal_constraints"][0]["expression"] = {"op": "python_eval", "source": "pass"}
    value["plan_digest"] = digest({key: item for key, item in value.items() if key != "plan_digest"})
    with pytest.raises(ContractError, match="unsupported constraint"):
        validate_program(value, snap)


def test_non_vlm_generator_is_rejected():
    snap = snapshot()
    value = program(snap)
    value["generator"] = {"mode": "template"}
    value["plan_digest"] = digest({key: item for key, item in value.items() if key != "plan_digest"})
    with pytest.raises(ContractError, match="must be vlm"):
        validate_program(value, snap)
