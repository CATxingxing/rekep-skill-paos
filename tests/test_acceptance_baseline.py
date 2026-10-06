import importlib.util
from pathlib import Path

import pytest

from helpers import program, snapshot

spec = importlib.util.spec_from_file_location("acceptance_baseline", Path(__file__).resolve().parents[1] / "scripts/acceptance_baseline.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


@pytest.mark.parametrize("phase", ["completed", "unknown", "failed"])
def test_single_admission_and_immediate_status_even_when_unknown(tmp_path, monkeypatch, phase):
    gateway = acceptance.Gateway("http://example.invalid", tmp_path)
    calls = []
    def request(path, body=None):
        calls.append((path, body))
        if body is not None:
            return {"invocation_id": "inv-1", "phase": "dispatching"}
        return {"phase": phase, "result": {"status": "succeeded" if phase == "completed" else "failed"}}
    monkeypatch.setattr(gateway, "request", request)
    monkeypatch.setattr(acceptance.time, "sleep", lambda _: pytest.fail("must query immediately and stop on terminal"))
    result = gateway.action("rekep.execute_task", {}, "execute")
    assert result["phase"] == phase
    assert len([body for _, body in calls if body is not None]) == 1
    assert calls[1][0] == "/invocations/inv-1"


def test_visual_verification_requires_the_bound_object_in_region():
    before = snapshot()
    after = snapshot()
    assert not acceptance.placement_evidence(program(), before, after)["passed"]
    after["keypoints"][0]["position_m"] = [0., -.4, .05]
    assert acceptance.placement_evidence(program(), before, after)["passed"]
    after["keypoints"][0]["object_id"] = "object_001"
    assert not acceptance.placement_evidence(program(), before, after)["passed"]
